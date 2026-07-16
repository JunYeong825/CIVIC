# from comet_ml import Experiment
import time
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
# import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR


from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset, CIRCODataset, MACIRDataset, GeneCISDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics

@torch.no_grad()
def blip_validate_genecis(blip_model_name, backbone, model_path, genecis_path, eval_mode, unchange_ratio):
    # 1. 모델 및 프로세서 로드
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        if blip_model.__class__.__name__ in checkpoint:
            state_dict = checkpoint[blip_model.__class__.__name__]
        else:
            state_dict = checkpoint
        blip_model.load_state_dict(state_dict, strict=False)
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode
    blip_model.unchange_ratio = unchange_ratio

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    # 2. 데이터셋 설정
    dataset = GeneCISDataset(genecis_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 3. 평가 수행
    task_results = {task: [] for task in dataset.tasks}
    
    print(f"Evaluating GeneCIS in {eval_mode} mode...")
    for ref_img, candidates, target_idx, condition, task_name in tqdm(dataloader):
        # ref_img: [1, 3, 224, 224]
        # candidates: [1, 15, 3, 224, 224]
        # target_idx: [1]
        # condition: list of 1 string
        # task_name: list of 1 string
        
        task = task_name[0]
        caption = txt_processors["eval"](condition[0])
        gt_idx = target_idx.item()
        
        ref_img = ref_img.to(device)
        candidates = candidates.squeeze(0).to(device) # [num_candidates, 3, 224, 224]
        
        with torch.no_grad():
            with blip_model.maybe_autocast():
                # Extract reference image features
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_img))
                ref_embeds = ref_embeds.float()
                
                # Extract candidate image features
                target_features, _ = blip_model.extract_target_features(candidates, mode="mean")
                target_features = F.normalize(target_features.float(), dim=-1)
            
                # Compute similarity
                sims = blip_model.inference(ref_embeds, target_features, [caption]) # [1, num_candidates]
            
        predicted_idx = torch.argmax(sims, dim=-1).item()
        task_results[task].append(1.0 if predicted_idx == gt_idx else 0.0)

    # 4. 결과 정리
    final_results = {}
    all_scores = []
    for task, scores in task_results.items():
        if scores:
            acc = np.mean(scores) * 100
            final_results[task] = acc
            all_scores.extend(scores)
    
    if all_scores:
        final_results['Average'] = np.mean(all_scores) * 100
    
    print("\n" + "="*50)
    print(f"--- GeneCIS Validation Results (Mode: {eval_mode}) ---")
    print("="*50)
    print(json.dumps(final_results, indent=4))
    print("="*50)
    
    return final_results

def blip_validate_macir(blip_model_name, backbone, model_path, json_path, img_dir, eval_mode, unchange_ratio):
    # 1. 모델 및 프로세서 로드
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        # Check if the class name is in the checkpoint or just the state_dict
        if blip_model.__class__.__name__ in checkpoint:
            state_dict = checkpoint[blip_model.__class__.__name__]
        else:
            state_dict = checkpoint
        msg = blip_model.load_state_dict(state_dict, strict=False)
        print("Missing keys {}".format(msg.missing_keys))
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode
    blip_model.unchange_ratio = unchange_ratio
    
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    gallery_path=Path('/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/image_ids.json')
    # 2. 데이터셋 설정
    classic_dataset = MACIRDataset(gallery_path, img_dir, mode='classic', preprocess=preprocess)
    relative_dataset = MACIRDataset(json_path, img_dir, mode='relative', preprocess=preprocess)

    # 3. Index 특징 추출
    print(f"Extracting MACIR index features ({eval_mode} mode)...")
    index_features, index_names = extract_index_blip_features(classic_dataset, blip_model, save_memory=True)

    # 4. 유사도 계산 및 지표 산출
    print(f"Evaluating MACIR in {eval_mode} mode...")
    relative_val_loader = DataLoader(dataset=relative_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    all_index_features = index_features[0].to(device)
    
    distance = []
    target_names = []
    conditions_all = []
    compositions_all = []
    reference_names = []

    total_inference_time = 0
    num_queries = 0
    
    for ref_imgs, batch_target_names, captions, conditions, compositions, batch_ref_names in tqdm(relative_val_loader):
        captions = [txt_processors["eval"](caption) for caption in captions]
        ref_imgs = ref_imgs.to(device)
        
        with torch.no_grad():
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with blip_model.maybe_autocast():
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_imgs))
            ref_embeds = ref_embeds.float()
            batch_distance = blip_model.inference(ref_embeds, all_index_features, captions)
            
            torch.cuda.synchronize()
            total_inference_time += (time.perf_counter() - start_time)
            num_queries += len(batch_target_names)
            
            distance.append(batch_distance.cpu())

        target_names.extend(batch_target_names)
        conditions_all.extend(conditions)
        compositions_all.extend(compositions)
        reference_names.extend(batch_ref_names)
    
    avg_time = total_inference_time / num_queries
    print(f"\nAverage MACIR inference time per query: {avg_time:.4f}s")
    
    distance = torch.vstack(distance)
    
    # Recall 계산 함수 정의 (공식 GitHub 방식 반영)
    def calculate_recalls(sims, targets, names_list, ref_names=None):
        dists = 1 - sims
        sorted_indices = torch.argsort(dists, dim=-1).cpu()
        sorted_names = np.array(names_list)[sorted_indices]
        
        if ref_names is not None:
            # 공식 GitHub 방식: Reference 이미지를 후보군에서 제외
            reference_mask = torch.tensor(
                sorted_names != np.repeat(np.array(ref_names), len(names_list)).reshape(len(targets), -1)
            )
            # (B, N) -> (B, N-1)
            sorted_names = sorted_names[reference_mask].reshape(len(targets), -1)

        labels = torch.tensor(
            sorted_names == np.repeat(np.array(targets), sorted_names.shape[1]).reshape(len(targets), -1))
        
        return {
            'recall_at1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
            'recall_at5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
            'recall_at10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
            'recall_at50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        }

    # 1. 전체 결과
    results = calculate_recalls(distance, target_names, index_names, reference_names)
    
    # 데이터 변환
    conditions_all = np.array(conditions_all)
    compositions_all = np.array(compositions_all)
    target_names = np.array(target_names)
    reference_names = np.array(reference_names)

    # 2. Condition별 결과
    unique_conditions = ['add', 'replace', 'remove']
    per_condition_results = {}
    for cond in unique_conditions:
        mask = (conditions_all == cond)
        if mask.sum() > 0:
            per_condition_results[cond] = calculate_recalls(
                distance[mask], target_names[mask], index_names, reference_names[mask])

    # 3. Composition별 결과 (공식 논문 기준)
    unique_compositions = ['left_right', 'top_bottom', 'spatial_reasoning', 'size', 'action', 'color','object_reasoning','naive_object']
    per_composition_results = {}
    for comp in unique_compositions:
        mask = (compositions_all == comp)
        if mask.sum() > 0:
            per_composition_results[comp] = calculate_recalls(
                distance[mask], target_names[mask], index_names, reference_names[mask])

    print("\n" + "="*50)
    print(f"--- MACIR Validation Results ) ---")
    print("="*50)
    print(f"[OVERALL]\n{json.dumps(results, indent=4)}")
    
    print("\n[CONDITION-WISE]")
    for cond, res in per_condition_results.items():
        print(f"{cond.upper()}: Recall@1 = {res['recall_at1']:.2f}")

    print("\n[COMPOSITION-WISE (Official)]")
    for comp, res in per_composition_results.items():
        print(f"{comp}: Recall@1 = {res['recall_at1']:.2f}")
    print("="*50)
    
    return results

import functools
import requests

# 모든 requests 요청에서 verify를 False로 강제 고정
old_merge_environment_settings = requests.Session.merge_environment_settings

@functools.wraps(old_merge_environment_settings)
def patched_merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = patched_merge_environment_settings

def clip_finetune_fiq(val_dress_types: List[str], blip_model_name, backbone, model_path, eval_mode, unchange_ratio):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    """

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode # 모드 설정 추가
    blip_model.unchange_ratio = unchange_ratio
    
    checkpoint_path = model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if blip_model.__class__.__name__ in checkpoint:
        state_dict = checkpoint[blip_model.__class__.__name__]
    else:
        state_dict = checkpoint
    msg = blip_model.load_state_dict(state_dict, strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []


    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)



    blip_model.eval()
    recalls_at10 = []
    recalls_at50 = []

    # Compute and log validation metrics for each validation dataset (which corresponds to a different
    # FashionIQ category)
    for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                idx_to_dress_mapping):
        
        index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)
        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                            index_features, index_names, txt_processors)
        
        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)
        torch.cuda.empty_cache()

    results_dict = {}
    for i in range(len(recalls_at10)):
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
    results_dict.update({
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })

    print(f"\n--- FashionIQ Validation Results (Mode: {eval_mode}) ---")
    print(json.dumps(results_dict, indent=4))




def blip_validate_cirr(blip_model_name, backbone, blip_model_path, eval_mode, unchange_ratio):
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode # 모드 설정 추가
    blip_model.unchange_ratio = unchange_ratio
    checkpoint_path = blip_model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if blip_model.__class__.__name__ in checkpoint:
        state_dict = checkpoint[blip_model.__class__.__name__]
    else:
        state_dict = checkpoint
    msg = blip_model.load_state_dict(state_dict, strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224

    preprocess = targetpad_transform(1.25, input_dim)

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)
    # 
    results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features,
                                        val_index_names, txt_processors)
    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
    results_dict = {
        'group_recall_at1': group_recall_at1,
        'group_recall_at2': group_recall_at2,
        'group_recall_at3': group_recall_at3,
        'recall_at1': recall_at1,
        'recall_at5': recall_at5,
        'recall_at10': recall_at10,
        'recall_at50': recall_at50,
        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
        'arithmetic_mean': mean(results),
        'harmonic_mean': harmonic_mean(results),
        'geometric_mean': geometric_mean(results)
    }
    print(f"\n--- CIRR Validation Results (Mode: {eval_mode}) ---")
    print(json.dumps(results_dict, indent=4))

def compute_ap_at_k(retrieved_ids, gt_ids, k):
    """
    하나의 쿼리에 대한 Average Precision @ K 계산
    """
    if not gt_ids:
        return 0.0
    
    # K개까지만 확인
    retrieved_ids = retrieved_ids[:k]
    relevant_count = 0
    precision_sum = 0.0
    
    for i, rid in enumerate(retrieved_ids):
        if rid in gt_ids:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
            
    if relevant_count == 0:
        return 0.0
    
    # CIRCO 기준: 정답 개수와 K 중 작은 값으로 나눔
    return precision_sum / min(len(gt_ids), k)

@torch.no_grad()
def blip_validate_circo(blip_model_name, backbone, model_path, circo_path, eval_mode, unchange_ratio):
    # 1. 모델 및 프로세서 로드
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    checkpoint = torch.load(model_path, map_location=device)
    if blip_model.__class__.__name__ in checkpoint:
        state_dict = checkpoint[blip_model.__class__.__name__]
    else:
        state_dict = checkpoint
    blip_model.load_state_dict(state_dict, strict=False)
    blip_model.eval()
    blip_model.eval_mode = eval_mode # 모드 설정 추가
    blip_model.unchange_ratio = unchange_ratio
    
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    # 2. 데이터셋 설정
    classic_dataset = CIRCODataset(circo_path, split='val', mode='classic', preprocess=preprocess)
    relative_dataset = CIRCODataset(circo_path, split='val', mode='relative', preprocess=preprocess)

    # 3. Index 특징 추출 (DB 이미지들)
    print(f"Extracting CIRCO index features ({eval_mode} mode)...")
    index_features = []
    index_names = []
    dataloader = DataLoader(classic_dataset, batch_size=32, num_workers=4, pin_memory=True)

    for batch in tqdm(dataloader):
        imgs = batch['img'].to(device)
        ids = batch['img_id']
        feats, _ = blip_model.extract_target_features(imgs) 
        index_features.append(feats.cpu())
        index_names.extend(ids)
    
    index_features = torch.cat(index_features, dim=0).to(device)
    index_features = F.normalize(index_features.float(), dim=-1)

    # 4. 쿼리 수행 및 지표 계산
    print(f"Evaluating CIRCO in {eval_mode} mode...")
    k_values = [5, 10, 25, 50]
    ap_results = {k: [] for k in k_values}
    recall_results = {k: [] for k in k_values}

    total_inference_time = 0
    num_queries = 0
    
    for i in tqdm(range(len(relative_dataset))):
        sample = relative_dataset[i]
        ref_img = sample['reference_img'].unsqueeze(0).to(device)
        
        # Text Processor로 캡션 정규화
        raw_caption = sample['relative_caption']
        caption = txt_processors["eval"](raw_caption)
        
        # GT 이미지 ID 리스트 정리
        gt_ids = np.array(sample['gt_img_ids'])
        gt_ids = gt_ids[gt_ids != ''].tolist() 

        # 유사도 계산
        with torch.no_grad():
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with blip_model.maybe_autocast():
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_img))
            ref_embeds = ref_embeds.float()
            sims = blip_model.inference(ref_embeds, index_features, [caption])
            
            torch.cuda.synchronize()
            total_inference_time += (time.perf_counter() - start_time)
            num_queries += 1
            
        _, top_indices = torch.topk(sims, k=50, largest=True)
        top_indices = top_indices.squeeze().cpu().numpy()
        sorted_retrieved_ids = np.array(index_names)[top_indices]

        # mAP 계산을 위한 Boolean Mask
        map_labels = torch.tensor(np.isin(sorted_retrieved_ids, gt_ids), dtype=torch.float32)
        
        # 각 위치에서의 Precision 계산
        precisions = torch.cumsum(map_labels, dim=0) * map_labels
        precisions = precisions / torch.arange(1, len(map_labels) + 1).float()

        for k in k_values:
            ap = torch.sum(precisions[:k]) / min(len(gt_ids), k)
            ap_results[k].append(ap.item())
            recall = 1.0 if torch.sum(map_labels[:k]) > 0 else 0.0
            recall_results[k].append(recall)

    avg_time = total_inference_time / num_queries
    print(f"\nAverage FashionIQ inference time per query: {avg_time:.4f}s")
    
    final_results = {}
    for k in k_values:
        final_results[f"mAP@{k}"] = np.mean(ap_results[k]) * 100
        final_results[f"Recall@{k}"] = np.mean(recall_results[k]) * 100

    print(f"\n--- CIRCO Validation Results (Mode: {eval_mode}) ---")
    print(json.dumps(final_results, indent=4))
    return final_results

@torch.no_grad()
def generate_circo_test_submission(blip_model_name, backbone, model_path, circo_path, eval_mode, unchange_ratio, output_filename="submission.json"):
    # 1. 모델 및 프로세서 로드
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    checkpoint = torch.load(model_path, map_location=device)
    if blip_model.__class__.__name__ in checkpoint:
        state_dict = checkpoint[blip_model.__class__.__name__]
    else:
        state_dict = checkpoint
    blip_model.load_state_dict(state_dict, strict=False)
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode # 모드 설정 추가
    blip_model.unchange_ratio = unchange_ratio
    
    input_dim = 364
    preprocess = targetpad_transform(1.25, input_dim)

    # 2. 데이터셋 설정
    classic_dataset = CIRCODataset(circo_path, split='test', mode='classic', preprocess=preprocess)
    relative_dataset = CIRCODataset(circo_path, split='test', mode='relative', preprocess=preprocess)

    # 3. 모든 대상 이미지(Index)의 특징 추출
    print(f"Extracting features for CIRCO test index ({eval_mode} mode)...")
    index_features = []
    index_image_ids = []
    
    dataloader = DataLoader(classic_dataset, batch_size=32, num_workers=4, pin_memory=True)
    for batch in tqdm(dataloader):
        imgs = batch['img'].to(device)
        ids = batch['img_id']
        feats, _ = blip_model.extract_target_features(imgs)
        index_features.append(feats.cpu())
        index_image_ids.extend(ids)
    
    index_features = torch.cat(index_features, dim=0).to(device)
    index_features = F.normalize(index_features.float(), dim=-1) # 정규화 추가

    # 4. 테스트 쿼리 수행 및 결과 저장
    print(f"Processing {len(relative_dataset)} test queries in {eval_mode} mode...")
    submission_dict = {}

    for i in tqdm(range(len(relative_dataset))):
        sample = relative_dataset[i]
        ref_img = sample['reference_img'].unsqueeze(0).to(device)
        
        # Text Processor 적용 (중요!)
        raw_caption = sample['relative_caption']
        caption = txt_processors["eval"](raw_caption)
        
        query_id = sample['query_id']

        with torch.no_grad():
            with blip_model.maybe_autocast():
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_img))
            ref_embeds = ref_embeds.float()
            sims = blip_model.inference(ref_embeds, index_features, [caption])
            
        top_k = 50
        _, top_indices = torch.topk(sims, k=top_k, largest=True)
        top_indices = top_indices.squeeze().cpu().tolist()
        
        retrieved_ids = [int(index_image_ids[idx]) for idx in top_indices]
        submission_dict[str(query_id)] = retrieved_ids

    output_path = Path(output_filename.replace(".json", f"_{eval_mode}.json"))
    with open(output_path, 'w') as f:
        json.dump(submission_dict, f, indent=4)
    
    print(f"Submission file saved to: {output_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be 'CIRR', 'fashionIQ', 'CIRCO' or 'MACIR'")
    parser.add_argument("--split", type=str, default="val", help="val or test")
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain_vitL", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--circo-path", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/CIRCO")
    parser.add_argument("--genecis-path", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/GeneCIS")
    parser.add_argument("--macir-json", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/macir_meta_full.json")
    parser.add_argument("--macir-img-dir", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/images")
    parser.add_argument("--eval-mode", default="fusion", choices=["dynamic", "fusion"], help="Evaluation mode")
    parser.add_argument("--unchange-ratio", default=0.2, type=float, help="Evaluation unchange-ratio")

    args = parser.parse_args()

    if args.dataset.lower() == 'cirr':
        print(f"Running CIRR validation in {args.eval_mode} mode...")
        blip_validate_cirr(args.blip_model_name, args.backbone, args.model_path, args.eval_mode, args.unchange_ratio)
    elif args.dataset.lower() == 'fashioniq':
        print(f"Running FashionIQ validation in {args.eval_mode} mode...")
        clip_finetune_fiq(['dress', 'toptee', 'shirt'], args.blip_model_name, args.backbone, args.model_path, args.eval_mode, args.unchange_ratio)
    elif args.dataset.lower() == 'circo':
        if args.split == 'val':
            print(f"Running CIRCO validation in {args.eval_mode} mode...")
            blip_validate_circo(args.blip_model_name, args.backbone, args.model_path, args.circo_path, args.eval_mode, args.unchange_ratio)
        elif args.split == 'test':
            print(f"Generating CIRCO test submission file in {args.eval_mode} mode...")
            generate_circo_test_submission(args.blip_model_name, args.backbone, args.model_path, args.circo_path, args.unchange_ratio, args.eval_mode)
    elif args.dataset.lower() == 'macir':
        print(f"Running MACIR validation in {args.eval_mode} mode...")
        blip_validate_macir(args.blip_model_name, args.backbone, args.model_path, args.macir_json, args.macir_img_dir, args.eval_mode, args.unchange_ratio)
    elif args.dataset.lower() == 'genecis':
        print(f"Running GeneCIS validation in {args.eval_mode} mode...")
        blip_validate_genecis(args.blip_model_name, args.backbone, args.model_path, args.genecis_path, args.eval_mode, args.unchange_ratio)
    else:
        raise ValueError("Dataset should be in ['CIRR', 'FashionIQ', 'CIRCO', 'MACIR','GeneCIS']")