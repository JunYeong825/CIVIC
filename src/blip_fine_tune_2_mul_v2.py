
# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import os

# Accelerate 추가
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


from data_utils import HardNegativeBatchSampler, videoDataset, base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics

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

# 경고 메시지 끄기
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 안 쓰이는 파라미터가 있어도 무시하고 학습을 진행하도록 DDP 옵션 추가
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])
device = accelerator.device

def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                      **kwargs):

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{blip_model_name}_{training_start}")
    
    # 메인 프로세스에서만 디렉토리 생성 및 하이퍼파라미터 저장
    if accelerator.is_main_process:
        training_path.mkdir(exist_ok=True, parents=True)
        print(f"save-memory-in: {save_memory}")
        with open(training_path / "training_hyperparameters.json", 'w+') as file:
            json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        if accelerator.is_main_process: print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        if accelerator.is_main_process: print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

    relative_train_dataset = videoDataset(
        Path('/home/summer24/sungonce/jyj/elite_cir_dataset_full_B_v4/metadata/combined.json'), 
        preprocess
    )

    hn_batch_sampler = HardNegativeBatchSampler(
        relative_train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_sampler=hn_batch_sampler,
        num_workers=kwargs['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 샘플 프린트 확인용 (메인 프로세스에서만 출력)
    if accelerator.is_main_process:
        for i, (refs, tars, captions, pids) in enumerate(relative_train_loader):
            print(f"--- Batch {i} Sample Pair IDs ---")
            print(pids)
            for j in range(0, 4, 2):
                print(f"Pair {j//2}:")
                print(f"  Forward  Caption: {captions[j]}")
                print(f"  Backward Caption: {captions[j+1]}")
            break

    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])

    # --- Accelerate Prepare (스케줄러 선언 전에 실행하여 데이터로더 길이 동기화) ---
    blip_model, optimizer, relative_train_loader = accelerator.prepare(
        blip_model, optimizer, relative_train_loader
    )

    # 데이터로더가 쪼개진 후의 길이로 스케줄러 정의
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scheduler = accelerator.prepare(scheduler)

    if save_best:
        best_avg_recall = 0

    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    if accelerator.is_main_process: print('Training loop started')
    
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        # tqdm 출력도 메인 프로세스에서만 보이게 설정 (로그가 2배로 찍히는 것 방지)
        train_bar = tqdm(relative_train_loader, ncols=150, disable=not accelerator.is_main_process)
        
        for idx, (reference_images, target_images, captions,_) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            # Accelerate가 자동으로 혼합 정밀도를 처리하므로 autocast는 필요 없음
            loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
            loss = 0.
            for key in loss_dict.keys():
                loss += loss_dict[key]

            # 기존 scaler.scale(loss).backward() 대체
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # 로그 저장은 메인 프로세스에서만
        if accelerator.is_main_process:
            loss_log_dict = {'epoch': epoch}
            for key in train_running_results.keys():
                if key != 'images_in_epoch':
                    loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            
            # DDP 모델이 아닌 순수 모델 가중치로 평가 진행
            unwrapped_model = accelerator.unwrap_model(blip_model)
            recalls_at10 = []
            recalls_at50 = []

            for relative_val_dataset, classic_val_dataset, idx_map in zip(relative_val_datasets, classic_val_datasets, idx_to_dress_mapping):
                index_features, index_names = extract_index_blip_features(classic_val_dataset, unwrapped_model, save_memory)
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, unwrapped_model, index_features, index_names, txt_processors, save_memory)
                
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

            if accelerator.is_main_process:
                print(json.dumps(results_dict, indent=4))
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                if save_training:
                    if save_best and results_dict['average_recall'] > best_avg_recall:
                        best_avg_recall = results_dict['average_recall']
                        save_model('tuned_clip_best', epoch, unwrapped_model, training_path)
        
        # 모든 프로세스가 동기화 되도록 대기
        accelerator.wait_for_everyone()


def clip_finetune_cirr(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                       **kwargs):

    rtc_weights = kwargs['loss_rtc']
    align_weights = kwargs['loss_align']
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{blip_model_name}_{training_start}")
    
    # 메인 프로세스에서만 디렉토리 생성
    if accelerator.is_main_process:
        training_path.mkdir(exist_ok=True, parents=True)
        with open(training_path / "training_hyperparameters.json", 'w+') as file:
            json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 364

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        if accelerator.is_main_process: print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        if accelerator.is_main_process: print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    relative_train_dataset = videoDataset(
        Path('/home/summer24/sungonce/jyj/elite_cir_dataset_full_B_v4/metadata/captioned_v5_merged_valid_data.json'), 
        preprocess
    )

    hn_batch_sampler = HardNegativeBatchSampler(
        relative_train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_sampler=hn_batch_sampler,
        num_workers=kwargs['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    if accelerator.is_main_process:
        for i, (refs, tars, captions, pids) in enumerate(relative_train_loader):
            print(f"--- Batch {i} Sample Pair IDs ---")
            print(pids)
            for j in range(0, 4, 2):
                print(f"Pair {j//2}:")
                print(f"  Forward  Caption: {captions[j]}")
                print(f"  Backward Caption: {captions[j+1]}")
            break

    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])

    # --- Accelerate Prepare ---
    blip_model, optimizer, relative_train_loader = accelerator.prepare(
        blip_model, optimizer, relative_train_loader
    )

    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = accelerator.prepare(scheduler)

    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150, disable=not accelerator.is_main_process)

        for idx, (reference_images, target_images, captions, _) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
            # loss = 0.
            # for key in loss_dict.keys():
            #     if key != 'loss_itc':
            #         loss += kwargs[key] * loss_dict[key]
            #     else:
            #         loss += loss_dict[key]
            total_loss = loss_dict['loss_itc'] + 0.1 * loss_dict['loss_bg']
            # Backward
            accelerator.backward(total_loss)
            optimizer.step()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # 메인 프로세스에서만 로깅
        if accelerator.is_main_process:
            loss_log_dict = {'epoch': epoch}
            for key in train_running_results.keys():
                if key != 'images_in_epoch':
                    loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
            
            training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            unwrapped_model = accelerator.unwrap_model(blip_model)
            
            val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, unwrapped_model)
            results = compute_cirr_val_metrics(relative_val_dataset, unwrapped_model, val_index_features, val_index_names, txt_processors)
            
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
            
            # 메인 프로세스에서만 결과 출력 및 모델 저장
            if accelerator.is_main_process:
                print(json.dumps(results_dict, indent=4))
                
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                if save_training:
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('tuned_clip_arithmetic', epoch, unwrapped_model, training_path)
        
        accelerator.wait_for_everyone()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir]")
    parser.add_argument("--backbone", type=str, default="coco", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--loss-align", default=0.4, type=float)
    parser.add_argument("--loss-rtc", default=0.4, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
    
    # 메인 프로세스에서만 출력
    if accelerator.is_main_process:
        print(f"save-memory: {args.save_memory}")
        
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_itm": args.loss_itm,
        "save_memory": args.save_memory
    }
    
    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(**training_hyper_params)