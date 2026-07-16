import time
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import PIL.Image

from data_utils import targetpad_transform, MACIRDataset
from utils import collate_fn, extract_index_blip_features, device

cache = {}
def load_image_for_display(img_dir, img_name):
    if img_name in cache:
        return cache[img_name]
    img_path = Path(img_dir) / img_name
    img = PIL.Image.open(img_path).convert("RGB")
    if len(cache) < 1000: # Limit cache size to avoid OOM
        cache[img_name] = img
    return img

@torch.no_grad()
def visualize_macir(blip_model_name, backbone, model_path, json_path, img_dir, eval_mode, unchange_ratio, output_pdf, start_query=0, end_query=100):
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
    gallery_path = Path('/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/image_ids.json')
    
    # 2. 데이터셋 설정
    classic_dataset = MACIRDataset(gallery_path, img_dir, mode='classic', preprocess=preprocess)
    relative_dataset = MACIRDataset(json_path, img_dir, mode='relative', preprocess=preprocess)

    # 3. Index 특징 추출
    print(f"Extracting MACIR index features ({eval_mode} mode)...")
    (index_features, _), index_names = extract_index_blip_features(classic_dataset, blip_model, save_memory=True)
    all_index_features = index_features.to(device)
    index_names = np.array(index_names)

    # 4. Visualization 준비
    print(f"Generating Visualization PDF (Range: {start_query} - {end_query}): {output_pdf}")
    relative_val_loader = DataLoader(dataset=relative_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    pdf = PdfPages(output_pdf)
    
    query_idx = 0
    visualized_count = 0
    
    total_range = end_query - start_query
    pbar = tqdm(total=total_range, desc="Visualizing")

    stop_flag = False
    for ref_imgs, batch_target_names, captions, conditions, compositions, batch_ref_names in relative_val_loader:
        if stop_flag:
            break
            
        batch_size = len(batch_target_names)
        batch_start_idx = query_idx
        batch_end_idx = query_idx + batch_size
        
        # 만약 이 배치가 아예 범위 밖이라면 건너뜀
        if batch_end_idx <= start_query:
            query_idx += batch_size
            continue
        if batch_start_idx >= end_query:
            stop_flag = True
            break

        processed_captions = [txt_processors["eval"](caption) for caption in captions]
        ref_imgs = ref_imgs.to(device)
        
        with torch.no_grad():
            with blip_model.maybe_autocast():
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_imgs))
            ref_embeds = ref_embeds.float()
            # sims shape: [batch_size, num_index_images]
            batch_sims = blip_model.inference(ref_embeds, all_index_features, processed_captions)
            
        # 각 쿼리에 대해 처리
        for i in range(len(batch_target_names)):
            current_q_idx = batch_start_idx + i
            
            if current_q_idx < start_query:
                continue
            if current_q_idx >= end_query:
                stop_flag = True
                break
                
            target_name = batch_target_names[i]
            ref_name = batch_ref_names[i]
            caption = captions[i]
            condition = conditions[i]
            composition = compositions[i]
            sims = batch_sims[i]
            
            # Reference 제외하고 정렬 (공식 방식)
            dists = 1 - sims
            sorted_indices = torch.argsort(dists, dim=-1).cpu().numpy()
            sorted_names = index_names[sorted_indices]
            
            # Reference 이미지 제외
            mask = (sorted_names != ref_name)
            sorted_names = sorted_names[mask]
            
            # GT 순위 찾기
            gt_rank = np.where(sorted_names == target_name)[0]
            if len(gt_rank) > 0:
                gt_rank = gt_rank[0] + 1
            else:
                gt_rank = -1
                
            # 시각화 (Top 5)
            top_k = 5
            top_k_names = sorted_names[:top_k]
            
            fig = plt.figure(figsize=(18, 4))
            plt.suptitle(f"Query {current_q_idx}: {condition} | {composition}\nCaption: {caption}\nGT: {target_name} (Rank: {gt_rank})", fontsize=11)
            
            # 1. Reference Image
            ax = plt.subplot(1, 7, 1)
            ref_img = load_image_for_display(img_dir, ref_name)
            ax.imshow(ref_img)
            ax.set_title(f"Reference:\n{ref_name}", fontsize=8)
            ax.axis('off')
            
            # 2. Target (GT) Image
            ax = plt.subplot(1, 7, 2)
            tar_img = load_image_for_display(img_dir, target_name)
            ax.imshow(tar_img)
            ax.set_title(f"Target (GT):\n{target_name}", fontsize=8)
            ax.axis('off')
            
            # 3. Top-K Results
            for k in range(top_k):
                ax = plt.subplot(1, 7, k + 3)
                res_name = top_k_names[k]
                res_img = load_image_for_display(img_dir, res_name)
                ax.imshow(res_img)
                title = f"Rank {k+1}:\n{res_name}"
                if res_name == target_name:
                    ax.set_title(title, color='red', fontweight='bold', fontsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                else:
                    ax.set_title(title, fontsize=8)
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            pdf.savefig(fig, dpi=80)
            plt.close(fig)
            
            visualized_count += 1
            pbar.update(1)
            
        query_idx += batch_size
        
    pbar.close()
    pdf.close()
    print(f"Finished. Visualized {visualized_count} queries. saved to {output_pdf}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain_vitL", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--macir-json", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/macir_meta_full.json")
    parser.add_argument("--macir-img-dir", type=str, default="/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/images")
    parser.add_argument("--eval-mode", default="fusion", choices=["dynamic", "fusion"], help="Evaluation mode")
    parser.add_argument("--unchange-ratio", default=0.2, type=float, help="Evaluation unchange-ratio")
    parser.add_argument("--output-pdf", type=str, default="macir_visualization_range.pdf")
    parser.add_argument("--start-query", type=int, default=0, help="Start index of queries to visualize")
    parser.add_argument("--end-query", type=int, default=100, help="End index of queries to visualize")

    args = parser.parse_args()
    
    visualize_macir(
        args.blip_model_name, args.backbone, args.model_path,
        args.macir_json, args.macir_img_dir, args.eval_mode, args.unchange_ratio,
        args.output_pdf, start_query=args.start_query, end_query=args.end_query
    )
