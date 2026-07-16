import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from data_utils import targetpad_transform, MACIRDataset
from utils import device
import os
import functools
import requests

# SSL Patch
old_merge_environment_settings = requests.Session.merge_environment_settings
@functools.wraps(old_merge_environment_settings)
def patched_merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings
requests.Session.merge_environment_settings = patched_merge_environment_settings

def get_heatmap(map_data, img_size):
    # map_data: (16, 16)
    map_data = map_data.astype(np.float32)
    map_data = (map_data - map_data.min()) / (map_data.max() - map_data.min() + 1e-8)
    heatmap = cv2.resize(map_data, (img_size[0], img_size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_macir_batch(start_idx=100, end_idx=200):
    # CPU 사용 강제 (OOM 방지)
    vis_device = torch.device("cpu")
    
    # 1. 설정
    model_name = "blip2_cir_align_prompt"
    model_path = "/home/summer24/sungonce/jyj/cir_dataset/models/blip2_cir_cirr_2026-04-03_12:52:47/mixkit_pexel/tuned_blip_best_cirr.pt"
    output_pdf = f"attention_report_macir_{start_idx}_to_{end_idx-1}.pdf"
    
    macir_json = "/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/macir_meta_full.json"
    macir_img_dir = "/home/summer24/sungonce/jyj/cir_dataset/MACIR_dataset/MACIR_dataset/images"

    # 2. 모델 로드
    print("Loading model on CPU...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, model_type="pretrain_vitL", is_eval=True, device=vis_device
    )
    checkpoint = torch.load(model_path, map_location=vis_device)
    state_dict = checkpoint[model.__class__.__name__] if model.__class__.__name__ in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. MACIR 데이터셋 로드
    print("Loading MACIR dataset...")
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    dataset = MACIRDataset(macir_json, macir_img_dir, 'relative', preprocess)
    
    # 4. Hook 설정
    attentions = []
    def hook_fn(module, input, output):
        # output is likely (context_layer, attention_probs)
        # We need attention_probs
        if isinstance(output, tuple):
            attn = output[1]
        else:
            attn = output
            
        if isinstance(attn, torch.Tensor):
            attentions.append(attn.detach().cpu())
        elif isinstance(attn, tuple): # Handle nested tuples if any
            attentions.append(attn[0].detach().cpu())

    last_layer_idx = -1
    for i in range(len(model.Qformer.bert.encoder.layer) - 1, -1, -1):
        if hasattr(model.Qformer.bert.encoder.layer[i], "crossattention"):
            last_layer_idx = i
            break
    handle = model.Qformer.bert.encoder.layer[last_layer_idx].crossattention.self.register_forward_hook(hook_fn)

    # 5. 시각화 및 PDF 저장
    print(f"Generating attention maps from index {start_idx} to {end_idx-1}...")
    
    with PdfPages(output_pdf) as pdf:
        # start_idx 부터 end_idx 직전까지 반복하되, 데이터셋 길이를 초과하지 않도록 방어 코드 작성
        for i in tqdm(range(start_idx, min(end_idx, len(dataset)))):
            # MACIRDataset 'relative' mode returns (ref_image, tar_name, caption, condition)
            item = dataset.triplets[i]
            ref_name = item['reference_image']
            tar_name = item['target_image']
            caption_raw = item['relative_caption']
            condition = item['condition_type']
            
            # 이미지 경로 설정
            img_path = dataset.img_dir / ref_name
            tar_img_path = dataset.img_dir / tar_name # 타겟 이미지 경로 추가
            
            try:
                # Reference Image 와 Target Image 함께 로드
                raw_image = Image.open(img_path).convert("RGB")
                target_image = Image.open(tar_img_path).convert("RGB") # 타겟 이미지 로드 추가
            except Exception as e:
                print(f"Error loading images for index {i}: {e}")
                continue
            
            # 시각화용 이미지와 모델 입력용 이미지가 동일해야 하므로 raw_image에 전처리 적용
            image_input = vis_processors["eval"](raw_image).unsqueeze(0).to(vis_device)

            attentions.clear()
            with torch.no_grad():
                # 이미지 임베딩 추출
                image_embeds = model.ln_vision(model.visual_encoder(image_input))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(vis_device)
                query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                
                # Q-former에 이미지만 입력 (텍스트 토큰 제외)
                _ = model.Qformer.bert(
                    input_ids=None,
                    query_embeds=query_tokens,
                    attention_mask=None,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            if not attentions:
                continue
                
            attn_map = attentions[0]
            if isinstance(attn_map, tuple): attn_map = attn_map[0]
            
            # [1, heads, image_len, query_len] -> [query_len, image_len]
            attn_map_transposed = attn_map.permute(0, 1, 3, 2)
            attn_map_queries = attn_map_transposed.mean(dim=1).squeeze(0)
            
            attn_map_qformer = attn_map_queries[:32, 1:] # [32, 256]
            attn_map_patches = attn_map_qformer.reshape(32, 16, 16)
            
            qs_map = attn_map_patches[0:16].mean(dim=0).cpu().numpy()
            qd_map = attn_map_patches[16:32].mean(dim=0).cpu().numpy()

            qs_heatmap = get_heatmap(qs_map, raw_image.size)
            qd_heatmap = get_heatmap(qd_map, raw_image.size)
            
            qs_vis = cv2.addWeighted(np.array(raw_image), 0.6, qs_heatmap, 0.4, 0)
            qd_vis = cv2.addWeighted(np.array(raw_image), 0.6, qd_heatmap, 0.4, 0)

            # 시각화 (기존 1x3 배열에서 1x4 배열로 수정)
            fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # 타겟 이미지를 위해 공간 및 axes 확장
            
            # 1. Reference Image
            axes[0].imshow(raw_image)
            axes[0].set_title(f"Sample {i} ({condition})\nRef: {ref_name}\nCap: {caption_raw}")
            axes[0].axis('off')
            
            # 2. Static Queries (Qs) Heatmap
            axes[1].imshow(qs_vis)
            axes[1].set_title("Static Queries (Qs)\nImage Focus")
            axes[1].axis('off')
            
            # 3. Dynamic Queries (Qd) Heatmap
            axes[2].imshow(qd_vis)
            axes[2].set_title("Dynamic Queries (Qd)\nImage Focus")
            axes[2].axis('off')
            
            # 4. Target Image (추가됨)
            axes[3].imshow(target_image)
            axes[3].set_title(f"Target Image\nTar: {tar_name}")
            axes[3].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    handle.remove()
    print(f"Report saved to '{os.path.abspath(output_pdf)}'")

if __name__ == "__main__":
    visualize_macir_batch(start_idx=0, end_idx=200)