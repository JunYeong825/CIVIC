import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from data_utils import targetpad_transform, CIRRDataset
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

def visualize_cirr_batch(num_samples=100):
    # CPU 사용 강제 (OOM 방지)
    vis_device = torch.device("cpu")
    
    # 1. 설정
    model_name = "blip2_cir_align_prompt"
    model_path = "/home/summer24/sungonce/jyj/cir_dataset/models/blip2_cir_cirr_2026-04-03_12:52:47/mixkit_pexel/tuned_blip_best_cirr.pt"
    output_pdf = "attention_report_cirr.pdf"

    # 2. 모델 로드
    print("Loading model on CPU...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name, model_type="pretrain_vitL", is_eval=True, device=vis_device
    )
    checkpoint = torch.load(model_path, map_location=vis_device)
    state_dict = checkpoint[model.__class__.__name__] if model.__class__.__name__ in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. CIRR 데이터셋 로드
    print("Loading CIRR dataset...")
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    dataset = CIRRDataset('val', 'relative', preprocess)
    
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
    print(f"Generating {num_samples} attention maps on CPU...")
    from data_utils import base_path
    
    with PdfPages(output_pdf) as pdf:
        for i in tqdm(range(min(num_samples, len(dataset)))):
            # CIRRDataset 'val' mode returns (reference_name, target_name, rel_caption, group_members)
            ref_name, tar_name, caption_raw, _ = dataset[i]
            
            # 이미지 로드
            rel_path = dataset.name_to_relpath[ref_name]
            # rel_path는 'dev/dev-123-1-img0.png'와 같은 형태
            img_path = base_path / 'CIRR' / rel_path
            
            try:
                raw_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            caption = txt_processors["eval"](caption_raw)
            # 시각화용 이미지와 모델 입력용 이미지가 동일해야 하므로 raw_image에 전처리 적용
            # 하지만 원본 가로세로비를 유지하면서 보여주기 위해 vis_processors 사용
            image_input = vis_processors["eval"](raw_image).unsqueeze(0).to(vis_device)

            attentions.clear()
            with torch.no_grad():
                # CPU에서는 maybe_autocast가 필요 없거나 다르게 동작할 수 있으나 안전을 위해 유지
                # 단, vis_device가 cpu면 autocast는 무시됨
                image_embeds = model.ln_vision(model.visual_encoder(image_input))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(vis_device)
                query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                text_tokens = model.tokenizer([caption], padding="max_length", truncation=True, max_length=model.max_txt_len, return_tensors="pt").to(vis_device)
                
                bsz = image_input.size(0)
                total_len = query_tokens.size(1) + text_tokens.input_ids.size(1)
                part_attention_mask = torch.ones((bsz, total_len, total_len), device=vis_device)
                part_attention_mask[:, 0:16, 32:] = 0
                text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
                part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask

                _ = model.Qformer.bert(
                    text_tokens.input_ids,
                    query_embeds=query_tokens,
                    attention_mask=part_attention_mask,
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

            # 시각화
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(raw_image); axes[0].set_title(f"Sample {i}\nText: {caption_raw[:50]}..."); axes[0].axis('off')
            axes[1].imshow(qs_vis); axes[1].set_title(f"Static Queries (Qs)\nBackground Focus"); axes[1].axis('off')
            axes[2].imshow(qd_vis); axes[2].set_title(f"Dynamic Queries (Qd)\nText-driven Focus"); axes[2].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    handle.remove()
    print(f"Report saved to '{os.path.abspath(output_pdf)}'")

if __name__ == "__main__":
    visualize_cirr_batch(100)
