"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.common.dist_utils import download_cached_file
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, video_embds: torch.Tensor, text_embds: torch.Tensor, temp):
        # 1. FP32 변환 (유지)
        video_embds = video_embds.float()
        text_embds = text_embds.float()

        batch_size = video_embds.size(0)
        sim_matrix = video_embds @ text_embds.T 
        sim_matrix = sim_matrix / temp

        nominator = torch.diagonal(sim_matrix)

        # 2. 수치 폭발 방지를 위한 Clamp 추가
        # exp(80)은 FP32 한계 내에서 매우 큰 값이므로 성능에 영향을 주지 않으면서 inf를 막습니다.
        beta_sim = torch.clamp(self.beta * sim_matrix, max=80) 
        
        # 분모가 0이 되는 것을 방지하기 위해 1e-8(eps) 추가
        exp_beta = torch.exp(beta_sim)
        diag_beta = torch.exp(torch.diagonal(beta_sim))
        
        w_v2t = (batch_size - 1) * exp_beta / (exp_beta.sum(dim=1, keepdim=True) - diag_beta.unsqueeze(1) + 1e-8)
        w_t2v = (batch_size - 1) * exp_beta / (exp_beta.sum(dim=0, keepdim=True) - diag_beta.unsqueeze(0) + 1e-8)

        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        # 3. Denominator 계산 시에도 exp 값 제한
        safe_sim_matrix = torch.clamp(sim_matrix, max=80)
        denominator_v2t = torch.log((torch.exp(safe_sim_matrix) * w_v2t).sum(dim=1) + 1e-8)
        denominator_t2v = torch.log((torch.exp(safe_sim_matrix) * w_t2v).sum(dim=0) + 1e-8)

        hn_nce_loss = (denominator_v2t - nominator).mean() + (denominator_t2v - nominator).mean()
        return hn_nce_loss
    
@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=364,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # if self.Qformer.cls.predictions.bias.shape[0] != len(self.tokenizer):
        #     old_bias = self.Qformer.cls.predictions.bias
        #     new_bias = nn.Parameter(torch.zeros(len(self.tokenizer), device=old_bias.device, dtype=old_bias.dtype))
        #     n = min(old_bias.size(0), len(self.tokenizer))
        #     new_bias.data[:n] = old_bias.data[:n]
        #     self.Qformer.cls.predictions.bias = new_bias
        #     self.Qformer.cls.predictions.decoder.bias = new_bias
        # self.Qformer.resize_token_embeddings(30523)
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        # self.temp = nn.Parameter(0.07 * torch.ones(1))

        self.max_txt_len = max_txt_len
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        # self.hn_nce_loss = HardNegativeNCE(alpha=1.0, beta=1.0)
        self.eval_mode = 'fusion' # 'dynamic' or 'fusion'
        self.unchange_ratio = 0.2
        
    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        ###============== image features extraction ===================###
        # reference image feature  
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        # target image feature
        target_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        ###============== Partitioned Attention Mask ===================###
        # query_tokens (32) + text_tokens (max_txt_len)
        # Static Queries (Qs): 0-15
        # Dynamic Queries (Qd): 16-31
        # Text Tokens (T): 32 onwards
        
        bsz = image.size(0)
        num_queries = query_tokens.size(1) # 32
        text_len = text_tokens.input_ids.size(1)
        total_len = num_queries + text_len
        
        # Initialize 3D attention mask: [bsz, total_len, total_len]
        # 1 means attend, 0 means mask
        part_attention_mask = torch.ones((bsz, total_len, total_len), device=image.device)
        
        # Qs (0-15) should NOT attend to Text Tokens (32 onwards)
        part_attention_mask[:, 0:16, 32:] = 0
        
        # (Optional) If we want to prevent Text Tokens from attending to Qs, we could set:
        part_attention_mask[:, 32:, 0:16] = 0
        # But the requirement says Qs should not refer to Text Tokens.
        
        # text_tokens.attention_mask handles padding tokens
        # It has shape [bsz, text_len]
        # We need to broadcast it to our 3D mask
        text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
        part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask

        ###============== (ABLATION) Without Masking ===================###
        # Standard mask: Qs can see Text Tokens
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        ###============== reference text fusion (Qs, Qd) ===================###
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=part_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        
        # Static features (Qs): mean of tokens 0-15
        f_s_ref = fusion_output.last_hidden_state[:, 0:16, :].mean(dim=1)
        # Dynamic features (Qd): mean of tokens 16-31
        f_d_ref_text = fusion_output.last_hidden_state[:, 16:32, :].mean(dim=1)
        
        fusion_feats = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

        ###============== Target features extraction (Qs, Qd) ===================###
        # For target, we only need visual features. Text is empty.
        # However, to keep it consistent with Qformer partitioning, 
        # we can pass target image and see what Qs, Qd extract.
        # Usually, for target image features in CIR, we don't use text.
        
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=target_embeds,
            encoder_attention_mask=target_atts,
            return_dict=True,
        )
        
        f_s_tar = target_output.last_hidden_state[:, 0:16, :].mean(dim=1)
        f_d_tar = target_output.last_hidden_state[:, 16:32, :].mean(dim=1)
        
        target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1) # (bsz, 32, dim)
        target_feats_d = F.normalize(self.vision_proj(target_output.last_hidden_state[:, 16:32, :]), dim=-1) # (bsz, 16, dim)
        
        # Representative target feature for contrastive loss (using dynamic queries)
        target_feats_representative = target_feats_d.mean(dim=1) # (bsz, dim)

        ###============== Loss Calculation ===================###
        
        # 1. CIR Contrastive Loss (L_CIR)
        # Using Qd derived features
        # Option A: Simple Cross Entropy (Current style)
        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats_d.permute(0, 2, 1)
        ).squeeze() # (bsz, bsz, 16)
        
        sim_i2t, _ = sim_t2q.max(-1) # (bsz, bsz)
        sim_i2t = sim_i2t / self.temp
        
        targets = torch.arange(image.size(0), device=image.device)
        loss_cir = F.cross_entropy(sim_i2t, targets)

        # 2. Background Invariance Loss (L_BG)
        # MSE between Qs(I_ref) and Qs(I_tar)
        loss_bg = F.mse_loss(f_s_ref, f_s_tar)

        return {
            'loss_itc': loss_cir, 
            'loss_bg': loss_bg
        }

    # def forward(self, samples):
    #     image = samples["image"]
    #     target = samples["target"]
    #     text = samples["text_input"]

    #     ###============== image features extraction ===================###
    #     # reference image feature  
    #     image_embeds = self.ln_vision(self.visual_encoder(image))
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
    #     # target image feature
    #     target_embeds = self.ln_vision(self.visual_encoder(target))
    #     target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)

    #     # query tokens
    #     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

    #     # text tokens
    #     text_tokens = self.tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_txt_len,
    #         return_tensors="pt",
    #     ).to(image.device)

    #     ###============== Partitioned Attention Mask ===================###
    #     # query_tokens (32) + text_tokens (max_txt_len)
    #     # Static Queries (Qs): 0-7 # [수정된 부분: 0-15 -> 0-7]
    #     # Dynamic Queries (Qd): 8-31 # [수정된 부분: 16-31 -> 8-31]
    #     # Text Tokens (T): 32 onwards
        
    #     bsz = image.size(0)
    #     num_queries = query_tokens.size(1) # 32
    #     text_len = text_tokens.input_ids.size(1)
    #     total_len = num_queries + text_len
        
    #     # Initialize 3D attention mask: [bsz, total_len, total_len]
    #     # 1 means attend, 0 means mask
    #     part_attention_mask = torch.ones((bsz, total_len, total_len), device=image.device)
        
    #     # Qs (0-7) should NOT attend to Text Tokens (32 onwards) # [수정된 부분: 0-15 -> 0-7]
    #     part_attention_mask[:, 0:8, 32:] = 0 # [수정된 부분: 0:16 -> 0:8]
        
    #     # (Optional) If we want to prevent Text Tokens from attending to Qs, we could set:
    #     part_attention_mask[:, 32:, 0:8] = 0 # [수정된 부분: 0:16 -> 0:8 주석 내용 반영]
    #     # But the requirement says Qs should not refer to Text Tokens.
        
    #     # text_tokens.attention_mask handles padding tokens
    #     # It has shape [bsz, text_len]
    #     # We need to broadcast it to our 3D mask
    #     text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
    #     part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask

    #     ###============== (ABLATION) Without Masking ===================###
    #     # Standard mask: Qs can see Text Tokens
    #     # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
    #     ###============== reference text fusion (Qs, Qd) ===================###
    #     fusion_output = self.Qformer.bert(
    #         text_tokens.input_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=part_attention_mask,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )
        
    #     # Static features (Qs): mean of tokens 0-7 # [수정된 부분: 0-15 -> 0-7]
    #     f_s_ref = fusion_output.last_hidden_state[:, 0:8, :].mean(dim=1) # [수정된 부분: 0:16 -> 0:8]
    #     # Dynamic features (Qd): mean of tokens 8-31 # [수정된 부분: 16-31 -> 8-31]
    #     f_d_ref_text = fusion_output.last_hidden_state[:, 8:32, :].mean(dim=1) # [수정된 부분: 16:32 -> 8:32]
        
    #     fusion_feats = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

    #     ###============== Target features extraction (Qs, Qd) ===================###
    #     # For target, we only need visual features. Text is empty.
    #     # However, to keep it consistent with Qformer partitioning, 
    #     # we can pass target image and see what Qs, Qd extract.
    #     # Usually, for target image features in CIR, we don't use text.
        
    #     target_output = self.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=target_embeds,
    #         encoder_attention_mask=target_atts,
    #         return_dict=True,
    #     )
        
    #     f_s_tar = target_output.last_hidden_state[:, 0:8, :].mean(dim=1) # [수정된 부분: 0:16 -> 0:8]
    #     f_d_tar = target_output.last_hidden_state[:, 8:32, :].mean(dim=1) # [수정된 부분: 16:32 -> 8:32]
        
    #     target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1) # (bsz, 32, dim)
    #     target_feats_d = F.normalize(self.vision_proj(target_output.last_hidden_state[:, 8:32, :]), dim=-1) # [수정된 부분: 16:32 -> 8:32, 주석 (bsz, 16, dim) -> (bsz, 24, dim) 반영]
        
    #     # Representative target feature for contrastive loss (using dynamic queries)
    #     target_feats_representative = target_feats_d.mean(dim=1) # (bsz, dim)

    #     ###============== Loss Calculation ===================###
        
    #     # 1. CIR Contrastive Loss (L_CIR)
    #     # Using Qd derived features
    #     # Option A: Simple Cross Entropy (Current style)
    #     sim_t2q = torch.matmul(
    #         fusion_feats.unsqueeze(1).unsqueeze(1), target_feats_d.permute(0, 2, 1)
    #     ).squeeze() # [수정된 부분: 주석 차원 설명 (bsz, bsz, 16) -> (bsz, bsz, 24) 반영]
        
    #     sim_i2t, _ = sim_t2q.max(-1) # (bsz, bsz)
    #     sim_i2t = sim_i2t / self.temp
        
    #     targets = torch.arange(image.size(0), device=image.device)
    #     loss_cir = F.cross_entropy(sim_i2t, targets)

    #     # 2. Background Invariance Loss (L_BG)
    #     # MSE between Qs(I_ref) and Qs(I_tar)
    #     loss_bg = F.mse_loss(f_s_ref, f_s_tar)

    #     return {
    #         'loss_itc': loss_cir, 
    #         'loss_bg': loss_bg
    #     }
    
    # def forward(self, samples):
    #     image = samples["image"]
    #     target = samples["target"]
    #     text = samples["text_input"]

    #     ###============== image features extraction ===================###
    #     # reference image feature  
    #     image_embeds = self.ln_vision(self.visual_encoder(image))
    #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
    #     # target image feature
    #     target_embeds = self.ln_vision(self.visual_encoder(target))
    #     target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)

    #     # query tokens
    #     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

    #     # text tokens
    #     text_tokens = self.tokenizer(
    #         text,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_txt_len,
    #         return_tensors="pt",
    #     ).to(image.device)

    #     ###============== Partitioned Attention Mask ===================###
    #     # query_tokens (32) + text_tokens (max_txt_len)
    #     # Static Queries (Qs): 0-23 # [수정된 부분: 0-7 -> 0-23]
    #     # Dynamic Queries (Qd): 24-31 # [수정된 부분: 8-31 -> 24-31]
    #     # Text Tokens (T): 32 onwards
        
    #     bsz = image.size(0)
    #     num_queries = query_tokens.size(1) # 32
    #     text_len = text_tokens.input_ids.size(1)
    #     total_len = num_queries + text_len
        
    #     # Initialize 3D attention mask: [bsz, total_len, total_len]
    #     # 1 means attend, 0 means mask
    #     part_attention_mask = torch.ones((bsz, total_len, total_len), device=image.device)
        
    #     # Qs (0-23) should NOT attend to Text Tokens (32 onwards) # [수정된 부분: 0-7 -> 0-23]
    #     part_attention_mask[:, 0:24, 32:] = 0 # [수정된 부분: 0:8 -> 0:24]
        
    #     # (Optional) If we want to prevent Text Tokens from attending to Qs, we could set:
    #     part_attention_mask[:, 32:, 0:24] = 0 # [수정된 부분: 0:8 -> 0:24 주석 내용 반영]
    #     # But the requirement says Qs should not refer to Text Tokens.
        
    #     # text_tokens.attention_mask handles padding tokens
    #     # It has shape [bsz, text_len]
    #     # We need to broadcast it to our 3D mask
    #     text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
    #     part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask

    #     ###============== (ABLATION) Without Masking ===================###
    #     # Standard mask: Qs can see Text Tokens
    #     # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
    #     ###============== reference text fusion (Qs, Qd) ===================###
    #     fusion_output = self.Qformer.bert(
    #         text_tokens.input_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=part_attention_mask,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )
        
    #     # Static features (Qs): mean of tokens 0-23 # [수정된 부분: 0-7 -> 0-23]
    #     f_s_ref = fusion_output.last_hidden_state[:, 0:24, :].mean(dim=1) # [수정된 부분: 0:8 -> 0:24]
    #     # Dynamic features (Qd): mean of tokens 24-31 # [수정된 부분: 8-31 -> 24-31]
    #     f_d_ref_text = fusion_output.last_hidden_state[:, 24:32, :].mean(dim=1) # [수정된 부분: 8:32 -> 24:32]
        
    #     fusion_feats = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

    #     ###============== Target features extraction (Qs, Qd) ===================###
    #     # For target, we only need visual features. Text is empty.
    #     # However, to keep it consistent with Qformer partitioning, 
    #     # we can pass target image and see what Qs, Qd extract.
    #     # Usually, for target image features in CIR, we don't use text.
        
    #     target_output = self.Qformer.bert(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=target_embeds,
    #         encoder_attention_mask=target_atts,
    #         return_dict=True,
    #     )
        
    #     f_s_tar = target_output.last_hidden_state[:, 0:24, :].mean(dim=1) # [수정된 부분: 0:8 -> 0:24]
    #     f_d_tar = target_output.last_hidden_state[:, 24:32, :].mean(dim=1) # [수정된 부분: 8:32 -> 24:32]
        
    #     target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1) # (bsz, 32, dim)
    #     target_feats_d = F.normalize(self.vision_proj(target_output.last_hidden_state[:, 24:32, :]), dim=-1) # [수정된 부분: 8:32 -> 24:32, 주석 (bsz, 24, dim) -> (bsz, 8, dim) 반영]
        
    #     # Representative target feature for contrastive loss (using dynamic queries)
    #     target_feats_representative = target_feats_d.mean(dim=1) # (bsz, dim)

    #     ###============== Loss Calculation ===================###
        
    #     # 1. CIR Contrastive Loss (L_CIR)
    #     # Using Qd derived features
    #     # Option A: Simple Cross Entropy (Current style)
    #     sim_t2q = torch.matmul(
    #         fusion_feats.unsqueeze(1).unsqueeze(1), target_feats_d.permute(0, 2, 1)
    #     ).squeeze() # [수정된 부분: 주석 차원 설명 (bsz, bsz, 24) -> (bsz, bsz, 8) 반영]
        
    #     sim_i2t, _ = sim_t2q.max(-1) # (bsz, bsz)
    #     sim_i2t = sim_i2t / self.temp
        
    #     targets = torch.arange(image.size(0), device=image.device)
    #     loss_cir = F.cross_entropy(sim_i2t, targets)

    #     # 2. Background Invariance Loss (L_BG)
    #     # MSE between Qs(I_ref) and Qs(I_tar)
    #     loss_bg = F.mse_loss(f_s_ref, f_s_tar)

    #     return {
    #         'loss_itc': loss_cir, 
    #         'loss_bg': loss_bg
    #     }
        
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text):
        device = self.device 
        reference_embeds = reference_embeds.to(device)
        target_feats = target_feats.to(device) # (num_index, 32, dim)

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

        text_tokens = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
        ).to(device)

        # 3. Partitioned Attention Mask
        bsz = reference_embeds.size(0)
        num_queries = query_tokens.size(1)
        total_len = num_queries + text_tokens.input_ids.size(1)
        
        part_attention_mask = torch.ones((bsz, total_len, total_len), device=device)
        part_attention_mask[:, 0:16, 32:] = 0
        part_attention_mask[:, 32:, 0:16] = 0
        text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
        part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask
        
        # (ABLATION) Without Masking
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=part_attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # 공통 특징 추출
        f_s_ref = fusion_output.last_hidden_state[:, 0:16, :].mean(dim=1)
        f_d_ref_text = fusion_output.last_hidden_state[:, 16:32, :].mean(dim=1)
        
        f_s_ref_proj = F.normalize(self.vision_proj(f_s_ref), dim=-1)
        f_d_ref_text_proj = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

        # Gallery 측 특징 명시적 분리 및 투영 (target_feats는 항상 32개 토큰이라고 가정)
        f_s_tar_all = F.normalize(self.vision_proj(target_feats[:, 0:16, :]), dim=-1)
        f_d_tar_all = F.normalize(self.vision_proj(target_feats[:, 16:32, :]), dim=-1)

        if self.eval_mode == 'dynamic':
            # 오직 Qd만 사용 (16:32) - max-similarity 방식 유지
            sim_t2q = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
            sim_i2t, _ = sim_t2q.max(-1)

        elif self.eval_mode == 'fusion':
            # Dynamic (Main) Similarity: Qd의 max-sim 활용 (Dynamic 단독 모드와 수식 통일)
            sim_t2q_d = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
            sim_dynamic, _ = sim_t2q_d.max(-1)
            
            # Static (Auxiliary) Similarity: Qs의 mean-sim 활용
            f_s_tar_mean = f_s_tar_all.mean(dim=1)
            sim_static = torch.matmul(f_s_ref_proj, f_s_tar_mean.T)
            
            # Weighted Sum: Dynamic 성능을 보존하며 Static으로 미세 조정
            alpha = self.unchange_ratio # 배경 영향력을 보조적 수준으로 설정
            sim_i2t = sim_dynamic + alpha * sim_static

        return sim_i2t

    # @torch.no_grad()
    # def inference(self, reference_embeds, target_feats, text):
    #     device = self.device 
    #     reference_embeds = reference_embeds.to(device)
    #     target_feats = target_feats.to(device) # (num_index, 32, dim)

    #     image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(device)
    #     query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

    #     text_tokens = self.tokenizer(
    #         text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
    #     ).to(device)

    #     # 3. Partitioned Attention Mask
    #     bsz = reference_embeds.size(0)
    #     num_queries = query_tokens.size(1)
    #     total_len = num_queries + text_tokens.input_ids.size(1)
        
    #     part_attention_mask = torch.ones((bsz, total_len, total_len), device=device)
    #     part_attention_mask[:, 0:8, 32:] = 0 # [수정된 부분: 0:16 -> 0:8]
    #     part_attention_mask[:, 32:, 0:8] = 0
    #     text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
    #     part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask
        
    #     # (ABLATION) Without Masking
    #     # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
    #     fusion_output = self.Qformer.bert(
    #         text_tokens.input_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=part_attention_mask,
    #         encoder_hidden_states=reference_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )

    #     # 공통 특징 추출
    #     f_s_ref = fusion_output.last_hidden_state[:, 0:8, :].mean(dim=1) # [수정된 부분: 0:16 -> 0:8]
    #     f_d_ref_text = fusion_output.last_hidden_state[:, 8:32, :].mean(dim=1) # [수정된 부분: 16:32 -> 8:32]
        
    #     f_s_ref_proj = F.normalize(self.vision_proj(f_s_ref), dim=-1)
    #     f_d_ref_text_proj = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

    #     # Gallery 측 특징 명시적 분리 및 투영 (target_feats는 항상 32개 토큰이라고 가정)
    #     f_s_tar_all = F.normalize(self.vision_proj(target_feats[:, 0:8, :]), dim=-1) # [수정된 부분: 0:16 -> 0:8]
    #     f_d_tar_all = F.normalize(self.vision_proj(target_feats[:, 8:32, :]), dim=-1) # [수정된 부분: 16:32 -> 8:32]

    #     if self.eval_mode == 'dynamic':
    #         # 오직 Qd만 사용 (8:32) - max-similarity 방식 유지 # [수정된 부분: 주석 내용 (16:32) -> (8:32) 변경]
    #         sim_t2q = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
    #         sim_i2t, _ = sim_t2q.max(-1)

    #     elif self.eval_mode == 'fusion':
    #         # Dynamic (Main) Similarity: Qd의 max-sim 활용 (Dynamic 단독 모드와 수식 통일)
    #         sim_t2q_d = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
    #         sim_dynamic, _ = sim_t2q_d.max(-1)
            
    #         # Static (Auxiliary) Similarity: Qs의 mean-sim 활용
    #         f_s_tar_mean = f_s_tar_all.mean(dim=1)
    #         sim_static = torch.matmul(f_s_ref_proj, f_s_tar_mean.T)
            
    #         # Weighted Sum: Dynamic 성능을 보존하며 Static으로 미세 조정
    #         alpha = self.unchange_ratio # 배경 영향력을 보조적 수준으로 설정
    #         sim_i2t = sim_dynamic + alpha * sim_static

    #     return sim_i2t
    
    # def inference(self, reference_embeds, target_feats, text):
    #     device = self.device 
    #     reference_embeds = reference_embeds.to(device)
    #     target_feats = target_feats.to(device) # (num_index, 32, dim)

    #     image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(device)
    #     query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
    #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

    #     text_tokens = self.tokenizer(
    #         text, padding="max_length", truncation=True, max_length=self.max_txt_len, return_tensors="pt"
    #     ).to(device)

    #     # 3. Partitioned Attention Mask
    #     bsz = reference_embeds.size(0)
    #     num_queries = query_tokens.size(1)
    #     total_len = num_queries + text_tokens.input_ids.size(1)
        
    #     part_attention_mask = torch.ones((bsz, total_len, total_len), device=device)
    #     part_attention_mask[:, 0:24, 32:] = 0 # [수정된 부분: 0:8 -> 0:24]
    #     part_attention_mask[:, 32:, 0:24] = 0 # [수정된 부분: 0:8 -> 0:24
    #     text_pad_mask = text_tokens.attention_mask.unsqueeze(1).expand(-1, total_len, -1)
    #     part_attention_mask[:, :, 32:] = part_attention_mask[:, :, 32:] * text_pad_mask
        
    #     # (ABLATION) Without Masking
    #     # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
    #     fusion_output = self.Qformer.bert(
    #         text_tokens.input_ids,
    #         query_embeds=query_tokens,
    #         attention_mask=part_attention_mask,
    #         encoder_hidden_states=reference_embeds,
    #         encoder_attention_mask=image_atts,
    #         return_dict=True,
    #     )

    #     # 공통 특징 추출
    #     f_s_ref = fusion_output.last_hidden_state[:, 0:24, :].mean(dim=1) # [수정된 부분: 0:8 -> 0:24]
    #     f_d_ref_text = fusion_output.last_hidden_state[:, 24:32, :].mean(dim=1) # [수정된 부분: 8:32 -> 24:32]
        
    #     f_s_ref_proj = F.normalize(self.vision_proj(f_s_ref), dim=-1)
    #     f_d_ref_text_proj = F.normalize(self.text_proj(f_d_ref_text), dim=-1)

    #     # Gallery 측 특징 명시적 분리 및 투영 (target_feats는 항상 32개 토큰이라고 가정)
    #     f_s_tar_all = F.normalize(self.vision_proj(target_feats[:, 0:24, :]), dim=-1) # [수정된 부분: 0:8 -> 0:24]
    #     f_d_tar_all = F.normalize(self.vision_proj(target_feats[:, 24:32, :]), dim=-1) # [수정된 부분: 8:32 -> 24:32]

    #     if self.eval_mode == 'dynamic':
    #         # 오직 Qd만 사용 (24:32) - max-similarity 방식 유지 # [수정된 부분: 주석 내용 (8:32) -> (24:32) 변경]
    #         sim_t2q = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
    #         sim_i2t, _ = sim_t2q.max(-1)

    #     elif self.eval_mode == 'fusion':
    #         # Dynamic (Main) Similarity: Qd의 max-sim 활용 (Dynamic 단독 모드와 수식 통일)
    #         sim_t2q_d = torch.matmul(f_d_ref_text_proj.unsqueeze(1).unsqueeze(1), f_d_tar_all.permute(0, 2, 1)).squeeze()
    #         sim_dynamic, _ = sim_t2q_d.max(-1)
            
    #         # Static (Auxiliary) Similarity: Qs의 mean-sim 활용
    #         f_s_tar_mean = f_s_tar_all.mean(dim=1)
    #         sim_static = torch.matmul(f_s_ref_proj, f_s_tar_mean.T)
            
    #         # Weighted Sum: Dynamic 성능을 보존하며 Static으로 미세 조정
    #         alpha = self.unchange_ratio # 배경 영향력을 보조적 수준으로 설정
    #         sim_i2t = sim_dynamic + alpha * sim_static

    #     return sim_i2t
    
    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # 인덱싱 시에는 모드와 상관없이 32개 쿼리 전체의 Raw Hidden State를 저장합니다.
        # 이렇게 해야 inference 시점에 자유롭게 모드를 전환할 수 있습니다.
        image_embeds = query_output.last_hidden_state # (bsz, 32, dim)
            
        return image_embeds, image_embeds_frozen
    
    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    # def load_checkpoint(self, url_or_filename):
    #     """
    #     Override load_checkpoint to handle temp parameter shape mismatch.
    #     """
    #     if is_url(url_or_filename):
    #         cached_file = download_cached_file(
    #             url_or_filename, check_hash=False, progress=True
    #         )
    #         checkpoint = torch.load(cached_file, map_location="cpu")
    #     elif os.path.isfile(url_or_filename):
    #         checkpoint = torch.load(url_or_filename, map_location="cpu")
    #     else:
    #         raise RuntimeError("checkpoint url or path is invalid")

    #     if "model" in checkpoint.keys():
    #         state_dict = checkpoint["model"]
    #     else:
    #         state_dict = checkpoint

    #     # Handle temp parameter shape mismatch (scalar vs 1D tensor)
    #     if "temp" in state_dict:
    #         if state_dict["temp"].shape != self.temp.shape:
    #             state_dict["temp"] = state_dict["temp"].reshape(self.temp.shape)

    #     msg = self.load_state_dict(state_dict, strict=False)

    #     logging.info("Missing keys {}".format(msg.missing_keys))
    #     logging.info("load checkpoint from %s" % url_or_filename)

    #     return msg
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
