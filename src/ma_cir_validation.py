import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from data_utils import targetpad_transform, MACIRDataset
from utils import collate_fn, extract_index_blip_features, device

def calculate_recalls(sims, targets, names_list):
    dists = 1 - sims
    sorted_indices = torch.argsort(dists, dim=-1).cpu()
    sorted_names = np.array(names_list)[sorted_indices]
    
    labels = torch.tensor(
        sorted_names == np.repeat(np.array(targets), len(names_list)).reshape(len(targets), -1))
    
    return {
        'recall_at1': (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        'recall_at5': (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        'recall_at10': (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        'recall_at50': (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    }

def compute_macir_val_metrics(relative_val_dataset: MACIRDataset, blip_model, index_features,
                             index_names: List[str], txt_processors, eval_mode='dynamic') -> dict:
    """
    Compute validation metrics on MACIR dataset including per-condition results
    """
    # Generate predictions
    pred_sim, target_names, conditions_all = generate_macir_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors, eval_mode)

    print(f"Compute MACIR validation metrics ({eval_mode} mode)")
    
    # 1. Overall Results
    results = calculate_recalls(pred_sim, target_names, index_names)
    
    # 2. Condition-wise Results
    conditions_all = np.array(conditions_all)
    target_names = np.array(target_names)
    unique_conditions = ['add', 'replace', 'remove']
    
    per_condition_results = {}
    for cond in unique_conditions:
        mask = (conditions_all == cond)
        if mask.sum() > 0:
            per_condition_results[cond] = calculate_recalls(pred_sim[mask], target_names[mask], index_names)

    return results, per_condition_results

def generate_macir_val_predictions(blip_model, relative_val_dataset: MACIRDataset, 
                                  index_names: List[str], index_features, txt_processors, eval_mode='dynamic') -> \
        Tuple[torch.tensor, List[str], List[str]]:
    """
    Compute MACIR predictions on the validation set
    """
    print(f"Compute MACIR validation predictions ({eval_mode} mode)")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=2,
                                     pin_memory=True, collate_fn=collate_fn)

    # index_features[0] are the projected features
    all_index_features = index_features[0].to(device)
    
    # Set model eval mode
    blip_model.eval_mode = eval_mode

    # Initialize predicted features, target_names and conditions
    distance = []
    target_names = []
    conditions_all = []

    for ref_imgs, batch_target_names, captions, conditions in tqdm(relative_val_loader):
        captions = [txt_processors["eval"](caption) for caption in captions]
        ref_imgs = ref_imgs.to(device)
        
        with torch.no_grad():
            with blip_model.maybe_autocast():
                ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(ref_imgs))
            ref_embeds = ref_embeds.float()
            batch_distance = blip_model.inference(ref_embeds, all_index_features, captions)
            distance.append(batch_distance.cpu())

        target_names.extend(batch_target_names)
        conditions_all.extend(conditions)
    
    distance = torch.vstack(distance)

    return distance, target_names, conditions_all

def blip_validate_macir(blip_model_name, backbone, model_path, json_path, img_dir, eval_mode):
    # 1. Load model and processors
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type=backbone, is_eval=False, device=device
    )
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        if blip_model.__class__.__name__ in checkpoint:
            state_dict = checkpoint[blip_model.__class__.__name__]
        else:
            state_dict = checkpoint
        msg = blip_model.load_state_dict(state_dict, strict=False)
        print("Missing keys {}".format(msg.missing_keys))
    
    blip_model.eval()
    blip_model.eval_mode = eval_mode

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    # 2. Datasets
    classic_dataset = MACIRDataset(json_path, img_dir, mode='classic', preprocess=preprocess)
    relative_dataset = MACIRDataset(json_path, img_dir, mode='relative', preprocess=preprocess)

    # 3. Extract Index Features
    print("Extracting MACIR index features...")
    index_features, index_names = extract_index_blip_features(classic_dataset, blip_model, save_memory=True)

    # 4. Compute Metrics
    results, per_condition_results = compute_macir_val_metrics(relative_dataset, blip_model, index_features, index_names, txt_processors, eval_mode)
    
    print("\n" + "="*50)
    print(f"--- MACIR Validation Results (Mode: {eval_mode}) ---")
    print("="*50)
    print(f"[OVERALL]\n{json.dumps(results, indent=4)}")
    
    for cond, res in per_condition_results.items():
        print(f"\n[{cond.upper()}]\n{json.dumps(res, indent=4)}")
    print("="*50)
    
    return results

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain_vitL", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--model-path", type=str, help="Path to trained checkpoint")
    parser.add_argument("--json-path", type=str, default="/home/junyeong/cir_dataset/MACIR_dataset/MACIR_dataset/meta_data/macir_meta_full.json")
    parser.add_argument("--img-dir", type=str, default="/home/junyeong/cir_dataset/MACIR_dataset/MACIR_dataset/images")
    parser.add_argument("--eval-mode", default="dynamic", choices=["dynamic", "fusion"], help="Evaluation mode")

    args = parser.parse_args()

    blip_validate_macir(args.blip_model_name, args.backbone, args.model_path, args.json_path, args.img_dir, args.eval_mode)
