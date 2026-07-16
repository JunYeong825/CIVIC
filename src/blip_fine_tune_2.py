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
import random

from data_utils import HardNegativeBatchSampler, videoDataset, base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results_dict, set_train_bar_description_dict, extract_index_blip_features, \
    save_model, device
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

def train_blip_cir(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                   validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, 
                   dataset: str, **kwargs):
    """
    Unified training function for Intra-video based learning
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/blip2_cir_{dataset}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif transform == "targetpad":
        preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # 1. Training Dataset: Always use Video Triplet Dataset (Intra-video learning)
    relative_train_dataset = videoDataset(
        Path('/home/summer24/sungonce/jyj/elite_cir_dataset_full_B_v4/metadata/captioned_v8_merged_valid_data.json'), 
        preprocess
    )

    # 2. Hard Negative Batch Sampler (Research Core)
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

    # [Ablation Point] Standard DataLoader (Random Batching) - Uncomment for comparison
    # relative_train_loader = DataLoader(
    #     dataset=relative_train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=kwargs['num_workers'],
    #     pin_memory=True,
    #     collate_fn=collate_fn
    # )

    # 3. Validation Datasets Setup
    val_data_dict = {}
    if dataset.lower() == 'cirr':
        val_data_dict['relative'] = CIRRDataset('val', 'relative', preprocess)
        val_data_dict['classic'] = CIRRDataset('val', 'classic', preprocess)
    elif dataset.lower() == 'fashioniq':
        val_dress_types = ['dress', 'toptee', 'shirt']
        val_data_dict['sets'] = []
        for dt in val_dress_types:
            val_data_dict['sets'].append({
                'type': dt,
                'relative': FashionIQDataset('val', [dt], 'relative', preprocess),
                'classic': FashionIQDataset('val', [dt], 'classic', preprocess)
            })

    # Optimizer, Scheduler, Scaler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_metric = 0
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    print(f'Intra-video Training started for {dataset} benchmark')
    for epoch in range(num_epochs):
        blip_model.train()
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)

        for idx, (reference_images, target_images, captions, _) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image": reference_images, "target": target_images, "text_input": captions})
                total_loss = loss_dict['loss_itc'] + kwargs['loss_bg'] * loss_dict['loss_bg']
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Logging Training Metrics
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # 4. Validation Step
        if epoch % validation_frequency == 0:
            blip_model.eval()
            blip_model.eval_mode = kwargs['eval_mode']
            results_dict = {'epoch': epoch}
            
            if dataset.lower() == 'cirr':
                idx_feats, idx_names = extract_index_blip_features(val_data_dict['classic'], blip_model, save_memory)
                results = compute_cirr_val_metrics(val_data_dict['relative'], blip_model, idx_feats, idx_names, txt_processors)
                group_r1, _, _, r1, r5, r10, r50 = results
                results_dict.update({'group_recall_at1': group_r1, 'recall_at5': r5, 'arithmetic_mean': mean(results)})
                cur_metric = results_dict['arithmetic_mean']
            
            elif dataset.lower() == 'fashioniq':
                recalls_at10, recalls_at50 = [], []
                for vset in val_data_dict['sets']:
                    idx_feats, idx_names = extract_index_blip_features(vset['classic'], blip_model, save_memory)
                    r10, r50 = compute_fiq_val_metrics(vset['relative'], blip_model, idx_feats, idx_names, txt_processors, save_memory)
                    recalls_at10.append(r10); recalls_at50.append(r50)
                    results_dict[f"{vset['type']}_r10"] = r10
                results_dict.update({'average_recall_at10': mean(recalls_at10), 'average_recall_at50': mean(recalls_at50)})
                cur_metric = (mean(recalls_at10) + mean(recalls_at50)) / 2

            print(json.dumps(results_dict, indent=4))
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=results_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training and cur_metric > best_metric:
                best_metric = cur_metric
                save_model(f'tuned_blip_best_{dataset}', epoch, blip_model, training_path)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="CIRR or FashionIQ")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain_vitL")
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--loss-bg", default=0.1, type=float, help="Background Invariance Loss weight")
    parser.add_argument("--eval-mode", default="dynamic", type=str, help="dynamic or fusion")
    parser.add_argument("--validation-frequency", default=1, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    parser.add_argument("--save-training", action='store_true')
    parser.add_argument("--save-best", action='store_true')
    parser.add_argument("--save-memory", action='store_true')

    args = parser.parse_args()
    
    training_hyper_params = vars(args)
    
    set_seed(912)
    
    # Start Unified Training
    train_blip_cir(**training_hyper_params)
