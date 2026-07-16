import json
from pathlib import Path
import random
from typing import Union, List, Dict, Literal
import torchvision.transforms as T

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch 
from torchvision import transforms

# base_path = Path(__file__).absolute().parents[1].absolute()
base_path= Path('/home/summer24/sungonce/jyj/cir_dataset')
video_path = Path('/home/summer24/sungonce/jyj/elite_cir_dataset_full_B_v4')

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def expand2square(pil_img, background_color=(0, 0, 0)):
    """GeneCIS official padding logic: pad with black to make it square"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
# class MACIRDataset(Dataset):
#     """
#     MACIR dataset class for evaluation.
#     - 'classic' mode: returns (image_name, image) for indexing
#     - 'relative' mode: returns (ref_image, target_name, caption, condition_type) for evaluation
#     """
#     def __init__(self, json_path: Union[str, Path], img_dir: Union[str, Path], mode: str, preprocess: callable):
#         self.mode = mode
#         self.img_dir = Path(img_dir)
#         self.preprocess = preprocess
        
#         with open(json_path, 'r') as f:
#             self.data = json.load(f)
            
#         if mode == 'classic':
#             # 모든 유니크 이미지 추출 (Reference와 Target 모두 포함)
#             all_images = set()
#             for item in self.data:
#                 all_images.add(item['reference_image'])
#                 all_images.add(item['target_image'])
#             self.image_names = sorted(list(all_images))
#         elif mode == 'relative':
#             self.triplets = self.data
#         else:
#             raise ValueError("mode should be in ['relative', 'classic']")

#     def __getitem__(self, index):
#         try:
#             if self.mode == 'relative':
#                 item = self.triplets[index]
#                 ref_name = item['reference_image']
#                 tar_name = item['target_image']
#                 caption = item['relative_caption']
#                 condition = item['condition_type']
                
#                 ref_path = self.img_dir / ref_name
#                 ref_image = self.preprocess(PIL.Image.open(ref_path).convert("RGB"))
#                 return ref_image, tar_name, caption, condition

#             elif self.mode == 'classic':
#                 image_name = self.image_names[index]
#                 image_path = self.img_dir / image_name
#                 image = self.preprocess(PIL.Image.open(image_path).convert("RGB"))
#                 return image_name, image
#         except Exception as e:
#             print(f"Error loading data at index {index}: {e}")
#             return None

#     def __len__(self):
#         if self.mode == 'relative':
#             return len(self.triplets)
#         return len(self.image_names)

class MACIRDataset(Dataset):
    """
    MACIR dataset class for evaluation.
    - 'classic' mode: returns (image_name, image) for indexing
    - 'relative' mode: returns (ref_image, target_name, caption, condition_type, composition_type, ref_name) for evaluation
    """
    def __init__(self, json_path: Union[str, Path], img_dir: Union[str, Path], mode: str, preprocess: callable):
        self.mode = mode
        self.img_dir = Path(img_dir)
        self.preprocess = preprocess
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        if mode == 'classic':
            # 모든 유니크 이미지 추출 (Reference와 Target 모두 포함)
            # all_images = set()
            # for item in self.data:
            #     all_images.add(item['reference_image'])
            #     all_images.add(item['target_image'])
            # self.image_names = sorted(list(all_images))
            self.image_names = self.data
        elif mode == 'relative':
            self.triplets = self.data
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                item = self.triplets[index]
                ref_name = item['reference_image']
                tar_name = item['target_image']
                caption = item['relative_caption']
                condition = item['condition_type']
                comp_type = item['composition_type']
                
                ref_path = self.img_dir / ref_name
                ref_image = self.preprocess(PIL.Image.open(ref_path).convert("RGB"))
                return ref_image, tar_name, caption, condition, comp_type, ref_name

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.img_dir / image_name
                image = self.preprocess(PIL.Image.open(image_path).convert("RGB"))
                return image_name, image
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            return None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        return len(self.image_names)
    
class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashion-iq' / 'val_captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashion-iq' / 'image_splits_original' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = base_path / 'fashion-iq' / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashion-iq' / 'images' / f"{target_name}.jpg"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashion-iq' / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashion-iq' / 'val_images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'CIRR' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'CIRR' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path  / 'CIRR' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'CIRR' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path  / 'CIRR' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
    

class videoDataset(Dataset):
    def __init__(self, json_path, preprocess):
        self.preprocess = preprocess
        
        # 표준 JSON 리스트([]) 형식으로 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f) # json.loads(line) 대신 json.load(f) 사용
            except json.JSONDecodeError as e:
                print(f"JSON 파일 파싱 에러: {e}")
                raise e
        
        if not self.data:
            raise ValueError(f"데이터셋이 비어있습니다: {json_path}")
            
        print(f"Successfully loaded {len(self.data)} pairs. Total indexing size: {len(self.data) * 2}")

    def __len__(self):
        # 하나의 리스트 아이템당 Fwd/Bwd 2개씩 생성하므로 전체 길이는 2배
        return len(self.data) * 2

    def __getitem__(self, idx):
        real_idx = idx // 2
        is_backward = idx % 2 == 1
        
        item = self.data[real_idx]
        
        try:
            # 새로운 JSON 키값에 맞게 추출
            if not is_backward:
                # Forward: Ref -> Tar
                ref_path = item['ref_path']
                tar_path = item['tar_path']
                caption = item['forward_caption']
            else:
                # Backward: Tar -> Ref
                ref_path = item['tar_path']
                tar_path = item['ref_path']
                caption = item['backward_caption']

            # 이미지 로드 (base_path가 필요하다면 앞에 붙여주세요)
            # 예: img_path = base_path / ref_path
            ref = PIL.Image.open(video_path/ref_path).convert("RGB")
            tar = PIL.Image.open(video_path/tar_path).convert("RGB")
            
            ref = self.preprocess(ref)
            tar = self.preprocess(tar)
            
            # video_id를 pair_id로 사용하여 반환
            return ref, tar, caption, item['video_id']

        except Exception as e:
            # 에러 발생 시 로그를 남기고 무작위 인덱스로 재시도
            print(f"Index {idx} 로드 실패: {e}")
            return self.__getitem__(random.randint(0, (len(self.data) * 2) - 1))

class HardNegativeBatchSampler(Sampler):
    """
    인덱스 2i와 2i+1(동일 비디오의 Fwd/Bwd)을 항상 같은 배치에 인접하게 배치
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even to accommodate pairs.")
        
        # 실제 데이터 쌍(Video)의 개수
        self.num_pairs = len(dataset) // 2

    def __iter__(self):
        # 쌍 단위로 인덱스를 생성 (0, 1, 2, ..., num_pairs-1)
        pair_indices = list(range(self.num_pairs))
        if self.shuffle:
            random.shuffle(pair_indices)
        
        batch = []
        for p_idx in pair_indices:
            # 하나의 쌍에서 Fwd(2*p_idx)와 Bwd(2*p_idx + 1)를 모두 추가
            batch.append(2 * p_idx)
            batch.append(2 * p_idx + 1)
            
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = []
                
    def __len__(self):
        return (self.num_pairs * 2) // self.batch_size

class GeneCISDataset(Dataset):
    """
    GeneCIS dataset class for evaluation.
    Each item contains a reference image, a target image, a gallery of distractor images, and a condition text.
    Tasks: change_attribute, change_object, focus_attribute, focus_object
    """
    def __init__(self, data_path: Union[str, Path], preprocess: callable):
        self.data_path = Path(data_path)
        self.preprocess = preprocess
        self.tasks = ['change_attribute', 'change_object', 'focus_attribute', 'focus_object']
        self.triplets = []
        for task in self.tasks:
            json_path = self.data_path / f'{task}.json'
            if not json_path.exists():
                print(f"Warning: {json_path} not found. Skipping task {task}.")
                continue
            with open(json_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    item['task_name'] = task
                    self.triplets.append(item)
        
        self.coco_dir = self.data_path / 'val2017'
        self.vg_dir = self.data_path / 'VG_100K'
        
        print(f"GeneCIS dataset initialized with {len(self.triplets)} triplets across {len(self.tasks)} tasks.")

    def _load_image(self, item_info, task_name):
        # task_name contains 'object' -> COCO images, otherwise -> Visual Genome images
        if 'object' in task_name:
            image_id = item_info.get('val_image_id') or item_info.get('image_id')
            image_path = self.coco_dir / (str(image_id).zfill(12) + '.jpg')
        else:
            image_id = item_info.get('image_id') or item_info.get('val_image_id')
            image_path = self.vg_dir / f"{image_id}.jpg"
            
        try:
            image = PIL.Image.open(image_path).convert("RGB")
        except Exception as e:
            # Return a blank image if failed to load
            return self.preprocess(PIL.Image.new('RGB', (224, 224), (0, 0, 0)))
        
        # Handle bbox if present (Standard GeneCIS evaluation protocol for attribute tasks)
        if 'instance_bbox' in item_info:
            bbox = item_info['instance_bbox'] # [orig_left, orig_top, width, height]
            img_w, img_h = image.size
            
            # --- GeneCIS Official Logic (from vaw_dataset.py) ---
            # DILATION = 0.7
            dilate = 0.7
            orig_left, orig_top, w, h = bbox
            
            if w <= 0 or h <= 0:
                return self.preprocess(image)

            # Official dilation: expands top/left, keeping original right/bottom behavior
            left = max(0, orig_left - dilate * w)
            top = max(0, orig_top - dilate * h)
            right = min(img_w, left + (1 + dilate) * w)
            bottom = min(img_h, top + (1 + dilate) * h)
            
            if True:
                image = image.crop((left, top, right, bottom))
                # Official Black Padding
                bg_color = (0, 0, 0) if image.mode != 'L' else (0,)
                image = expand2square(image, bg_color)
            else:
                return self.preprocess(image)

            """
            # --- Previous Balanced Square Crop Logic (Rollback-able) ---
            dilation = 0.2
            dx = w * dilation
            dy = h * dilation
            x = orig_left - dx
            y = orig_top - dy
            box_w = w + 2 * dx
            box_h = h + 2 * dy
            
            max_side = max(box_w, box_h)
            cx = x + box_w / 2
            cy = y + box_h / 2
            x = cx - max_side / 2
            y = cy - max_side / 2
            
            left = max(0, x)
            top = max(0, y)
            right = min(img_w, x + max_side)
            bottom = min(img_h, y + max_side)
            image = image.crop((left, top, right, bottom))
            """
            
        return self.preprocess(image)

    def __getitem__(self, index):
        item = self.triplets[index]
        task_name = item['task_name']
        condition = item['condition']
        
        ref_image = self._load_image(item['reference'], task_name)
        target_image = self._load_image(item['target'], task_name)
        
        gallery_images = []
        for g in item['gallery']:
            gallery_images.append(self._load_image(g, task_name))
        
        # Combine gallery and target. In GeneCIS, we need to pick the target among candidates.
        # Gallery is the list of distractors.
        candidates = torch.stack(gallery_images + [target_image]) # [num_candidates, 3, 224, 224]
        target_idx = len(gallery_images) # Target is the last one
        
        return ref_image, candidates, target_idx, condition, task_name

    def __len__(self):
        return len(self.triplets)
    
class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
