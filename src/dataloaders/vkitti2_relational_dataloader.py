"""
VKITTI2 Relational Dataset
===========================
VKITTI2 데이터셋 + Relational Annotations + Caption Embeddings

기존 NYU dataloader (nyu_relational_dataloader.py)와 동일한 인터페이스 제공

Structure:
    vkitti_2.0.3/
    ├── Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
    ├── Scene01/clone/frames/depth/Camera_0/depth_00000.png
    └── ...
    
    vkitti2_relational/
    ├── Scene01_clone/
    │   ├── rgb_00000_masks.npy
    │   ├── rgb_00000_relations.json
    │   └── ...
    
    vkitti2_captions/
    ├── vkitti2_captions.json
    └── vkitti2_embeddings.npz
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import cv2
import platform


# Global cache for relations
VKITTI2_RELATIONS_CACHE = {}


class VKITTI2RelationalDataset(Dataset):
    """
    VKITTI2 Dataset with relational annotations
    
    NYU dataset과 동일한 출력 형식:
        - image: [3, H, W] normalized
        - depth: [1, H, W] in meters
        - masks: [N, H, W] object masks
        - relations: list of relation dicts
        - text_embedding: [768] CLIP embedding (optional)
    """
    
    SCENES = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    
    def __init__(
        self,
        vkitti2_root,
        relations_base_path=None,
        caption_cache_dir=None,
        filenames_file=None,
        is_train=True,
        scenes=None,
        conditions=None,
        input_height=352,
        input_width=704,
        max_depth=80.0,
        min_depth=0.1,
        use_relational_loss=True,
        use_text_embedding=True,
        do_random_rotate=False,
        degree=2.5,
        debug_relational=False,
        cache_images=False,
    ):
        """
        Args:
            vkitti2_root: VKITTI2 데이터셋 루트 경로
            relations_base_path: relational annotations 경로
            caption_cache_dir: caption/embedding 캐시 경로
            filenames_file: 파일 리스트 (optional, 없으면 자동 스캔)
            is_train: 학습 모드
            scenes: 처리할 scene 리스트
            conditions: 환경 조건 리스트 ['clone', 'fog', ...]
        """
        self.vkitti2_root = vkitti2_root
        self.relations_base_path = relations_base_path
        self.caption_cache_dir = caption_cache_dir
        self.is_train = is_train
        # scenes / conditions가 문자열 또는 "리스트 한 줄짜리 문자열"로 들어오는 경우 보정
        def _normalize_list_arg(val, default_list):
            # 이미 리스트고 정상 요소들이면 그대로 사용
            if isinstance(val, list):
                # ['Scene01', 'Scene02']처럼 정상인 경우
                ok = all(isinstance(v, str) and not (v.startswith('[') and v.endswith(']')) for v in val)
                if ok:
                    return val
                # ["['Scene01', 'Scene02']"] 같이 리스트 한 칸에 직렬화된 문자열이 들어온 경우 파싱
                if len(val) == 1 and isinstance(val[0], str):
                    raw = val[0]
                else:
                    return default_list
            elif isinstance(val, str):
                raw = val
            else:
                return default_list

            cleaned = raw.strip().strip('[]')
            parsed = [
                x.strip().strip("'").strip('"')
                for x in cleaned.split(',')
                if x.strip().strip("'").strip('"')
            ]
            return parsed or default_list

        scenes = _normalize_list_arg(scenes, self.SCENES)
        conditions = _normalize_list_arg(conditions, ['clone'])

        self.scenes = scenes or self.SCENES
        self.conditions = conditions or ['clone']
        
        self.input_height = input_height
        self.input_width = input_width
        self.max_depth = max_depth
        self.min_depth = min_depth
        
        self.use_relational_loss = use_relational_loss
        self.use_text_embedding = use_text_embedding
        self.do_random_rotate = do_random_rotate
        self.degree = degree
        self.debug_relational = debug_relational
        
        # 샘플 수집
        if filenames_file and os.path.exists(filenames_file):
            with open(filenames_file, 'r') as f:
                self.samples = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.samples = self._scan_samples()
        
        # Caption embeddings 로드
        self.captions = {}
        self.embeddings = {}
        if caption_cache_dir and use_text_embedding:
            self._load_caption_cache()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Image cache
        self.image_cache = {}
        if cache_images and is_train:
            self._preload_images()
        
        print(f"[VKITTI2] Loaded {len(self.samples)} samples")
        print(f"  Scenes: {self.scenes}")
        print(f"  Conditions: {self.conditions}")
        if self.use_relational_loss:
            print(f"  Relations path: {self.relations_base_path}")
        if self.use_text_embedding:
            print(f"  Embeddings: {len(self.embeddings)} loaded")
    
    def _scan_samples(self):
        """RGB-Depth 쌍 스캔"""
        samples = []
        
        for scene in self.scenes:
            for condition in self.conditions:
                rgb_dir = os.path.join(
                    self.vkitti2_root, scene, condition,
                    'frames', 'rgb', 'Camera_0'
                )
                
                if not os.path.exists(rgb_dir):
                    continue
                
                rgb_files = sorted(glob.glob(os.path.join(rgb_dir, 'rgb_*.jpg')))
                
                for rgb_path in rgb_files:
                    fname = os.path.basename(rgb_path)
                    frame_id = fname.replace('rgb_', '').replace('.jpg', '')
                    
                    # sample_path format: "Scene01/clone/00000"
                    sample_path = f"{scene}/{condition}/{frame_id}"
                    samples.append(sample_path)
        
        return samples
    
    def _load_caption_cache(self):
        """Caption/Embedding 캐시 로드"""
        caption_path = os.path.join(self.caption_cache_dir, 'vkitti2_captions.json')
        emb_path = os.path.join(self.caption_cache_dir, 'vkitti2_embeddings.npz')
        
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                self.captions = json.load(f)
            print(f"[Caption] Loaded {len(self.captions)} captions")
        
        if os.path.exists(emb_path):
            emb_data = np.load(emb_path)
            self.embeddings = {k: emb_data[k] for k in emb_data.files}
            print(f"[Embedding] Loaded {len(self.embeddings)} embeddings")
            # 키 형식 검증: 샘플 몇 개로 caption_key가 캐시에 있는지 확인
            if self.samples and self.embeddings:
                missing = []
                for sp in self.samples[:20]:
                    parts = sp.split("/")
                    if len(parts) >= 3:
                        ck = f"{parts[0]}/{parts[1]}/rgb_{parts[2]}.jpg"
                        if ck not in self.embeddings:
                            missing.append(ck)
                if missing:
                    print(f"[경고] 일부 샘플의 caption 키가 캐시에 없음 (키 형식 불일치 가능성). 예: {missing[0]}")
                else:
                    print("[Embedding] 키 형식 검증 OK (샘플 20개 기준)")
    
    def _preload_images(self):
        """이미지 캐시 (optional)"""
        from tqdm import tqdm
        print(f"[VKITTI2] Caching {len(self.samples)} images...")
        
        for sample_path in tqdm(self.samples, desc="Caching"):
            try:
                rgb_path, depth_path, _ = self._resolve_paths(sample_path)
                
                if rgb_path not in self.image_cache:
                    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.image_cache[rgb_path] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if depth_path not in self.image_cache:
                    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if depth is not None:
                        self.image_cache[depth_path] = depth
            except Exception:
                pass
    
    def _resolve_paths(self, sample_path):
        """
        sample_path에서 실제 파일 경로 추출
        
        sample_path format: "Scene01/clone/00000"
        """
        parts = sample_path.split('/')
        
        if len(parts) == 3:
            scene, condition, frame_id = parts
        else:
            # Fallback: try to parse
            scene = parts[0]
            condition = parts[1] if len(parts) > 1 else 'clone'
            frame_id = parts[-1].replace('rgb_', '').replace('.jpg', '').replace('.png', '')
        
        rgb_path = os.path.join(
            self.vkitti2_root, scene, condition,
            'frames', 'rgb', 'Camera_0', f'rgb_{frame_id}.jpg'
        )
        depth_path = os.path.join(
            self.vkitti2_root, scene, condition,
            'frames', 'depth', 'Camera_0', f'depth_{frame_id}.png'
        )
        
        return rgb_path, depth_path, f"{scene}_{condition}"
    
    def _load_relational_annotations(self, scene_condition, frame_id):
        """Masks와 Relations 로드"""
        if not self.relations_base_path:
            return None, None
        
        mask_path = os.path.join(
            self.relations_base_path, scene_condition,
            f"rgb_{frame_id}_masks.npy"
        )
        rel_path = os.path.join(
            self.relations_base_path, scene_condition,
            f"rgb_{frame_id}_relations.json"
        )
        
        # Masks
        masks_tensor = None
        if os.path.exists(mask_path):
            try:
                masks_np = np.load(mask_path)
                
                # Resize if needed
                if masks_np.shape[1:] != (self.input_height, self.input_width):
                    import torch.nn.functional as F
                    masks_t = torch.from_numpy(masks_np).float().unsqueeze(1)
                    masks_t = F.interpolate(
                        masks_t,
                        size=(self.input_height, self.input_width),
                        mode='nearest'
                    )
                    masks_tensor = masks_t.squeeze(1)
                else:
                    masks_tensor = torch.from_numpy(masks_np).float()
                
                masks_tensor = (masks_tensor > 0.5).float()
            except Exception as e:
                if self.debug_relational:
                    print(f"[VKITTI2] Failed to load masks: {e}")
        
        # Relations
        relations = None
        global VKITTI2_RELATIONS_CACHE
        
        if os.path.exists(rel_path):
            try:
                if rel_path not in VKITTI2_RELATIONS_CACHE:
                    with open(rel_path, 'r') as f:
                        VKITTI2_RELATIONS_CACHE[rel_path] = json.load(f)
                relations = VKITTI2_RELATIONS_CACHE[rel_path]
            except Exception as e:
                if self.debug_relational:
                    print(f"[VKITTI2] Failed to load relations: {e}")
        
        return masks_tensor, relations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        rgb_path, depth_path, scene_condition = self._resolve_paths(sample_path)
        
        # Frame ID 추출
        frame_id = sample_path.split('/')[-1]
        
        # Load image
        if rgb_path in self.image_cache:
            image = self.image_cache[rgb_path].copy()
        else:
            image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth (VKITTI2: cm in uint16)
        if depth_path in self.image_cache:
            depth = self.image_cache[depth_path].copy().astype(np.float32)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32)
        
        depth = depth / 100.0  # cm -> m
        
        # Center crop to input size
        h, w = image.shape[:2]
        crop_h, crop_w = self.input_height, self.input_width
        
        if h > crop_h or w > crop_w:
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            image = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
            depth = depth[start_h:start_h+crop_h, start_w:start_w+crop_w]
        elif h < crop_h or w < crop_w:
            # Resize if smaller
            image = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        # Valid mask
        valid_mask = (depth > self.min_depth) & (depth < self.max_depth)
        
        # Training augmentations
        if self.is_train:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                depth = depth[:, ::-1].copy()
                valid_mask = valid_mask[:, ::-1].copy()
            
            # Random augmentation
            if np.random.random() > 0.5:
                image = self._augment_image(image)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Depth to tensor
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        valid_mask = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)
        
        sample = {
            'image': image,
            'depth': depth,
            'has_valid_depth': torch.tensor(True),
            'sample_path': sample_path,
        }
        
        # Text embedding (키 형식: SceneXX/condition/rgb_XXXXX.jpg — generate_vkitti2_captions.py 출력과 동일)
        if self.use_text_embedding:
            caption_key = f"{sample_path.split('/')[0]}/{sample_path.split('/')[1]}/rgb_{frame_id}.jpg"
            
            if caption_key in self.embeddings:
                text_emb = torch.from_numpy(self.embeddings[caption_key]).float()
            else:
                # Fallback: zero embedding
                text_emb = torch.zeros(768)
            
            sample['text_embedding'] = text_emb
            sample['caption'] = self.captions.get(caption_key, "a driving scene with road")
        
        # Relational annotations
        if self.use_relational_loss:
            masks, relations = self._load_relational_annotations(scene_condition, frame_id)
            sample['masks'] = masks
            sample['relations'] = relations if relations else []
        
        return sample
    
    def _augment_image(self, image):
        """Data augmentation"""
        # Gamma
        gamma = np.random.uniform(0.9, 1.1)
        image = np.power(image / 255.0, gamma) * 255.0
        
        # Brightness
        brightness = np.random.uniform(0.9, 1.1)
        image = image * brightness
        
        # Color
        colors = np.random.uniform(0.9, 1.1, size=3)
        image = image * colors
        
        return np.clip(image, 0, 255).astype(np.uint8)


def vkitti2_collate_fn(batch):
    """
    VKITTI2 배치 collate function
    masks와 relations는 가변 길이이므로 리스트로 유지
    """
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    has_valid_depth = torch.stack([item['has_valid_depth'] for item in batch])
    sample_paths = [item['sample_path'] for item in batch]
    
    result = {
        'image': images,
        'depth': depths,
        'has_valid_depth': has_valid_depth,
        'sample_path': sample_paths,
    }
    
    # Text embeddings
    if 'text_embedding' in batch[0]:
        result['text_embedding'] = torch.stack([item['text_embedding'] for item in batch])
        result['caption'] = [item.get('caption', '') for item in batch]
    
    # Masks and relations (variable size)
    if 'masks' in batch[0]:
        masks_list = []
        relations_list = []
        
        for item in batch:
            masks = item.get('masks')
            relations = item.get('relations', [])
            
            if masks is None:
                masks = torch.zeros((0, item['image'].shape[1], item['image'].shape[2]))
            
            masks_list.append(masks)
            relations_list.append(relations if relations else [])
        
        result['masks'] = masks_list
        result['relations'] = relations_list
    
    return result


def create_vkitti2_dataloader(args, mode='train', use_ddp=False):
    """
    VKITTI2 DataLoader 생성
    
    NYU dataloader와 동일한 인터페이스
    """
    is_train = (mode == 'train')
    
    # 경로 설정 (config의 여러 가능한 key 지원)
    vkitti2_root = getattr(args, 'vkitti2_data_path', None) or getattr(args, 'vkitti2_root', './data/vkitti_2.0.3')
    relations_path = getattr(args, 'vkitti2_relations_dir', None)
    caption_cache = getattr(args, 'vkitti2_caption_cache', None)
    
    # Scene/Condition
    scenes = getattr(args, 'vkitti2_scenes', None)
    conditions = getattr(args, 'vkitti2_conditions', ['clone'])
    
    # Filenames file (optional)
    filenames_file = getattr(args, 'vkitti2_filenames_file', None)
    
    use_relational = getattr(args, 'use_relational_loss', False)
    use_text_emb = not getattr(args, 'baseline_mode', False)
    
    # Input size: VKITTI2 specific 또는 global
    input_height = getattr(args, 'vkitti2_input_height', None) or getattr(args, 'input_height', 352)
    input_width = getattr(args, 'vkitti2_input_width', None) or getattr(args, 'input_width', 704)
    max_depth = getattr(args, 'vkitti2_max_depth', None) or getattr(args, 'max_depth', 80.0)
    min_depth = getattr(args, 'vkitti2_min_depth', None) or getattr(args, 'min_depth', 0.1)
    
    dataset = VKITTI2RelationalDataset(
        vkitti2_root=vkitti2_root,
        relations_base_path=relations_path if use_relational else None,
        caption_cache_dir=caption_cache if use_text_emb else None,
        filenames_file=filenames_file,
        is_train=is_train,
        scenes=scenes,
        conditions=conditions,
        input_height=input_height,
        input_width=input_width,
        max_depth=max_depth,
        min_depth=min_depth,
        use_relational_loss=use_relational,
        use_text_embedding=use_text_emb,
        do_random_rotate=getattr(args, 'do_random_rotate', False),
        degree=getattr(args, 'degree', 2.5),
        debug_relational=getattr(args, 'debug_relational', False),
        cache_images=getattr(args, 'cache_images', False),
    )
    
    # Sampler for DDP
    train_sampler = None
    if is_train and use_ddp:
        try:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(dataset)
        except Exception:
            pass
    
    # Workers
    num_workers = 0 if platform.system() == 'Windows' else getattr(args, 'num_threads', 4)
    prefetch = 4 if num_workers > 0 and is_train else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=(train_sampler is None) and is_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        collate_fn=vkitti2_collate_fn if use_relational else None
    )
    
    # Wrapper for compatibility
    class DataLoaderWrapper:
        def __init__(self, dl, sampler=None):
            self.data = dl
            self.train_sampler = sampler
    
    return DataLoaderWrapper(dataloader, train_sampler)


# ============================================================
# 테스트
# ============================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vkitti2_root', type=str, default='./data/vkitti_2.0.3')
    parser.add_argument('--relations_dir', type=str, default='./data/vkitti2_relational')
    parser.add_argument('--caption_cache', type=str, default='./data/vkitti2_captions')
    args = parser.parse_args()
    
    # Test dataset
    dataset = VKITTI2RelationalDataset(
        vkitti2_root=args.vkitti2_root,
        relations_base_path=args.relations_dir,
        caption_cache_dir=args.caption_cache,
        is_train=True,
        scenes=['Scene01'],
        conditions=['clone'],
        use_relational_loss=True,
        use_text_embedding=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Depth range: [{sample['depth'].min():.2f}, {sample['depth'].max():.2f}]")
    
    if 'text_embedding' in sample:
        print(f"Text embedding: {sample['text_embedding'].shape}")
        print(f"Caption: {sample.get('caption', 'N/A')}")
    
    if 'masks' in sample and sample['masks'] is not None:
        print(f"Masks shape: {sample['masks'].shape}")
        print(f"Relations: {len(sample['relations'])}")

