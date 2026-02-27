"""
nyu_relational_dataset.py
NYU-Depth-v2 Dataset with Relational Annotations
WorDepth + RelationalDepthLoss를 위한 Dataset 클래스
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import platform
import cv2


# Global cache for relational annotations (relations.json)
RELATIONS_CACHE = {}
RELATIONS_CACHE_STATS = {'hits': 0, 'misses': 0}


def preload_relations_cache(relations_base_path, filenames_file):
    """
    Preload all relations.json files into cache for faster training
    
    Args:
        relations_base_path: path to relations directory
        filenames_file: path to filenames file to know which files to preload
    """
    import os
    import json
    
    print("Preloading relations cache...")
    preloaded_count = 0
    
    # Read filenames to know which relations files to preload
    with open(filenames_file, 'r') as f:
        filenames = f.readlines()
    
    for filename in filenames:
        filename = filename.strip()
        if not filename:
            continue
            
        # Parse filename to get scene and rgb name
        parts = filename.split()
        rgb_file = parts[0].lstrip('/')
        
        # Normalize to forward slash
        rgb_file_unix = rgb_file.replace(os.sep, '/')
        
        # Construct scene name
        scene_name = os.path.dirname(rgb_file_unix)
        first_seg = scene_name.split('/', 1)[0]
        if first_seg in ('train', 'test'):
            scene_name = scene_name.split('/', 1)[1] if '/' in scene_name else ''
        
        rgb_basename = os.path.basename(rgb_file_unix).replace('.jpg', '').replace('.png', '')
        rel_path = os.path.join(relations_base_path, scene_name, f"{rgb_basename}_relations.json")
        
        # Load into cache if not already loaded
        if rel_path not in RELATIONS_CACHE and os.path.exists(rel_path):
            try:
                with open(rel_path, 'r') as f:
                    RELATIONS_CACHE[rel_path] = json.load(f)
                preloaded_count += 1
            except Exception as e:
                print(f"Warning: Failed to preload {rel_path}: {e}")
    
    print(f"Preloaded {preloaded_count} relations files into cache")
    return preloaded_count


class NYURelationalDataset(Dataset):
    """
    NYU-Depth-v2 with relational annotations (masks + relations)
    
    Structure:
        nyu-depth-v2/train/scene_name/rgb_XXXXX.png
        nyu-depth-v2/train/scene_name/depth_XXXXX.png
        nyu-processed/train/scene_name/rgb_XXXXX_masks.npy
        nyu-processed/train/scene_name/rgb_XXXXX_relations.json
    """
    
    def __init__(self, 
                 filenames_file=None,  # Optional: if None, auto-scan directories
                 data_path=None,
                 gt_path=None,
                 relations_base_path=None,
                 is_train=True,
                 input_height=480,
                 input_width=640,
                 max_depth=10.0,
                 do_random_rotate=False,
                 degree=2.5,
                 use_relational_loss=True,
                 debug_relational=False,
                 depth_in_mm=True,
                 use_dense_depth=False,
                 cache_images=False,
                 depth_scale=None):
        """
        Args:
            filenames_file: path to filenames list (optional, if None auto-scans directories)
            data_path: path to RGB images (nyu-depth-v2/train)
            gt_path: path to depth GT (same as data_path for NYU)
            relations_base_path: path to processed relations (nyu-processed/train)
            is_train: training mode
            use_relational_loss: whether to load masks and relations
            depth_in_mm: if True (WorDepth convention), depth file is in mm, load as /1000 -> m.
                         if False (matched), depth file is 0-65535 scale, load as /6553.5 -> m.
            use_dense_depth: if True, load depth from <scene>/dense/sync_depth_dense_XXXXX.png and keep rgb as .jpg.
            cache_images: if True and is_train, preload all RGB/depth into memory for faster iteration.
            depth_scale: divisor for raw depth -> m (default 1000 if depth_in_mm else 6553.5).
        """
        self.is_train = is_train
        self.depth_in_mm = depth_in_mm
        self.use_dense_depth = use_dense_depth
        self.depth_scale = depth_scale if depth_scale is not None else (1000.0 if depth_in_mm else 6553.5)
        # Convert relative paths to absolute paths for Windows compatibility
        self.data_path = os.path.abspath(data_path) if data_path and not os.path.isabs(data_path) else data_path
        self.gt_path = os.path.abspath(gt_path) if gt_path and not os.path.isabs(gt_path) else gt_path
        self.relations_base_path = relations_base_path
        if relations_base_path and not os.path.isabs(relations_base_path):
            self.relations_base_path = os.path.abspath(relations_base_path)
        self.use_relational_loss = use_relational_loss
        self.debug_relational = debug_relational
        
        self.input_height = input_height
        self.input_width = input_width
        self.max_depth = max_depth
        
        self.do_random_rotate = do_random_rotate
        self.degree = degree
        
        # Auto-scan directories if filenames_file not provided
        if filenames_file is None or filenames_file == '':
            self.filenames = self._auto_scan_rgb_depth_pairs()
        else:
            # Read filenames from file
            with open(filenames_file, 'r') as f:
                self.filenames = f.readlines()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Image cache (optional): same as paper (dataloader.py DataLoadPreprocess).
        # Single dict: image_path -> RGB numpy, depth_path -> raw depth numpy (cv2 IMREAD_UNCHANGED).
        self.image_cache = {}
        if cache_images and is_train and len(self.filenames) > 0:
            from tqdm import tqdm
            for idx in tqdm(range(len(self.filenames)), desc="Loading images"):
                try:
                    image_path, depth_path, _ = self._resolve_paths(idx)
                    if image_path not in self.image_cache:
                        image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        if image_cv is not None:
                            self.image_cache[image_path] = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    if depth_path not in self.image_cache:
                        depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        if depth_cv is not None:
                            self.image_cache[depth_path] = depth_cv
                except (FileNotFoundError, IndexError):
                    pass
            n_pairs = len(self.image_cache) // 2
            est_gb = len(self.image_cache) * 640 * 480 * 2 / (1024 ** 3)
            print(f"Cached {n_pairs} image pairs (~{est_gb:.1f} GB)")
    
    def _resolve_paths(self, idx):
        """Resolve (image_path, depth_path, rgb_file) for the given index. Used by cache fill and __getitem__."""
        sample_path = self.filenames[idx].strip()
        parts = sample_path.split()
        rgb_file = parts[0].lstrip('/')
        depth_file = parts[1].lstrip('/') if len(parts) > 1 else None
        rgb_file = rgb_file.replace('/', os.sep)
        if depth_file:
            depth_file = depth_file.replace('/', os.sep)
        if self.use_dense_depth:
            if not rgb_file.endswith('.jpg') and not rgb_file.endswith('.png'):
                rgb_file = rgb_file + '.jpg'
            base_name = os.path.basename(rgb_file)
            num = base_name.replace('rgb_', '').replace('.jpg', '').replace('.png', '') if base_name.startswith('rgb_') else base_name.replace('.jpg', '').replace('.png', '')
            scene_name = os.path.dirname(rgb_file)
            depth_file = os.path.join(scene_name, 'dense', f'sync_depth_dense_{num}.png')
        else:
            if rgb_file.endswith('.jpg'):
                rgb_file = rgb_file.replace('.jpg', '.png')
            elif not rgb_file.endswith('.png'):
                rgb_file = rgb_file + '.png'
            if depth_file:
                if 'sync_depth_' in depth_file:
                    depth_file = depth_file.replace('sync_depth_', 'depth_')
                if depth_file.endswith('.jpg'):
                    depth_file = depth_file.replace('.jpg', '.png')
                elif not depth_file.endswith('.png'):
                    depth_file = depth_file + '.png'
        image_path = os.path.normpath(os.path.join(self.data_path, rgb_file))
        depth_path = os.path.normpath(os.path.join(self.gt_path, depth_file or rgb_file.replace('rgb_', 'depth_')))
        if not self.use_dense_depth and not os.path.exists(depth_path) and 'sync_depth_' in depth_path:
            depth_path_alt = depth_path.replace('sync_depth_', 'depth_')
            if os.path.exists(depth_path_alt):
                depth_path = depth_path_alt
        if not os.path.exists(image_path) and image_path.endswith('.png'):
            image_path_jpg = image_path.replace('.png', '.jpg')
            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
        if not os.path.exists(image_path):
            scene_dir = os.path.dirname(image_path)
            if os.path.exists(scene_dir):
                rgb_files = [f for f in os.listdir(scene_dir) if f.startswith('rgb_') and (f.endswith('.png') or f.endswith('.jpg'))]
                if rgb_files:
                    image_path = os.path.join(scene_dir, rgb_files[0])
                    rgb_file = os.path.join(os.path.basename(scene_dir), rgb_files[0]).replace(os.sep, '/')
                else:
                    raise FileNotFoundError(f"RGB file not found: {image_path}")
            else:
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        if not os.path.exists(depth_path):
            scene_dir = os.path.dirname(depth_path)
            if os.path.exists(scene_dir):
                # Support multiple NYU depth naming conventions:
                # - depth_XXXXX.png
                # - sync_depth_XXXXX.png
                # - sync_depth_dense_XXXXX.png (converted dense depth)
                candidates = []
                for fname in os.listdir(scene_dir):
                    if not fname.endswith(".png"):
                        continue
                    if fname.startswith("depth_") or fname.startswith("sync_depth_"):
                        candidates.append(fname)
                depth_files = sorted(candidates)
                if depth_files:
                    rgb_basename = os.path.basename(image_path).replace("rgb_", "").replace(".png", "").replace(".jpg", "")
                    matching = [f for f in depth_files if rgb_basename in f]
                    chosen = matching[0] if matching else depth_files[0]
                    depth_path = os.path.join(scene_dir, chosen)
                else:
                    raise FileNotFoundError(f"Depth file not found: {depth_path}")
            else:
                raise FileNotFoundError(f"Scene directory not found: {os.path.dirname(depth_path)}")
        return image_path, depth_path, rgb_file.replace(os.sep, '/')
    
    def _auto_scan_rgb_depth_pairs(self):
        """
        Automatically scan directories and match RGB-Depth pairs by filename.
        Returns list of pairs in format: "scene_name/rgb_XXXXX.png scene_name/depth_XXXXX.png"
        """
        pairs = []
        
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Scan all scene directories
        scene_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Auto-scanning {len(scene_dirs)} scenes in {self.data_path}...")
        
        for scene_name in sorted(scene_dirs):
            scene_dir = os.path.join(self.data_path, scene_name)
            
            # Find all RGB files (PNG only - actual data format)
            rgb_files = [f for f in os.listdir(scene_dir) 
                        if f.startswith('rgb_') and f.endswith('.png')]
            
            # Find all depth files (PNG only - actual data format)
            depth_files = [f for f in os.listdir(scene_dir) 
                          if f.startswith('depth_') and f.endswith('.png')]
            
            # Match RGB and Depth by number
            for rgb_file in sorted(rgb_files):
                # Extract number from rgb_XXXXX.png
                import re
                rgb_match = re.search(r'rgb_(\d+)', rgb_file)
                if not rgb_match:
                    continue
                rgb_num = rgb_match.group(1)
                
                # Find matching depth file
                depth_file = None
                for d_file in depth_files:
                    if f'depth_{rgb_num}' in d_file or f'sync_depth_{rgb_num}' in d_file:
                        depth_file = d_file
                        break
                
                if depth_file:
                    # Format: "scene_name/rgb_XXXXX.png scene_name/depth_XXXXX.png"
                    pair = f"{scene_name}/{rgb_file} {scene_name}/{depth_file}"
                    pairs.append(pair)
        
        print(f"Found {len(pairs)} RGB-Depth pairs")
        return pairs
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx].strip()
        image_path, depth_path, rgb_file = self._resolve_paths(idx)

        # Load image and depth from cache or disk (cache read identical to paper DataLoadPreprocess)
        if image_path in self.image_cache and depth_path in self.image_cache:
            image = Image.fromarray(self.image_cache[image_path])  # paper: RGB stored in cache
            depth_gt = Image.fromarray(self.image_cache[depth_path])  # paper: raw in cache
            depth_gt = np.array(depth_gt, dtype=np.float32) / self.depth_scale
            depth_gt = np.expand_dims(depth_gt, axis=0)  # (1, H, W)
        else:
            image = Image.open(image_path).convert('RGB')
            depth_gt = np.array(Image.open(depth_path), dtype=np.float32) / self.depth_scale
            depth_gt = np.expand_dims(depth_gt, axis=0)  # (1, H, W)
        
        # Resize if needed
        if image.size != (self.input_width, self.input_height):
            image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
            depth_gt_pil = Image.fromarray(depth_gt[0])
            depth_gt_pil = depth_gt_pil.resize((self.input_width, self.input_height), Image.NEAREST)
            depth_gt = np.array(depth_gt_pil)[np.newaxis, :, :]
        
        # Random rotation (training only)
        if self.is_train and self.do_random_rotate:
            angle = (np.random.random() - 0.5) * 2 * self.degree
            image = image.rotate(angle, resample=Image.BILINEAR)
            depth_gt_pil = Image.fromarray(depth_gt[0])
            depth_gt_pil = depth_gt_pil.rotate(angle, resample=Image.NEAREST)
            depth_gt = np.array(depth_gt_pil)[np.newaxis, :, :]
        
        # To tensor
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth_gt = torch.from_numpy(depth_gt).float()
        
        # Valid depth mask
        has_valid_depth = torch.any(depth_gt > 0.1)
        
        # For text feature loading in train.py, use the format expected by existing code
        # train.py expects: sample_path in format "/scene/rgb_XXXXX.jpg /scene/depth_XXXXX.png"
        # We keep sample_path as original, but use rgb_file (actual loaded file) for relations
        
        # Base sample
        sample = {
            'image': image,
            'depth': depth_gt,
            'sample_path': sample_path,  # Original path from filenames_file (for text feature)
            'has_valid_depth': has_valid_depth
        }
        
        # Load relational annotations if needed
        if self.use_relational_loss and self.relations_base_path:
            masks, relations = self._load_relational_annotations(rgb_file)
            # 디버깅: 마스크/관계 정보 shape, 길이, 값 확인 (옵션)
            if self.debug_relational:
                if masks is not None:
                    try:
                        print(f"[RelAnn] {rgb_file} masks shape: {masks.shape}, min: {masks.min().item():.4f}, max: {masks.max().item():.4f}, N_obj: {masks.shape[0]}")
                    except Exception:
                        print(f"[RelAnn] {rgb_file} masks shape: {getattr(masks,'shape',None)}")
                else:
                    print(f"[RelAnn] {rgb_file} masks: None")
                if relations is not None:
                    print(f"[RelAnn] {rgb_file} relations: {len(relations)}")
                else:
                    print(f"[RelAnn] {rgb_file} relations: None")
            sample['masks'] = masks
            sample['relations'] = relations
        
        return sample
    
    def _load_relational_annotations(self, rgb_file):
        """
        Load masks and relations for a given RGB file
        
        Args:
            rgb_file: e.g., "kitchen_0028b/rgb_00045.png" (OS-specific separator)
        
        Returns:
            masks: torch.Tensor (N_obj, H, W) or None
            relations: list[dict] or None
        """
        # Normalize to forward slash for path construction
        rgb_file_unix = rgb_file.replace(os.sep, '/')

        # Construct paths
        scene_name = os.path.dirname(rgb_file_unix)
        rgb_basename = os.path.basename(rgb_file_unix).replace('.jpg', '').replace('.png', '')

        # Remove duplicated dataset split prefix if present (e.g., filenames use 'train/scene/..' but
        # relations_base_path already points to the '.../train' folder). This avoids paths like
        #  .../nyu-processed-matched/train/train/scene/...
        first_seg = scene_name.split('/', 1)[0]
        if first_seg in ('train', 'test'):
            # drop the leading 'train/' or 'test/'
            scene_name = scene_name.split('/', 1)[1] if '/' in scene_name else ''

        mask_path = os.path.join(
            self.relations_base_path,
            scene_name,
            f"{rgb_basename}_masks.npy"
        )
        rel_path = os.path.join(
            self.relations_base_path,
            scene_name,
            f"{rgb_basename}_relations.json"
        )

        # Debug: show the constructed paths and whether files exist (only when debugging enabled)
        try:
            if getattr(self, 'debug_relational', False):
                print(f"[RelAnn.path] mask_path={mask_path}, exists={os.path.exists(mask_path)}")
                print(f"[RelAnn.path] rel_path={rel_path}, exists={os.path.exists(rel_path)}")
        except Exception:
            pass
        
        # Load masks
        try:
            masks_np = np.load(mask_path)  # (N_obj, H, W)
            
            # Resize masks if needed
            if masks_np.shape[1:] != (self.input_height, self.input_width):
                import torch.nn.functional as F
                masks_tensor = torch.from_numpy(masks_np).float().unsqueeze(1)  # (N, 1, H, W)
                masks_tensor = F.interpolate(
                    masks_tensor,
                    size=(self.input_height, self.input_width),
                    mode='nearest'
                )
                masks_tensor = masks_tensor.squeeze(1)  # (N, H, W)
            else:
                masks_tensor = torch.from_numpy(masks_np).float()
            
            # Apply binary threshold for cleaner masks
            masks_tensor = (masks_tensor > 0.5).float()
            
        except Exception as e:
            # If masks not found, return None
            masks_tensor = None
        
        # Load relations
        try:
            global RELATIONS_CACHE, RELATIONS_CACHE_STATS
            if rel_path not in RELATIONS_CACHE:
                with open(rel_path, 'r') as f:
                    RELATIONS_CACHE[rel_path] = json.load(f)
                RELATIONS_CACHE_STATS['misses'] += 1
            else:
                RELATIONS_CACHE_STATS['hits'] += 1
            relations = RELATIONS_CACHE[rel_path]
            
            # Filter out unsupported relations ('above' is not supported by RelationalDepthLoss)
            if relations is not None:
                relations = [rel for rel in relations if rel.get('relation', '').lower() != 'above']
        except Exception as e:
            # If relations not found, return None
            relations = None
        
        return masks_tensor, relations


def collate_fn_with_relations(batch):
    """
    Custom collate function for batching with variable-size masks
    
    Args:
        batch: list of samples from dataset
    
    Returns:
        batched_sample: dict with batched tensors
    """
    # Regular fields - can be stacked normally
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    sample_paths = [item['sample_path'] for item in batch]
    has_valid_depth = torch.stack([item['has_valid_depth'] for item in batch])
    
    batched_sample = {
        'image': images,
        'depth': depths,
        'sample_path': sample_paths,
        'has_valid_depth': has_valid_depth
    }
    
    # Relational annotations - need special handling
    if 'masks' in batch[0]:
        masks_list = []
        relations_list = []
        
        for item in batch:
            masks = item.get('masks')
            relations = item.get('relations')
            
            # Handle None cases
            if masks is None or relations is None:
                # Create empty placeholders
                masks_list.append(torch.zeros((0, item['image'].shape[1], item['image'].shape[2])))
                relations_list.append([])
            else:
                masks_list.append(masks)
                relations_list.append(relations)
        
        # Keep as list (simpler, works with variable number of objects per image)
        batched_sample['masks'] = masks_list
        batched_sample['relations'] = relations_list
    
    return batched_sample


def create_nyu_relational_dataloader(args, mode='train', train_sampler=None, use_ddp=False):
    """
    Create DataLoader for NYU-Depth-v2 with relational annotations

    Args:
        args: training arguments
        mode: 'train' or 'online_eval'
        train_sampler: optional DistributedSampler for DDP; when set, shuffle=False for train.
        use_ddp: if True and mode=='train', create DistributedSampler inside and use it.

    Returns:
        DataLoaderWrapper with .data (DataLoader) and .train_sampler (when DDP train).
    """
    is_train = (mode == 'train')
    
    if is_train:
        filenames_file = getattr(args, 'filenames_file', None)
        data_path = args.data_path
        gt_path = args.gt_path
        relations_path = getattr(args, 'relations_dir_train', None)
    else:
        filenames_file = getattr(args, 'filenames_file_eval', None)
        data_path = args.data_path_eval
        gt_path = args.gt_path_eval
        relations_path = getattr(args, 'relations_dir_eval', None)
    
    # Check if relational loss is enabled
    use_relational_loss = getattr(args, 'use_relational_loss', False)
    
    if use_relational_loss and relations_path is None:
        print("Warning: use_relational_loss=True but relations directory not specified!")
        print("         Relations will not be loaded.")
        use_relational_loss = False
    
    # WorDepth: NYU = file in mm, /1000 -> m. Converted sync = 16-bit raw/6553.5 -> m.
    # Prefer depth_scale from args (same as train.py); else fall back to dataset name.
    depth_scale = getattr(args, "depth_scale", 6553.5)
    depth_in_mm = (depth_scale == 1000.0)
    use_dense_depth = getattr(args, "use_dense_depth", False)
    if use_dense_depth:
        depth_in_mm = False  # dense in data/ is 16-bit raw/6553.5 = m

    # Create dataset
    dataset = NYURelationalDataset(
        filenames_file=filenames_file,
        data_path=data_path,
        gt_path=gt_path,
        relations_base_path=relations_path if use_relational_loss else None,
        is_train=is_train,
        input_height=args.input_height,
        input_width=args.input_width,
        max_depth=args.max_depth,
        do_random_rotate=getattr(args, 'do_random_rotate', False),
        degree=getattr(args, 'degree', 2.5),
        use_relational_loss=use_relational_loss,
        depth_in_mm=depth_in_mm,
        use_dense_depth=use_dense_depth,
        debug_relational=getattr(args, 'debug_relational', False),
        cache_images=getattr(args, 'cache_images', False),
        depth_scale=getattr(args, 'depth_scale', 6553.5),
    )
    
    # Create dataloader
    # Windows multiprocessing issue: use num_workers=0 on Windows to avoid yaml import errors
    num_workers = 0 if platform.system() == 'Windows' else getattr(args, 'num_threads', 4)

    if is_train and use_ddp and train_sampler is None:
        try:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(dataset)
        except Exception:
            train_sampler = None
    use_sampler = is_train and train_sampler is not None
    prefetch_factor = 4 if (num_workers > 0 and is_train) else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=not use_sampler and is_train,
        sampler=train_sampler if use_sampler else None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_with_relations if use_relational_loss else None
    )

    # Wrap in object with .data and .train_sampler for compatibility with NewDataLoader / DDP set_epoch
    class DataLoaderWrapper:
        def __init__(self, dataloader, train_sampler=None):
            self.data = dataloader
            self.train_sampler = train_sampler

    return DataLoaderWrapper(dataloader, train_sampler=train_sampler if is_train else None)
