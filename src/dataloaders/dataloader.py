import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import random
import cv2


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))

            if getattr(args, 'use_ddp', False) and torch.distributed.is_initialized():
                self.train_sampler = DistributedSampler(
                    self.training_samples, shuffle=True
                )
            else:
                self.train_sampler = None

            self.data = DataLoader(
                self.training_samples,
                args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_threads,
                pin_memory=True,
                sampler=self.train_sampler,
                persistent_workers=True if args.num_threads > 0 else False,
                prefetch_factor=4 if args.num_threads > 0 else None,
            )

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))

            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        
        # Image cache
        self.image_cache = {}
        self.cache_images = getattr(args, 'cache_images', False)
        
        if self.cache_images and mode == 'train':
            from tqdm import tqdm
            print(f"Caching {len(self.filenames)} images to memory...")
            for idx in tqdm(range(len(self.filenames)), desc="Loading images"):
                sample_path = self.filenames[idx]
                if self.args.dataset == 'kitti':
                    rgb_file = sample_path.split()[0]
                    depth_file = sample_path.split()[1]
                else:
                    rgb_file = sample_path.split()[0][1:]
                    depth_file = sample_path.split()[1][1:]
                
                image_path = os.path.join(self.args.data_path, rgb_file)
                depth_path = os.path.join(self.args.gt_path, depth_file)
                
                image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
                depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                self.image_cache[image_path] = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                self.image_cache[depth_path] = depth_cv
            
            print(f"Cached {len(self.image_cache)//2} image pairs (~{len(self.image_cache)*640*480*2/1024/1024/1024:.1f} GB)")

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]

                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0][1:]
                depth_file = sample_path.split()[1][1:]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)

            # Use cached images if available
            if self.cache_images and image_path in self.image_cache:
                image = Image.fromarray(self.image_cache[image_path])
                depth_gt = Image.fromarray(self.image_cache[depth_path])
            else:
                # Use cv2 for faster loading
                image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_gt = Image.fromarray(depth_cv)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # NYU: align with NYURelationalDataset when use_relational_loss=False — same load, no depth-only
            # masking, no eigen crop. RGB and depth stay spatially matched; resize (not random_crop) below.
            # Previously: valid_mask[45:472,43:608] zeroed depth only while image stayed 480x640, which
            # caused RGB–depth mismatch and border artifacts in depth prediction.

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / getattr(self.args, 'depth_scale', 6553.5)
            else:
                depth_gt = depth_gt / 256.0

            # NYU: match nyu_relational_dataloader — resize to input (bilinear / nearest), no random_crop;
            # no train_preprocess (no random flip / gamma), only rotate above like relational.
            if self.args.dataset == 'nyu':
                if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                    image = cv2.resize(
                        image,
                        (self.args.input_width, self.args.input_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    depth_gt = cv2.resize(
                        depth_gt.squeeze(axis=2),
                        (self.args.input_width, self.args.input_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                # relational NYURelationalDataset does not apply train_preprocess
            else:
                if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                    image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
                image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, "sample_path": sample_path}

        else:
            # online_eval or test mode - use original resolution (no resize/crop)
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
                gt_path = self.args.gt_path_eval
            else:
                data_path = self.args.data_path
                gt_path = self.args.gt_path

            rgb_file = sample_path.split()[0]
            depth_file = sample_path.split()[1]
            # Remove leading slash for NYU dataset paths
            if self.args.dataset == 'nyu':
                rgb_file = rgb_file.lstrip('/')
                depth_file = depth_file.lstrip('/')
            
            image_path = os.path.join(data_path, rgb_file)
            depth_path = os.path.join(gt_path, depth_file)
            
            # Load image (cv2 to avoid PIL lazy-load AssertionError in DataLoader workers)
            image_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_cv is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            has_valid_depth = False
            depth_gt = None
            
            if self.mode == 'online_eval':
                depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_cv is not None:
                    has_valid_depth = True
                    depth_gt = depth_cv.astype(np.float32)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / getattr(self.args, 'depth_scale', 6553.5)
                    else:
                        depth_gt = depth_gt / 256.0
                    depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            # Resize to train resolution for consistent eval (256x320)
            if self.mode == 'online_eval' and (image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width):
                image = cv2.resize(image, (self.args.input_width, self.args.input_height), interpolation=cv2.INTER_LINEAR)
                if has_valid_depth:
                    depth_gt = cv2.resize(depth_gt, (self.args.input_width, self.args.input_height), interpolation=cv2.INTER_NEAREST)
                    if depth_gt.ndim == 2:
                        depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, "sample_path": sample_path}
            else:
                sample = {'image': image, 'focal': focal, "sample_path": sample_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        sample_path = sample['sample_path']

        if self.mode == 'test':
            return {'image': image, 'focal': focal, "sample_path": sample_path}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, "sample_path": sample_path}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, "sample_path": sample_path}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
