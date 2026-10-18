"""Train U-Net on the HD10K IROS2022 liquid segmentation subset."""
import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault('WANDB_MODE', 'disabled')

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.data_loading import BasicDataset
from unet import UNet
import train


class HD10KDataset(Dataset):
    mask_values = [0, 1]

    def __init__(self, root, scale):
        self.scale = scale
        root = Path(root)
        if (root / 'IROS2022_Dataset').is_dir():
            root = root / 'IROS2022_Dataset'
        root = root / 'train' / 'liquid_dirts'
        image_root = root / 'images'
        mask_root = root / 'liquid_dirts_masks'
        self.pairs = []
        missing = []
        for image in sorted(image_root.rglob('*.jpg')):
            if image.name.startswith('.'):
                continue
            mask = (mask_root / image.relative_to(image_root)).with_suffix('.png')
            if not mask.is_file():
                missing.append(str(mask))
            else:
                self.pairs.append((image, mask))
        if missing:
            raise RuntimeError(f'Missing {len(missing)} masks; first: {missing[0]}')
        if not self.pairs:
            raise RuntimeError(f'No paired images found under {root}')
        logging.info('HD10K liquid segmentation: %s image/mask pairs', len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        with Image.open(mask_path) as source:
            # Normalize 1-bit, palette and RGB masks to binary foreground.
            mask_rgb = np.asarray(source.convert('RGB'))
            mask = Image.fromarray(np.any(mask_rgb != 0, axis=-1).astype(np.uint8))
        if image.size != mask.size:
            raise ValueError(f'Image/mask size mismatch: {image_path}')
        image = BasicDataset.preprocess(None, image, self.scale, is_mask=False)
        mask = BasicDataset.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        return {'image': torch.as_tensor(image.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='C:/program1/HD10K_IROS2022')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0.5)
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument('--load', type=Path)
    loading.add_argument('--resume', type=Path)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path(__file__).resolve().parent / 'checkpoints_hd10k_fixed_20261014')
    args = parser.parse_args()
    if not 0 <= args.start_epoch < args.epochs:
        parser.error('--start-epoch must be less than --epochs')
    if args.start_epoch and not (args.load or args.resume):
        parser.error('--start-epoch requires --load')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    logging.info('Using device %s; AdamW; full precision; finite-value checks enabled', device)
    dataset = HD10KDataset(args.data_root, args.scale)
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model = model.to(device=device, memory_format=torch.channels_last)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=True)
        args.start_epoch = resume_state['epoch']
        if args.start_epoch >= args.epochs:
            parser.error('Resume checkpoint has already reached the requested epochs')
        state = resume_state['model']
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Cannot resume checkpoint containing NaN or infinity')
        model.load_state_dict(state)
        logging.info('Resuming complete training state from %s', args.resume)
    if args.load:
        state = torch.load(args.load, map_location=device, weights_only=True)
        state.pop('mask_values', None)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Cannot load checkpoint containing NaN or infinity')
        model.load_state_dict(state)
        logging.info('Loaded model weights from %s', args.load)
    train.dir_checkpoint = args.checkpoint_dir
    train.train_model(model=model, device=device, epochs=args.epochs, batch_size=1,
                      img_scale=args.scale, amp=False, learning_rate=args.learning_rate,
                      dataset=dataset, num_workers=0, start_epoch=args.start_epoch, resume_state=resume_state)

