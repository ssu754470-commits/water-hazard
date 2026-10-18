import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent / 'Pytorch-UNet-master'))
import torch
from train_hd10k import HD10KDataset
from unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
root = Path(__file__).parent / 'Pytorch-UNet-master'
checkpoint = root / 'checkpoints_hd10k_fixed_20261014' / f'checkpoint_epoch{args.epoch}.pth'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
mask_values = state.pop('mask_values')
bad = [name for name, value in state.items() if not torch.isfinite(value).all()]
if bad:
    raise RuntimeError(f'Invalid model tensors: {bad[:5]}')
model = UNet(n_channels=3, n_classes=2, bilinear=False).eval()
model.load_state_dict(state)
dataset = HD10KDataset('C:/program1/HD10K_IROS2022', 0.5)
with torch.inference_mode():
    sample = dataset[0]
    prediction = model(sample['image'].unsqueeze(0))
    assert torch.isfinite(prediction).all(), 'Non-finite inference output'
    assert prediction.shape == (1, 2, 240, 320), prediction.shape
export = root / 'MODEL_VALID.pth'
temporary = export.with_suffix('.tmp')
shutil.copyfile(checkpoint, temporary)
temporary.replace(export)
(root / 'MODEL_VALID.info.txt').write_text(
    f'Valid HD10K liquid segmentation model\nEpoch: {args.epoch}/30\nSource: {checkpoint}\n'
    'All model tensors finite; real-image inference finite.\n', encoding='utf-8')
print(f'PASS: epoch {args.epoch}, {len(state)} finite tensors, real-image inference finite')
print(f'EXPORTED: {export}')
