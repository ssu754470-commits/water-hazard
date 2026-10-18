"""Regression checks for finite-value failure handling and full-state resume."""
import os
os.environ['WANDB_MODE'] = 'disabled'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'Pytorch-UNet-master'))
import torch
from torch import nn
from torch.utils.data import Dataset
import train

torch.set_num_threads(1)

class Samples(Dataset):
    mask_values = [0, 1]
    def __len__(self):
        return 10
    def __getitem__(self, index):
        return {'image': torch.full((3, 16, 16), index / 10),
                'mask': torch.full((16, 16), index % 2, dtype=torch.long)}

class TinyNet(nn.Module):
    n_channels = 3
    n_classes = 2
    def __init__(self, broken=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.broken = broken
    def forward(self, image):
        result = self.conv(image)
        return result * float('nan') if self.broken else result

root = Path(__file__).parent / 'repair_regression_checks'
train.dir_checkpoint = root / 'must_not_save'
try:
    train.train_model(TinyNet(broken=True), torch.device('cpu'), epochs=1,
                      dataset=Samples(), val_percent=0.2, amp=False)
except FloatingPointError:
    assert not train.dir_checkpoint.exists(), 'Invalid model was saved'
else:
    raise AssertionError('Non-finite predictions were silently accepted')
print('PASS: non-finite predictions stop training before checkpoint save')
train.dir_checkpoint = root / 'finite_resume'
model = TinyNet()
train.train_model(model, torch.device('cpu'), epochs=1, dataset=Samples(),
                  val_percent=0.2, amp=False)
state = torch.load(train.dir_checkpoint / 'training_state.pth', weights_only=True)
assert state['epoch'] == 1 and state['global_step'] == 8
assert all(torch.isfinite(v).all() for v in state['model'].values())
restored = TinyNet()
restored.load_state_dict(state['model'])
train.train_model(restored, torch.device('cpu'), epochs=2, start_epoch=1,
                  resume_state=state, dataset=Samples(), val_percent=0.2, amp=False)
state = torch.load(train.dir_checkpoint / 'training_state.pth', weights_only=True)
assert state['epoch'] == 2 and state['global_step'] == 16
assert all(torch.isfinite(v).all() for v in state['model'].values())
print('PASS: empty/foreground masks train with finite loss; checkpoint and optimizer resume succeed')
