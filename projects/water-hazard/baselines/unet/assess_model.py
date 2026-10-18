import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'Pytorch-UNet-master'))
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_hd10k import HD10KDataset
from unet import UNet

torch.set_num_threads(2)
root = Path(__file__).parent
model_path = root / 'Pytorch-UNet-master/MODEL_VALID.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load(model_path, map_location='cpu', weights_only=True)
state.pop('mask_values', None)
model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device).eval()
model.load_state_dict(state)
dataset = HD10KDataset('C:/program1/HD10K_IROS2022', 0.5)
_, validation = random_split(dataset, [3600, 400], generator=torch.Generator().manual_seed(0))
records = []
tp_total = fp_total = fn_total = 0
with torch.inference_mode():
    for i, batch in enumerate(DataLoader(validation, batch_size=1, num_workers=0)):
        output = model(batch['image'].to(device))
        if not torch.isfinite(output).all():
            raise RuntimeError('Non-finite output during evaluation')
        predicted = output.argmax(1).cpu().bool()
        target = batch['mask'].bool()
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        positive = bool(target.any())
        dice = (2*tp+1e-6)/(2*tp+fp+fn+1e-6)
        iou = (tp+1e-6)/(tp+fp+fn+1e-6)
        records.append(dict(index=i, positive=positive, dice=dice, iou=iou, predicted_pixels=int(predicted.sum())))
        tp_total += tp; fp_total += fp; fn_total += fn
        if (i+1) % 100 == 0:
            print('Evaluated',i+1,'/400',flush=True)
positive_records = [r for r in records if r['positive']]
empty_records = [r for r in records if not r['positive']]
metrics = dict(model=str(model_path), epoch=15, split='400 held-out training frames; random seed 0',
    samples=400, positive_samples=len(positive_records), empty_samples=len(empty_records),
    mean_dice=sum(r['dice'] for r in records)/400,
    positive_mean_dice=sum(r['dice'] for r in positive_records)/len(positive_records),
    positive_mean_iou=sum(r['iou'] for r in positive_records)/len(positive_records),
    micro_dice=2*tp_total/max(1,2*tp_total+fp_total+fn_total),
    foreground_precision=tp_total/max(1,tp_total+fp_total), foreground_recall=tp_total/max(1,tp_total+fn_total),
    empty_false_positive_samples=sum(r['predicted_pixels']>0 for r in empty_records))
(root / 'model-quality.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2),flush=True)
ordered = sorted(positive_records, key=lambda r:r['dice'])
chosen = [ordered[-1], ordered[len(ordered)//2], ordered[0], max(empty_records,key=lambda r:r['predicted_pixels'])]
fig, axes = plt.subplots(4,3,figsize=(12,12))
with torch.inference_mode():
    for row, record in enumerate(chosen):
        sample = validation[record['index']]
        predicted = model(sample['image'].unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        axes[row,0].imshow(sample['image'].permute(1,2,0).numpy())
        axes[row,1].imshow(sample['mask'].numpy(),vmin=0,vmax=1,cmap='gray')
        axes[row,2].imshow(predicted,vmin=0,vmax=1,cmap='gray')
        axes[row,0].set_title(['Best positive','Median positive','Worst positive','Empty: most false positives'][row])
        axes[row,1].set_title('Ground truth')
        axes[row,2].set_title(f"Prediction | Dice {record['dice']:.3f}")
        for ax in axes[row]: ax.axis('off')
fig.tight_layout()
fig.savefig(root / 'model-quality.png', dpi=130)
print('Saved model-quality.png',flush=True)
