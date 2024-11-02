import os
import sys
sys.path.append("/data/ephemeral/home/code")
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, 
                        default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--device', 
                        default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num', type=int, default=1,
                        help='Dataset number to use (default: 1)')
    
    args = parser.parse_args()

    return args

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, model_dir, filename):
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    checkpoint_path = osp.join(model_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    return start_epoch, best_val_loss

def get_cosine_scheduler(optimizer, num_epochs, num_warmup_epochs=5):
    # Warm-up 기간 동안 선형적으로 learning rate 증가
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_epochs
    )
    
    # Cosine Annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - num_warmup_epochs,
        eta_min=1e-6
    )
    
    # Scheduler 순차 결합
    from torch.optim.lr_scheduler import SequentialLR
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_epochs]
    )

def do_training(data_dir, model_dir, device, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, resume=None, num=1):
    # Dataset 및 DataLoader 설정
    train_dataset = SceneTextDataset(
        root_dir=data_dir,
        split='train',
        num=num,
        color_jitter=True,
        normalize=True,
        map_scale=0.5
    )
    num_train_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_dataset = SceneTextDataset(
        root_dir=data_dir,
        split='val',
        num=num,
        color_jitter=False,
        normalize=True,
        map_scale=0.5
    )
    num_val_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True
    )

    # 모델, optimizer, scheduler 초기화
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    scheduler = get_cosine_scheduler(optimizer, max_epoch, num_warmup_epochs=5)
    # 초기값 설정
    start_epoch = 0
    best_val_loss = float('inf')

    # Resume 체크포인트가 있다면 로드
    if resume is not None and osp.exists(resume):
        print(f'Resuming from checkpoint: {resume}')
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume)
        print(f'Resumed from epoch {start_epoch} with best validation loss {best_val_loss}')
    
    # Training loop
    for epoch in range(start_epoch, max_epoch):
        # Training
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_train_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}/{}]'.format(epoch + 1, max_epoch))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        train_loss = epoch_loss / num_train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'cls_loss': 0, 'angle_loss': 0, 'iou_loss': 0}
        
        with torch.no_grad():
            with tqdm(total=num_val_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    pbar.set_description('[Validation]')
                    
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    val_loss += loss.item()
                    
                    for key in val_metrics.keys():
                        val_metrics[key] += extra_info[key]
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Val Cls loss': extra_info['cls_loss'], 
                        'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    })

        val_loss = val_loss / num_val_batches
        for key in val_metrics.keys():
            val_metrics[key] /= num_val_batches

        scheduler.step()

        # 결과 출력
        print('Epoch {}/{} | Train Loss: {:.4f} | Elapsed time: {}'.format(
            epoch + 1, max_epoch, train_loss, timedelta(seconds=time.time() - epoch_start)))
        print('Validation Metrics | Cls Loss: {:.4f} | Angle Loss: {:.4f} | IoU Loss: {:.4f}'.format(
            val_metrics['cls_loss'], val_metrics['angle_loss'], val_metrics['iou_loss']))

        # 모델 저장
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, 
                epoch + 1, best_val_loss,
                model_dir, 'checkpoint_epoch{}.pth'.format(epoch + 1)
            )
        
        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, best_val_loss,
                model_dir, 'best.pth'
            )
            print(f'Saved best model with validation loss: {val_loss:.4f}')


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)