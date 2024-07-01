import wandb
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.unet import Unet
from src.data import BCSSDataset
from src.loss import DiceLoss
from src.metric import *
from src.utils import get_gpu_count, get_lr


load_dotenv()


if __name__ == '__main__':
    # setup wandb to track training progress
    batch = 16
    epochs = 10
    STEP_PER_LOG = 100

    wandb.login()

    wandb.init(
        project='BCSS-segmentation',
        name='Unet',
        config={
            'epoch': epochs,
            'batch_size': batch
        },
    )
    
    # initialize training stuff
    data_path = './data/bcss'
    NUM_CLASSES = 22
    train_dataset = BCSSDataset(path=data_path, split='train')
    val_dataset = BCSSDataset(path=data_path, split='val')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Unet(
        in_channels=3,
        output_classes=22,
        down_conv_kwargs={'kernel_size': 3, 'padding': 1},
        down_sample_kwargs={'kernel_size': 2, 'stride': 2},
        up_conv_kwargs={'kernel_size': 3, 'padding': 1},
        up_sample_kwargs={'kernel_size': 2, 'stride': 2}
    )
    if device == 'cuda:0' and get_gpu_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(get_gpu_count())))
        model.to(device)

    ce_loss = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss().to(device)

    max_lr = 1e-3
    weight_decay = 1e-4

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_dataloader))

    # training
    train_loss, val_loss, train_acc, val_acc, train_iou, val_iou, lrs = [], [], [], [], [], [], []

    for epoch in range(epochs):
        running_loss, iou_score, accuracy = 0, 0, 0
        batch_count, num_log = 0, 1
        last_train_data = None

        # Training loop
        model.train()
        train_loop = tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}/{epochs}', leave=True)
        for i, data in enumerate(train_loop):
            X, y = (_.to(device) for _ in data)

            # Forward
            y_pred = model(X)

            # compute loss
            loss = dice_loss(y_pred, y) + ce_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update metrics
            running_loss += loss.item()
            iou_score += mean_iou(y_pred, y, num_classes=NUM_CLASSES)
            accuracy += pixel_accuracy(y_pred, y)

            # update progress bar
            logging_dict = {
                'loss': running_loss / (i + 1),
                'mean IoU': iou_score / (i + 1),
                'accuracy': accuracy / (i + 1)
            }
            train_loop.set_postfix(logging_dict)

            # step learning rate scheduler
            lrs.append(get_lr(optimizer))
            scheduler.step()

            # update wandb
            batch_count += 1
            if batch_count // STEP_PER_LOG == num_log or i == len(train_dataloader) - 1:
                logging_dict['epoch'] = batch_count / len(train_dataloader)
                wandb.log({f'train/{k}': v for k, v in logging_dict.items()}, step=batch_count)
                
                num_log += 1

        # Validation loop
        model.eval()
        val_running_loss, val_iou_score, val_accuracy = 0, 0, 0
        val_loop = tqdm(val_dataloader, desc='Validation', leave=True)
        with torch.no_grad():
            for i, data in enumerate(val_loop):
                X, y = (_.to(device) for _ in data)

                # Forward
                y_pred = model(X)

                # compute loss
                loss = dice_loss(y_pred, y) + ce_loss(y_pred, y)

                # update metrics
                val_running_loss += loss.item()
                val_iou_score += mean_iou(y_pred, y, num_classes=NUM_CLASSES)
                val_accuracy += pixel_accuracy(y_pred, y)

                # update progress bar
                logging_dict = {
                    'loss': val_running_loss / (i + 1),
                    'mean IoU': val_iou_score / (i + 1),
                    'accuracy': val_accuracy / (i + 1)
                }
                val_loop.set_postfix(logging_dict)

        # Log the evaluation data together with train data
        wandb.log({
            'train/epoch': epoch + 1,
            'eval/loss': val_running_loss / len(val_dataloader),
            'eval/mean IoU': val_iou_score / len(val_dataloader),
            'eval/accuracy': val_accuracy / len(val_dataloader)
        })
