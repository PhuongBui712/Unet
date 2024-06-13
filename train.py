from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split

from data import CarvanaDataset
from unet import Unet


if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    EPOCHS = 2
    DATA_PATH = './data'
    MODEL_SAVE_PATH = './model'

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    dataset = CarvanaDataset(DATA_PATH, 'train')

    torch.manual_seed(42)
    generator = torch.Generator()
    train_split, val_split = random_split(dataset, [0.8, 0.2], generator)

    train_dataloader = DataLoader(dataset=train_split,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_split,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = Unet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.BCEWithLogitsLoss()

    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss': ^10}")
    for epoch in tqdm(range(EPOCHS)):
        # train
        model.train()
        train_running_loss = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img, mask = tuple(t.to(device) for t in img_mask)

            y_pred = model(img)

            loss = loss_func(y_pred, mask)
            train_running_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        # evaluate
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(val_dataloader):
                img, mask = tuple(t.to(device) for t in img_mask)

                y_pred = model(img)
                loss = loss_func(y_pred, mask)
                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {val_loss:^10.6f}")