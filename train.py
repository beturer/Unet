import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs

#hyperparameters

LEAANING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'train_images'
TRAIN_MASK_DIR = 'train_masks'
VAL_IMG_DIR = 'val_images'
VAL_MASK_DIR = 'val_masks'

def train(loader, model, optimizer, loss, scaler):
    """
    训练模型

    :param loader: 训练数据加载器
    :param model: 待训练模型
    :param optimizer: 优化器
    :param loss: 损失函数
    :param scaler: 缩放器
    """
    loop = tqdm(loader)
    
    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.float().unsqueeze(1).to(DEVICE)
        
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss(predictions, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())



def main():
    """
    主函数
    1. 获取数据加载器
    2. 定义模型、优化器、损失函数
    3. 训练模型
    """
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontallyFlip(p=0.5),
        A.VerticallyFlip(p=0.1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1,0), max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1,0), max_pixel_value=255.0),
        ToTensorV2()
    ])
    

    model = UNet(in_channels=3, out_channels=1)
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEAANING_RATE)
    loss = nn.BCEWithLogitsLoss()

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, 
                                           TRAIN_MASK_DIR, 
                                           VAL_IMG_DIR, 
                                           VAL_MASK_DIR, 
                                           BATCH_SIZE, 
                                           train_transform, 
                                           test_transform,
                                           NUM_WORKERS,
                                           PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load('best_model.pth'), model)

    
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss, scaler)
        check_accuracy(val_loader, model, DEVICE)


        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, DEVICE)

        save_predictions_as_imgs(val_loader, model)
if __name__ == '__main__':
    main()