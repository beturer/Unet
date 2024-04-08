import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    """
    Saves checkpoint to file
    :param state: Checkpoint dictionary
    :param filename: Name of checkpoint file
    :return: None
    """
    print('==> Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
    Loads model checkpoint
    :param checkpoint: Path to checkpoint file
    :param model: Model to load checkpoint into
    :return: None
    """
    print("==> Loading from checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir, 
                train_maskdir,
                val_dir,
                val_maskdir,
                batch_size,
                train_transform,
                val_transform,
                num_workers=4,
                pin_memory=True):
    """
    Creates data loaders for training and validation sets
    :param train_dir: Path to training images
    :param train_maskdir: Path to training masks
    :param val_dir: Path to validation images
    :param val_maskdir: Path to validation masks
    :param batch_size: Batch size for dataloaders
    :param train_transform: Transform for training data
    :param val_transform: Transform for validation data
    :param num_workers: Number of workers for dataloaders
    :param pin_memory: Whether to pin memory for dataloaders
    :return: Train and validation data loaders
    :rtype: tuple
    """
    train_ds = CarvanaDataset(image_dir=train_dir,
                              mask_dir=train_maskdir,
                              transform=train_transform)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    val_ds = CarvanaDataset(image_dir=val_dir,
                            mask_dir=val_maskdir,
                            transform=val_transform)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0      
    model.eval()
    
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).float().sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).pow(2).sum() + 1e-8) #计算dice score

    print("Accuracy on the validation set: ", num_correct/num_pixels)
    print("Dice score on the validation set: ", dice_score/len(loader))
    model.train()


def save_predictions_as_imgs(loader, model, folder="save_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, folder+str(idx)+".png")

        torchvision.utils.save_image(y.unsqeeze(1), folder+str(idx)+"_gt.png")
        model.train()
    


