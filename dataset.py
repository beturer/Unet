import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    """
    构建一个用于读取Carvana数据集的PyTorch Dataset。
    Args:
        image_dir (string): 图片文件的目录。
        mask_dir (string): 标注文件的目录。
        transform (callable, optional): 一个可选的PyTorch转换，将用于转换图像和标注。
    """
    def __init__(self, image_dir, mask_dir, trandform=None) -> None:
        super(CarvanaDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = trandform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "._mask.gif"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.flaot32)
        mask[mask == 255.0] = 1.0
         
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
