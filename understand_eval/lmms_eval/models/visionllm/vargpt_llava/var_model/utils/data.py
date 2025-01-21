import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


import pandas as pd
import io
from PIL import Image
import torch
from torch.utils.data import Dataset

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_files, transform=None):
        """
        参数:
            parquet_files: str 或 list, parquet文件路径或路径列表
            transform: 图像转换函数
        """
        self.transform = transform
        
        # 处理单个文件或文件列表
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
            
        if not parquet_files:
            raise ValueError("没有提供parquet文件路径")
            
        # 读取所有parquet文件并合并
        dfs = []
        for file in parquet_files:
            if not osp.exists(file):
                raise FileNotFoundError(f"找不到文件: {file}")
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                raise Exception(f"读取文件 {file} 时出错: {str(e)}")
        
        if not dfs:
            raise ValueError("没有成功读取任何parquet文件")
            
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"成功加载数据集，共 {len(self.df)} 条记录")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 从字节数据读取图像
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = row['label']
        return image, label

# 修改build_dataset函数
def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # 构建数据集
    import glob
    train_files = glob.glob(osp.join(data_path, 'train-*.parquet'))
    val_files = glob.glob(osp.join(data_path, 'test-*.parquet'))
    
    # 添加文件检查
    if not train_files:
        raise ValueError(f"未找到训练数据文件，路径: {osp.join(data_path, 'train-*.parquet')}")
    if not val_files:
        raise ValueError(f"未找到验证数据文件，路径: {osp.join(data_path, 'val-*.parquet')}")
        
    print(f"找到训练文件: {len(train_files)} 个")
    print(f"找到验证文件: {len(val_files)} 个")
    print(f"训练文件示例: {train_files[:2]}")
    print(f"验证文件示例: {val_files[:2]}")
    
    train_set = ParquetImageDataset(train_files, transform=train_aug)
    val_set = ParquetImageDataset(val_files, transform=val_aug)
    num_classes = 1000  # 假设类别数量为1000
    
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set



def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)



def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
