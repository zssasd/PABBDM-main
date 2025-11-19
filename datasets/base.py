from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import torch

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name

# class ImagePathDataset(Dataset):
#     def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
#         self.image_size = image_size  # 最终输出的尺寸 (256,256)
#         self.image_paths = image_paths
#         self._length = len(image_paths)
#         self.flip = flip
#         self.to_normal = to_normal  # 是否归一化到[-1, 1]
        
#         # 定义预处理流程
#         self.base_transform = transforms.Compose([
#             transforms.Resize(512),  # 先resize到512x512
#             transforms.RandomCrop(256),  # 再随机裁剪到256x256
#             transforms.RandomHorizontalFlip(p=0.5 if flip else 0),  # 随机水平翻转
#             transforms.ToTensor()  # 转为Tensor [0,1]
#         ])

#     def __len__(self):
#         if self.flip:
#             return self._length * 2
#         return self._length

#     def __getitem__(self, index):
#         p = 0.0
#         if index >= self._length:
#             index = index - self._length
#             p = 1.0

#         img_path = self.image_paths[index]
#         try:
#             image = Image.open(img_path)
#             if not image.mode == 'RGB':
#                 image = image.convert('RGB')
                
#             # 应用预处理流程
#             image = self.base_transform(image)
            
#             if self.to_normal:
#                 image = (image - 0.5) * 2.  # 归一化到[-1,1]
#                 image.clamp_(-1., 1.)
                
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             # 返回一个空白图像
#             image = torch.zeros((3, 256, 256)) if not self.to_normal else torch.zeros((3, 256, 256)) * 2 - 1

#         image_name = Path(img_path).stem
#         return image, image_name