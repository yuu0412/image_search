import torch
import cv2
import os
import hydra
import albumentations
from albumentations.pytorch.transforms import ToTensorV2


class JPODataset(torch.utils.data.Dataset):
    def __init__(self, data, label=None, is_train=True):
        self.data = data
        self.is_train = is_train
        if self.is_train:
            self.label = label
            self.transforms = albumentations.Compose(
                [
                    albumentations.Resize(512, 512, always_apply=True),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.VerticalFlip(p=0.5),
                    albumentations.Rotate(limit=120, p=0.8),
                    albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
                    albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
                    albumentations.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.1, rotate_limit=0),
                    albumentations.Normalize(),
                    ToTensorV2(p=1.0)
                ])
        else:
            self.transforms =  albumentations.Compose(
                [
                    albumentations.Resize(512, 512, always_apply=True),
                    albumentations.Normalize(),
                    ToTensorV2(p=1.0)
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        if self.is_train:
            image_path = hydra.utils.get_original_cwd()+"/data/input/crop_train_apply_images/"+x["path"]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.imshow()ではBGRの順番で読み込まれるため、RGBに変換
            augmented = self.transforms(image=image)
            image = augmented['image']
            y = self.label.iloc[idx]
            return image, y
        else:
            image_path = x
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.imshow()ではBGRの順番で読み込まれるため、RGBに変換
            augmented = self.transforms(image=image)
            image = augmented['image']
            return image