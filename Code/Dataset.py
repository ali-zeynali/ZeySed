from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
import random
from addict import Dict

class PlantDiseaseDataset(Dataset):
    def __init__(self, path,select_p=1, augmentations=None, image_shape=(256, 256), channels="RGB"):
        self.__images_labels = []
        self.image_shape = image_shape
        self.channels = channels
        self.augmentations = augmentations
        if os.path.exists(path):
            self.labels = os.listdir(path)
            for label in self.labels:
                label_path = os.path.join(path, label)
                if os.path.isdir(label_path):
                    files = os.listdir(label_path)
                    for file in files:
                        if file.endswith("jpg") or file.endswith("png"):
                            if np.random.random() < select_p:
                                image_path = os.path.join(label_path, file)
                                self.__images_labels.append((image_path, label))
                        else:
                            pass
                else:
                    pass
                
        else:
            pass
        
    def _load(self, path, channels="RGB"):
        width, height = self.image_shape
        loader = A.Compose([
            A.Resize(width=width, height=height),
            ToTensorV2(),
        ])
        
        image_array = np.array(Image.open(path).convert(channels))
        return loader(image=image_array)["image"]
#         if self.max_size == None:
#             return loader(image=image_array)["image"]
#         else:
#             return loader(image=image_array)["image"][:self.max_size]
    
    def __len__(self):
        return len(self.__images_labels)
    
    def __getitem__(self, index):
        path, label = self.__images_labels[index]
        image = self._load(path)
        
        if self.augmentations is not None:
            image = image.permute(1, 2, 0).numpy()
            image = self.augmentations(image=image)["image"]
            
        label = self.labels.index(label)
        
        return Dict({
            "image": image,
            "label": label,
        })
