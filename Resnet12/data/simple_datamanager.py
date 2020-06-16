import torch
from abc import abstractmethod
import os
from PIL import Image
import json

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataset:
    def __init__(self, data_file, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        #self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SimpleDataManager(DataManager):
    def __init__(self, dataset, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def get_data_loader(self): #parameters that would change on train/val set
        dataset = self.dataset#SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader