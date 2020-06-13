import os.path as osp
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import h5py

class OpenMIC_Data(Dataset):

    def __init__(self, setname, data_path):
        data = None
        lb = -1
        all_labels = []
        for i in range(len(data_path)):
            file = h5py.File(data_path[i] + '.hd5', 'r')
           # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
            labels = file["label"]
            signlabel = []
            for label in labels:
                if not label in signlabel:
                    signlabel.append(label)
                    lb = lb + 1
                all_labels.append(lb)

            if data is None:
                data = file["data"]
            else:
                data = np.concatenate((data, file["data"]), axis=0)

            #data.append(file["data"])
            #all_labels.append(all_labels)

        self.data = data
        self.label = all_labels
        self.setname = setname
        if setname == 'train':
            self.transform = transforms.Compose([
                #transforms.Resize(84),
                transforms.RandomRotation([-90, 90]),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                #transforms.Resize(84),
                transforms.ToTensor()
            ])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_arr, label = self.data[i], self.label[i]
        if self.setname == 'train':
            image = self.transform(Image.fromarray(image_arr))
        else:
            image = self.transform(image_arr)
        #image = self.transform(Image.fromarray(image_arr))

        return image, label