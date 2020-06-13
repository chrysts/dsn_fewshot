import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
ROOT_PATH = './materials/'



class MiniImageNet(Dataset):

    def __init__(self, setname, img_path):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        IMG_PATH = img_path
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMG_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img  =Image.open(path).convert('RGB')
        img = img.resize((84, 84)).convert('RGB')
        image = self.transform(img)

        return image, label


