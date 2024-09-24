import os
import json
import random
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def ImageLabel_ClassID(data_path, json_path):
    with open(json_path, 'r') as f:
        #-------------------------------- process the json file -------------------------------- #
        cat2name_mapping = json.load(f) # {'21': 'fire lily', '3': 'canterbury bells', '45': 'bolero deep blue'}
        cat_list = [cat for cat in cat2name_mapping] # category old idx, ['21', '3', '45']
        class_names = [cat2name_mapping[cat] for cat in cat2name_mapping] # category names, ['fire lily', 'canterbury bells', 'bolero deep blue']

        #---------------------------------- json new mapping ---------------------------------- #
        N = list(range(len(class_names))) # category new idx (0, 1, ..., N-1)
        cat2N_mapping = dict(zip(cat_list, N))     # {'21': 0, '3': 1, '45': 2}
        name2N_mapping = dict(zip(class_names, N)) # {'fire lily': 0, 'canterbury bells': 1, 'bolero deep blue': 2}
        
        path_label = []
        for dirname, _, filenames in os.walk(data_path):
            # dirname: /mnt/evo1/xiangyu/data/flowers-dataset/train_dir/1
            # filenames: ['image_06744.jpg', 'image_06768.jpg', ..., 'image_06755.jpg']
            for filename in filenames:
                if dirname.split('/')[-1] in cat_list:
                    path = os.path.join(dirname, filename)
                    label = dirname.split('/')[-1] # category name: "21" "21", ... , "21", "3", "3", ..., "3", "45", "45", ... , "45"
                    path_label += [(path, cat2N_mapping[label])]
    
    return path_label, name2N_mapping # path_label: [xxx/yyy/.../zzz.jpg, 0] 


class ImageDataset(Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class TrainLoader():
    def __init__(self, size, dataroot, batchSize, n_cpu, json_path=None):
        resize = transforms.Resize((size, size))
        random_resized_crop = transforms.RandomResizedCrop((size, size), scale=[0.9, 1.0], ratio=[0.90, 1.10])
        rand_crop_resize = transforms.Lambda(lambda x: random_resized_crop(x) if random.random() < 0.5 else resize(x))
        if 'cifar' in dataroot.split('/')[-2]:
            self.transform = [transforms.RandomCrop((size, size), padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        else:
            self.transform = [rand_crop_resize,
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]            
        self.path_label, self.classID = ImageLabel_ClassID(dataroot, json_path)
        print("The number of the train images: ", len(self.path_label))
        self.labels = [pl[1] for pl in self.path_label]
        self.image_dataset = ImageDataset(self.path_label, transform=self.transform)
        self.data_loader = DataLoader(self.image_dataset, batch_size=batchSize, shuffle=True, num_workers=n_cpu, pin_memory=True)


class ValidLoader():
    def __init__(self, size, dataroot, batchSize, n_cpu, json_path=None):
        self.transform = [transforms.Resize((size,size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.path_label, self.classID = ImageLabel_ClassID(dataroot, json_path)
        print("The number of the valid images: ", len(self.path_label))
        self.image_dataset = ImageDataset(self.path_label, transform=self.transform)
        self.data_loader = DataLoader(self.image_dataset, batch_size=batchSize, shuffle=True, num_workers=n_cpu, pin_memory=True)


class TestLoader():
    def __init__(self, size, dataroot, batchSize, n_cpu, json_path=None):
        self.transform = [transforms.Resize((size,size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.path_label, self.classID = ImageLabel_ClassID(dataroot, json_path)
        print("The number of the test images: ", len(self.path_label))
        self.labels = [pl[1] for pl in self.path_label]
        self.image_dataset = ImageDataset(self.path_label, transform=self.transform)
        self.data_loader = DataLoader(self.image_dataset, batch_size=batchSize, shuffle=False, num_workers=n_cpu, pin_memory=True)