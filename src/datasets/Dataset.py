import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2

class WBC_Dataset(data.Dataset):
    def __init__(self,image_paths,labels,resize=None, augment=False, dataset="AML"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_paths=image_paths
        self.labels=labels
        self.resize=resize #number of pixels in one dimension (square image)
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            ])
        if dataset =="AML":
            self.norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.82069695, 0.7281261, 0.836143],std=[0.16157213, 0.2490039, 0.09052657])])
        elif dataset =="PBC":
            self.norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.8746204, 0.7487587, 0.7203138],std=[0.15061052, 0.17617777, 0.07467376])])
        elif dataset =="MLL":
            self.norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.74053776, 0.6514114, 0.7785342],std=[0.18301032, 0.24672535, 0.16100405])])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        if self.resize is not None:
            image=image.resize((self.resize,self.resize))
        image=np.array(image)[:,:,0:3]
        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image)
        label=torch.zeros(13)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label
    
class WBC_Seg_Dataset(data.Dataset):
    def __init__(self,image_paths,seg_paths,resize=None, augment=False):
        self.resize=resize
        self.image_paths=image_paths
        self.seg_paths=seg_paths
        self.resize=resize #number of pixels in one dimension (square image)
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            ])
        self.norm = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.82069695, 0.7281261, 0.836143],std=[0.16157213, 0.2490039, 0.09052657])])
       
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        seg = Image.open(self.seg_paths[idx])
        if self.resize is not None:
            image=image.resize((self.resize,self.resize))
            seg=seg.resize((self.resize,self.resize))
        image=np.array(image)
        seg=np.round(np.array(seg)/255)
        seg=seg.astype(int)
        return image, seg
    
class MIL_Dataset(data.Dataset):
    def __init__(self,features,labels):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features=features
        self.labels=labels
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        feat=self.features[idx]
        label=torch.zeros(5)
        label[self.labels[idx]-1]=1
        return feat,label
    

class AML_Dataset(data.Dataset):
    def __init__(self,image_paths,labels,resize=None, augment=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_paths=image_paths
        self.labels=labels
        self.resize=resize #number of pixels in one dimension (square image)
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        if self.resize is not None:
            image=image.resize((self.resize,self.resize))
        if self.augment:
            image = self.transforms(image)
            
        image=np.array(image)[:,:,0:3]
        label=torch.zeros(13) #one hot encoded image
        label[self.labels[idx]]=1

        return image, label


class PBC_Dataset(data.Dataset):
    def __init__(self,image_paths,labels,resize=None, augment=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_paths=image_paths
        self.labels=labels
        self.resize=resize #number of pixels in one dimension (square image)
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        if self.resize is not None:
            image=image.resize((self.resize,self.resize))
        if self.augment:
            image = self.transforms(image)
            
        image=np.array(image)
        label=torch.zeros(13) #one hot encoded image
        label[self.labels[idx]]=1

        return image, label
    
class MLL_Dataset(data.Dataset):
    def __init__(self,image_paths,labels,resize=None, augment=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_paths=image_paths
        self.labels=labels
        self.resize=resize #number of pixels in one dimension (square image)
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        if self.resize is not None:
            image=image.resize((self.resize,self.resize))
        if self.augment:
            image = self.transforms(image)
            
        image=np.array(image)
        label=torch.zeros(13) #one hot encoded image
        label[self.labels[idx]]=1

        return image, label