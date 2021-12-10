# https://deep-learning-study.tistory.com/470
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class CelebA(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None):
        self.data_dir = data_dir # 이미지 데이터 경로
        self.labels_dir = labels_dir # 레이블 데이터 경로
        
        self.image_names = os.listdir(self.data_dir) # 이미지 데이터 내의 모든 파일(디렉토리) 리스트를 리턴한다. -> ['000019.jpg', '000002.jpg', ...]
        self.full_image_names = [os.path.join(self.data_dir, f) for f in self.image_names] # ['./data/train/image/000019.jpg', './data/train/image/000002.jpg', ...]
        self.labels_df = pd.read_csv(self.labels_dir, index_col='image_id') # csv 파일 읽어들이기
        self.labels_df = self.labels_df[['Male']] # 'Male'에 해당하는 열 추출하기
        self.labels = [self.labels_df.loc[image_name].values[0]-1 if self.labels_df.loc[image_name].values[0] == 1 else self.labels_df.loc[image_name].values[0]+2 for image_name in self.image_names] 
        # male : 0, female : 1
        self.transform = transform
    
    def __len__(self):
        return len(self.full_image_names)

    def __getitem__(self, idx):
        image = Image.open(self.full_image_names[idx])
        image = self.transform(image)
        return image, self.labels[idx]

def load_dataset(bs):
    
    data_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_datasets = CelebA(data_dir='./data/train/train_image', labels_dir='./data/train/train_label.csv', transform=data_transformer)
    valid_datasets = CelebA(data_dir='./data/valid/valid_image', labels_dir='./data/valid/valid_label.csv', transform=data_transformer)
    # test_datasets = CelebA(data_dir='./data/test/test_image', labels_dir='./data/test/test_label.csv', transform=data_transformer)
    test_datasets = CelebA(data_dir='./test_30', labels_dir='./test_30_label.csv', transform=data_transformer)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=bs, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=False, num_workers=4)

    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader
    }

    dataset_sizes = {
        'train': len(train_datasets),
        'valid': len(valid_datasets),
        'test': len(test_datasets)
    }

    return dataloaders, dataset_sizes
    
def imsave(input):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 출력
    plt.imshow(input)
    plt.savefig('test.png')

if __name__ == "__main__":
    data_transformer = transforms.Compose([transforms.ToTensor()])
    custom_dataset = CelebA(data_dir='./data/img_align_celeba', labels_dir='./data/list_attr_celeba.csv', transform=data_transformer)
    # print('length of custom dataset is ', len(custom_dataset))
    image, label = custom_dataset[2]
    imsave(image)
    print(label)