'''
train : 12000
test : 4000
valid : 4000
6:2:2 = 12000 : 4000 : 4000
12000, 16000, 20000
'''
# https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
import random

# import torch.nn.parallel
# import torch.distributed as dist
# import torch.nn.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed

import argparse

from torchvision import transforms, datasets, models
from model import VGG19
from dataset import load_dataset as ld

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

class Train(object):
    def __init__(self, args):
        super().__init__()
        
        self.model_pth = args.model_pth
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr

        self.model = VGG19()
        self.model.to(device).train()
        self.crit = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=0.001)
        # self.lr_scheduler = lr_scheduler.StepLR(self.optim, step_size=7, gamma=0.1)

        self.dataloaders, self.dataset_sizes = ld(self.batch_size)

    def train_model(self):

        since = time.time()
    
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.n_epochs):
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)

            # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()  # 모델을 학습 모드로 설정
                else:
                    self.model.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                # 데이터를 반복
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if i%10 == 0:
                        print(f"It is {i}-th iteration / {epoch}-th epoch")
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 매개변수 경사도를 0으로 설정
                    self.optim.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, _ = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.crit(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # if phase == 'train':
                #     self.lr_scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 모델을 깊은 복사(deep copy)함
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 가장 나은 모델 가중치를 불러옴
        self.model.load_state_dict(best_model_wts)

            # 모델 저장
        torch.save(self.model.state_dict(), self.model_pth)
        # model.state_dict()를 저장하는 것은 모델을 저장하는 것이 아닌, 모델의 파라미터만 저장하는 것이다.

        return self.model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for model')
    
    parser.add_argument('--n_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('--model_pth', type=str, default='gender_classifier_params.pt', help='path where the model exists')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    args = parser.parse_args()
    model = Train(args)
    model = model.train_model()
    # model = transfer_learning()
    # model = Train(args)
    # model.train_model()
    