import torch
import torch.nn as nn
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv = nn.Sequential(
            #3 224 224
            nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 112 112
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #128 56 56
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 28 28
            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #512 14 14
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        #512 7 7
        
        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        
        self.classifier = nn.Linear(512, 2)
        # 1 2
        """
        self.fc1 = nn.Linear(512*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        """

    def forward(self, x):
        features = self.conv(x)
        # print(features.type) -> tensor
        x = self.avg_pool(features).view(features.size(0), -1)
        # print(x.type) -> tensor
        x = self.classifier(x)
        # print(x.type) -> tensor
        # print(x.shape, features.shape) -> torch.Size([1, 2]) torch.Size([1, 512, 7, 7])
        return x, features

if __name__ == '__main__':
    model = VGG19()
    
    img_path = 'yoona.jpg'
    img = read_image(img_path)
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = input_tensor.view(-1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2])
    mod = model(input_tensor)
