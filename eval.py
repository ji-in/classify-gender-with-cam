import torch
import torch.nn as nn
import time
import argparse
from dataset import load_dataset as ld
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Evaluation(object):
    def __init__(self, args):
        super().__init__()

        self.crit = nn.CrossEntropyLoss()
        self.model = model.VGG19()
        self.model.load_state_dict(torch.load(args.model_pth))
        self.model.eval()
        self.batch_size = args.batch_size

        self.dataloaders, self.dataset_sizes = ld(self.batch_size)

    def eval_model(self):
        start_time = time.time()
        criterion = nn.CrossEntropyLoss()

        class_names = ['0', '1']
        race_list = {'0':  '남자', '1': '여자'}

        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs, _ = self.model(inputs)
                _, preds = torch.max(outputs, 1) # 행 방향으로 최댓값 구해라
                loss = self.crit(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print(preds)
                # print(labels.data)
                # 한 배치의 첫 번째 이미지에 대하여 결과 시각화
                print(f'[예측 결과: {race_list[class_names[preds[0]]]}] (실제 정답: {race_list[class_names[labels.data[0]]]})')
                # print(running_loss, running_corrects)
        
            epoch_loss = running_loss / self.dataset_sizes['test']
            epoch_acc = running_corrects.double() / self.dataset_sizes['test']
        
            print('[Test Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc,
                                                                                time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for model')
    parser.add_argument('--model_pth', type=str, default='gender_classifier_params.pt',
                        help='path where the model exists')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    
    args = parser.parse_args()

    evaluation = Evaluation(args)
    evaluation.eval_model()

    # [Test Phase] Loss: 0.1190 Acc: 0.9555% Time: 255.6142s