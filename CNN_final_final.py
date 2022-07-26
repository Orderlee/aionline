from asyncio import open_connection
from typing_extensions import Self
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import functional as FM
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import os
device ='cpu'
torch.__version__
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import f1_score, zero_one_loss
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_imgs = ImageFolder(os.path.join(os.getcwd(), 'train_img'),
                         transform=transforms.Compose([transforms.ToTensor()]))
val_imgs = ImageFolder(os.path.join(os.getcwd(), 'TEST_input'),
                        transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_imgs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_imgs, batch_size=1, shuffle=False)

loss_function = nn.CrossEntropyLoss()

class CNN(LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256*2*2, 1000),
            nn.Linear(1000, 5))
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

def train_net(n_epoch): # Training our network
    losses = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            losses.append(loss)
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Training Finished')

# Train model
# epoch_number = 3
# train_net(epoch_number)

# Save model
# PATH = './cnn_model_{}epochs.pth'.format(epoch_number)
# torch.save(cnn.state_dict(), PATH)

# Loading the trained network
PATH = './cnn_model_{}epochs.pth'.format(100)
cnn.load_state_dict(torch.load(PATH))

all_labels = []
all_predicted = []
correct = 0
total = 0
all_samples_paths = []
with torch.no_grad():
    for i,data in enumerate(val_loader):
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        sample_fname, what = val_loader.dataset.samples[i]
        all_samples_paths.append(sample_fname)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.tolist())
        all_predicted.extend(predicted.tolist())

all_photos = []
for asp in all_samples_paths:
    temp_sample = asp.split('\\')
    all_photos.append(int(temp_sample[-1].replace('.png','')))

print("len(all_photos)",len(all_photos))
print("len(all_predicted)",len(all_predicted))

# print(all_labels[:500],'\n')
# print(all_photos[:500],'\n')
# print(all_predicted[:500])

final_predicted = []
for z in range(7820):
    for img_i,img_num in enumerate(all_photos):
        if z == img_num:
            final_predicted.append(all_predicted[img_i])
# print(final_predicted)
print(len(final_predicted))

zero_cnt = 0
one_cnt = 0
two_cnt = 0
three_cnt = 0
four_cnt = 0
for ppp in all_predicted:
    if ppp == 0:
        zero_cnt +=1
    elif ppp == 1:
        one_cnt += 1
    elif ppp == 2:
        two_cnt += 1
    elif ppp == 3:
        three_cnt += 1
    elif ppp == 4:
        four_cnt += 1
print(zero_cnt, one_cnt, two_cnt, three_cnt, four_cnt)


print("F1 score:",f1_score(all_labels,all_predicted,average=None))
print("Macro F1 score:",f1_score(all_labels,all_predicted,average='macro'))
print("correct:",correct," /  total:",total)
print('Accuracy of the model on the %d test images: %f %%' % (len(all_predicted), 100 * correct / total))

import pandas as pd
test_file = pd.read_csv("sample_submission.csv")
id_header = test_file["id"].to_list()

class_predicted = []
for ap in final_predicted:
    if ap == 0:
        class_predicted.append('in')
    elif ap == 1:
        class_predicted.append('noise')
    elif ap == 2:
        class_predicted.append('normal')
    elif ap == 3:
        class_predicted.append('other')
    elif ap == 4:
        class_predicted.append('out')

in_cnt = 0
noise_cnt = 0
normal_cnt = 0
other_cnt = 0
out_cnt = 0
for ppp2 in class_predicted:
    if ppp2 == 'in':
        in_cnt +=1
    elif ppp2 == 'noise':
        noise_cnt += 1
    elif ppp2 == 'normal':
        normal_cnt += 1
    elif ppp2 == 'other':
        other_cnt += 1
    elif ppp2 == 'out':
        out_cnt += 1
print(in_cnt, noise_cnt, normal_cnt, other_cnt, out_cnt)


import csv
r = zip(id_header,class_predicted)
header = ['id','leaktype']
with open('final3.csv', 'w+', newline ='',encoding='UTF-8') as file:
    w = csv.writer(file)
    dw = csv.DictWriter(file,delimiter=',',fieldnames=header)
    dw.writeheader()
    for row in r:
        w.writerow(row)