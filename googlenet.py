# -*- coding: utf-8 -*-
"""GoogleNet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y2BfB4qwbkDRZSkpICQdqqndLWOeoobg
"""

import os
 import cv2
 import shutil
 import urllib.request
 import scipy.stats as stats
 from collections import OrderedDict
 from IPython.display import clear_output
 import numpy as np
 import matplotlib.pyplot as plt
 import torch
 import torch.nn as nn
 import torch.optim as optim
 from torch.utils.data import Dataset, DataLoader, random_split
 from torchvision import transforms, datasets
 from torchsummary import summary
 from PIL import Image

 device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/drive')

# Building the initial Convolutional Block
# this will be used in hte stem part of the network i.e. initial
class ConvBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride , padding):
    super(ConvBlock, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride , padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.activation = nn.ReLU()

  def forward( self, x):
    out = self.conv(x)

    out = self.bn(out)
    out = self.activation(out)
    return out

# class ConvBlock(nn.Module):

#   def __init__(self, in_channels, out_channels, kernel_size, stride , padding):
#     super(ConvBlock, self).__init__()

#     self.Block = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, stride , padding),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU()
#     )

#   def forward( self, x):
#     out = self.Block(x)
#     return out

"""## Building the INCEPTION BLOCK
### '#3x3 redunce' and "#5x5 reduce"

From paper - "#3 x 3 reduce" and "#5x5 reduce" stands for the number of 1x1 filters in the reduction layer used before the 3x3 and 5x5 convolutions.
One can see the number of 1x1 filters in the projection layer after the built-in max pooling in the "pool proj" column. \
the purpose of using 1x1 conv filters is to reduce the number of parameters and the number of operations
All these reduction/projection layers use rectified linear (ReLU) activation
"""

class Inceptions(nn.Module):

  def __init__( self, in_channels, num1x1, num3x3_reduce, num3x3 , num5x5_reduce, num5x5, pool_proj):
    super(Inceptions, self).__init__()

    self.block1 = nn.Sequential(
        ConvBlock(in_channels, num1x1 , kernel_size=1 , stride =1, padding = 0)
    )

    self.block2 = nn.Sequential(
        ConvBlock(in_channels, num3x3_reduce, kernel_size = 1, stride = 1, padding = 0),
        ConvBlock( num3x3_reduce, num3x3, kernel_size =3, stride = 1, padding =1)
    )

    self.block3 = nn.Sequential(
        ConvBlock(in_channels, num5x5_reduce, kernel_size =1, stride = 1 , padding = 0 ),
        ConvBlock(num5x5_reduce, num5x5, kernel_size= 5, stride = 1, padding = 2)
    )

    self.block4 = nn.Sequential(
        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
        ConvBlock( in_channels, pool_proj, kernel_size = 1, stride = 1, padding = 0)
    )

  def forward(self, x):
    out1 = self.block1(x)
    out2 = self.block2(x)
    out3 = self.block3(x)
    out4 = self.block4(x)

    out = torch.cat([out1, out2, out3, out4],1)
    return out

class Auxillary(nn.Module):

  def __init__(self, in_channels, num_classes):
    super(Auxillary, self).__init__()

    self.pool = nn.AdaptiveAvgPool2d((4,4))
    self.conv = nn.Conv2d(in_channels, 128 , kernel_size =1, stride = 1, padding = 0)
    self.activation = nn.ReLU()

    self.fc1 = nn.Linear(2048 , 1024)
    self.dropout = nn.Dropout(0.7)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    print("x shape is: ", x.shape)
    out = self.pool(x)
    out = self.conv(out)
    out = self.activation(out)
    print("out shape is: ", out.shape)
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    otu = self.activation(out)
    out = self.dropout(out)
    out = self.fc2(out)

    return out

class GoogleNet(nn.Module):
  def __init__(self, num_classes= 10):

    super(GoogleNet, self).__init__()

    self.conv1 = ConvBlock( 3, 64, kernel_size= 7 , stride = 2, padding = 3)
    self.pool1 = nn.MaxPool2d( 3, stride = 2, padding = 0, ceil_mode = True)
    self.conv2 = ConvBlock(64, 64, kernel_size = 1, stride = 1, padding=0 )
    self.conv3 = ConvBlock(64, 192, kernel_size = 3, stride = 1, padding = 1)
    self.pool3 = nn.MaxPool2d(3, stride = 2, padding = 0, ceil_mode=True)

    self.inception3A = Inceptions( in_channels = 192, num1x1 = 64, num3x3_reduce =96, num3x3= 128, num5x5_reduce = 16 ,num5x5 = 32 , pool_proj = 32)
    self.inception3B = Inceptions( in_channels = 256, num1x1 = 128, num3x3_reduce =128, num3x3= 192, num5x5_reduce = 32 ,num5x5 = 96 , pool_proj = 64)

    self.pool4 = nn.MaxPool2d(3, stride= 2, padding = 0 , ceil_mode = True)

    self.inception4A = Inceptions( in_channels = 480, num1x1 = 192, num3x3_reduce =96, num3x3= 208, num5x5_reduce = 16 ,num5x5 = 48 , pool_proj = 64)
    self.inception4B = Inceptions( in_channels = 512, num1x1 = 160, num3x3_reduce =112, num3x3= 224, num5x5_reduce = 24 ,num5x5 = 64 , pool_proj = 64)
    self.inception4C = Inceptions( in_channels = 512, num1x1 = 128, num3x3_reduce =128, num3x3= 256, num5x5_reduce = 24 ,num5x5 = 64 , pool_proj = 64)
    self.inception4D = Inceptions( in_channels = 512, num1x1 = 112, num3x3_reduce =144, num3x3= 288, num5x5_reduce = 32 ,num5x5 = 64 , pool_proj = 64)
    self.inception4E = Inceptions( in_channels = 528, num1x1 = 256, num3x3_reduce =160, num3x3= 320, num5x5_reduce = 32 ,num5x5 = 128 , pool_proj = 128)

    self.pool5 = nn.MaxPool2d(3, stride = 2, padding = 0, ceil_mode = True)

    self.inception5A = Inceptions( in_channels = 832, num1x1 = 256, num3x3_reduce =160, num3x3= 320, num5x5_reduce = 32 ,num5x5 = 128 , pool_proj = 128)
    self.inception5B = Inceptions( in_channels = 832, num1x1 = 384, num3x3_reduce =192, num3x3= 384, num5x5_reduce = 48 ,num5x5 = 128 , pool_proj = 128)

    self.pool6 = nn.AdaptiveAvgPool2d( (1,1))

    self.dropout = nn.Dropout(0.4)
    self.fc = nn.Linear( 1024, num_classes)

    self.aux4A = Auxillary(512, num_classes)
    self.aux4D = Auxillary(528, num_classes)

  def forward(self, x):
    out = self.conv1(x)
    out = self.pool1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.pool3(out)

    out = self.inception3A(out)
    out = self.inception3B(out)
    out = self.pool4(out)

    out = self.inception4A(out)
    aux1 = self.aux4A(out)

    out = self.inception4B(out)
    out = self.inception4C(out)

    out = self.inception4D(out)
    aux2 = self.aux4D(out)

    out = self.inception4E(out)
    out = self.pool5(out)

    out = self.inception5A(out)
    out = self.inception5B(out)
    out = self.pool6(out)

    out = torch.flatten(out, 1)

    out = self.dropout(out)
    out = self.fc(out)

    return out, aux1, aux2

def train_model(model, train_loader, val_loader, criterion, optimizer ):
  EPOCHS = 15
  train_samples_num = 45000
  val_samples_num = 5000
  train_epoch_loss_history, val_epoch_loss_history = [], []

  for epoch in range(EPOCHS):

    train_running_loss = 0
    correct_train = 0

    # model.train().cuda()
    model.train()

    for inputs, labels in train_loader:
      inputs , labels = inputs.to(device), labels.to(device)

      """ for every mini batch during the training phase we typically want to explicitly set the gradients to zero before starting to do backpropagation """
      optimizer.grad()

      # start the forward pass
      prediction0, aux_pred_1, aux_pred_2 = model(inputs)

      # Compute the loss
      real_loss = criterion(prediction0, labels)
      aux_loss1 = criterion(aux_pred_1, labels)
      aux_loss2 = criterion(aux_pred_2, labels)

      loss = real_loss + aux_loss1 + aux_loss2

      # do the backpropagation and update weights with .step() backward pass
      loss.backward()
      optimizer.step()

      # Update the running corrects
      _, predicted = torch.max(prediction0.data, 1)
      correct_train += (predicted == labels).float().sum().item()

      ''' Compare mathc loss
      Multiply each average batch loss with batch length
      The batch length is inputs.size(0) which gives the number total images in each batch.
      Essentially I am un averaging the previously calculated Loss'''
      train_running_loss += loss.data.item()*inputs.size(0)

    train_epoch_loss = train_running_loss/train_samples_num
    train_epoch_loss_history.append(train_epoch_loss)
    train_epoch_acc = correct_train/train_samples_num

    val_loss = 0
    correct_val = 0

    # model.eval().cuda()
    model.eval()

    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs, labels  = inputs.to(device), labels.to(device)

        prediction0, aux_pred_1, aux_pred_2 = model(inputs)

        real_loss = criterion(prediction0, labels)
        aux_loss1 = criterion(aux_pred_1, labels)
        aux_loss2 = criterion(aux_pred_2, labels)

        loss = real_loss + aux_loss1 + aux_loss2

        _, predicted = torch.max(prediction0.data, 1)

        correct_val += (predicted == labels).float().sum().item()

        val_loss += loss.data.item()*inputs.size(0)

      val_epoch_loss = val_loss/val_samples_num
      val_epoch_loss_history.append(val_epoch_loss)
      val_epoch_acc = correct_val/val_samples_num

    info= f"For Epoch {epoch+1}/{EPOCHS}: train-loss = {train_epoch_loss:0.5f} | train-acc = {train_acc:0.5f} | val-loss = {val_epoch_loss:0.5f} | val-acc = {val_epoch_acc:0.5f}"

    print(info)

    torch.save( model.state.dict(), "/content/drive/MyDrive/GoogleNet" )

  torch.save(model.state_dict(),"/content/drive/MyDrive/GoogleNet")

  return train_epoch_loss_history, val_epoch_loss_history

model = GoogleNet()

model.to(device)
summary(model, (3, 96, 96))

def cifar_dataloader():

  transform = transforms.Compose([
      transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])
  ])

  # iNPUT DATA IN Google Drive

  train_dataset = datasets.CIFAR10('/content/drive/MyDrive/input_dataset', train = True, download = True, transform = transform  )
  test_dataset = datasets.CIFAR10( '/content/drive/MyDrive/input_dataset', train = False, download = True, transform = transform)

  train_dataset , val_dataset = random_split(train_dataset, (45000, 5000))

  print( f"Image shape of random sample image: {train_dataset[0][0].numpy().shape}", end= '\r\n')

  print(f"Training Set: {len(train_dataset)} images")
  print(f"Validation Set: {len(val_dataset)} images")
  print(f"Test Set: {len(test_dataset)}")

  BATCH_SIZE = 128

  #generate DataLoader

  train_loader = DataLoader( train_dataset, batch_size = BATCH_SIZE, shuffle = True)
  val_loader = DataLoader( val_dataset, batch_size = BATCH_SIZE, shuffle = True)
  test_loader = DataLoader( test_dataset, batch_size = BATCH_SIZE, shuffle = True)

  return train_loader , val_loader, test_loader

train_loader , val_loader, test_loader = cifar_dataloader()

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

train_epoch_loss_history, val_epoch_loss_history = train_model(model, train_loader, val_loader, criterion, optimizer)

model = GoogleNet()
model.load_state_dict(torch.load("/content/drive/MyDrive/GoogleNet"))

run_test_smaples = 10000
correct = 0
model.eval()
with torch.no_grad():
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    prediction0, aux_pred_1, aux_pred_2 = model(inputs)
    _, predicted = torch.max(prediction0.data, 1)
    correct += (predicted == labels).float().sum().item()

  test_acc = correct/run_test_smaples
  print(f"Test Accuracy: {test_acc}")
