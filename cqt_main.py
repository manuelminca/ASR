import sys
import numpy as np
import random
import pandas as pd
import torch
from torch.utils import data
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import sigmoid
from torch.utils.data import DataLoader

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class Dataset_CSV_train(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, location, minsize = 128):
        'Initialization'
        self.locationX = location + '/data/'
        self.locationY = location + '/labels/'
        self.minsize = minsize
        #to avoid loading hidden files produced by the OS
        files = listdir_nohidden(self.locationX)
        self.list_IDs = list(filter(lambda f: f.endswith('.csv'), files))

    def __len__(self):
        'Denotes the total number of samples'
        return 160
        #return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        ID = self.list_IDs[index]

        # Load data and get label

        df_data = pd.read_csv(self.locationX + ID)
        df_labels = pd.read_csv(self.locationY + ID)
        size = df_data.shape[0]


        try:
            position = random.randrange(0, size - self.minsize) #The start of the slice
        except ValueError:
            position = 0

        X = np.empty((self.minsize, 5, df_data.shape[1])) #5520 because its the smallest song size
        Y = np.zeros((self.minsize,88))

        auxiliary = 0

        for j in range(position, position+self.minsize - 5):

            if j == position:
                for k in range(5):
                    #Creating the 5 windows image
                    X[auxiliary][k] = df_data.iloc[j + k]
                if j < (position + self.minsize - 2):
                    #Creating the labels, taking the label of the mid window
                    Y[auxiliary] = df_labels.iloc[j + 2]

                auxiliary+=1
            else:
                for k in range(5):
                    if k < 3:
                      #Copying the values of the previous image to avoid data repetition
                      X[auxiliary][k] = X[auxiliary - 1][k + 1]
                    else:
                        X[auxiliary][k] = df_data.iloc[j + k]
                if j < (position + self.minsize - 2):
                    #Creating the labels, taking the label of the mid window
                    Y[auxiliary] = df_labels.iloc[j + 2]

                auxiliary+=1

        #Normalise X using min-max normalisation

        #For filterbanks
        #max_value = 120.1003100275508
        #min_value = -330.7445963332333

        #For MFCC
        #max_value = 841.837813253034
        #min_value = -831.0670773337255

        #For CQT
        max_value = 41.775738
        min_value = 0

        x_norm = (X  - min_value) / (max_value - min_value)
        x_norm = np.reshape(x_norm,(self.minsize,1, 5, df_data.shape[1]))

        #Code to activate also contiguous musical notes
        #for i,row in enumerate(Y):
        #    yes = np.where(row==1)[0]
        #    for element in yes:
        #        if element -1 >= 0:
        #            Y[i,element-1] = 1
        #        if element +1 <= 87:
        #            Y[i,element+1] = 1

        return x_norm, Y


class Dataset_CSV_test(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, location, minsize = 128):
        'Initialization'
        self.locationX = location + '/data/'
        self.locationY = location + '/labels/'
        self.minsize = minsize
        files = listdir_nohidden(self.locationX)
        self.list_IDs = list(filter(lambda f: f.endswith('.csv'), files))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        ID = self.list_IDs[index]

        # Load data and get label

        df_data = pd.read_csv(self.locationX + ID)
        df_labels = pd.read_csv(self.locationY + ID)
        size = df_data.shape[0]

        #for mfcc
        #position = 3500 #The shortest song has 9263 windows, so if all the songs start at the window 3500
                         #We are avoiding the start of the song when there's nothing being played.

        #for CQT
        position = 172

        X = np.empty((self.minsize, 5, df_data.shape[1])) #minsize because its the smallest song size
        Y = np.zeros((self.minsize,88))

        auxiliary = 0

        for j in range(position, position+self.minsize -5):

            if j == position:
                for k in range(5):
                    #Creating the 5 windows image
                    X[auxiliary][k] = df_data.iloc[j + k]
                if j < (position + self.minsize - 2):
                    #Creating the labels, taking the label of the mid window
                    Y[auxiliary] = df_labels.iloc[j + 2]

                auxiliary+=1
            else:
                for k in range(5):
                    if k < 3:
                      #Copying the values of the previous image to avoid data repetition
                      X[auxiliary][k] = X[auxiliary - 1][k + 1]
                    else:
                        X[auxiliary][k] = df_data.iloc[j + k]

                if j < (position + self.minsize - 2):
                    #Creating the labels, taking the label of the mid window
                    Y[auxiliary] = df_labels.iloc[j + 2]

                auxiliary+=1

        #Normalise X using min-max normalisation

        #For filterbanks
        #max_value = 120.1003100275508
        #min_value = -330.7445963332333

        #For MFCC
        #max_value = 841.837813253034
        #min_value = -831.0670773337255

        #For CQT
        max_value = 41.775738
        min_value = 0

        x_norm = (X  - min_value) / (max_value - min_value)
        x_norm = np.reshape(x_norm,(self.minsize,1, 5, df_data.shape[1]))

        #for i,row in enumerate(Y):
        #    yes = np.where(row==1)[0]
        #    for element in yes:
        #        if element -1 >= 0:
        #            Y[i,element-1] = 1
        #        if element +1 <= 87:
        #            Y[i,element+1] = 1
        #

        return x_norm, Y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1,3))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3))
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1,3))
        self.conv3_bn = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d((1, 2))
        self.drop = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3*46*16, 256)
        self.fc2 = nn.Linear(256, 88)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.drop(self.pool(F.relu(self.conv3_bn(self.conv3(x)))))
        x = x.view(-1, 3*46*16)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return sigmoid(x)


class longNet(nn.Module):
    def __init__(self):
        super(longNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3,3))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.dropbig = torch.nn.Dropout(p=0.5)
        self.dropsmall = torch.nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1,3))
        self.conv3_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d((1, 2))

        self.linear1 = nn.Linear(3*46*32, 64)
        self.linear1_bn = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.linear2_bn = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 88)
        self.leakyRelu = torch.nn.LeakyReLU()

    def forward(self, x):
        #print("0 ", x.shape)

        #print("XX0: ", x)
        x = self.conv1(x)
        #print("XX1: ", x)
        x = self.leakyRelu(x)
        #print("XX2: ", x)
        x = self.conv1_bn(x)
        #print("XX3: ", x)
        x = self.pool(x)
        #print("XX4: ", x)
        x = self.dropsmall(x)
        #print("XX5: ", x)
        #print("1: ", x)
        #print("1 ", x.shape)
        x = self.conv2(x)
        x = self.leakyRelu(x)
        x = self.conv2_bn(x)
        x = self.dropsmall(self.pool(x))
        #print("2: ", x)
        #print("2 ", x.shape)
        x = self.dropsmall(self.pool(self.leakyRelu(self.conv3_bn(self.conv3(x)))))
        #print("3: ", x)
        #print("3 ", x.shape)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.dropsmall(self.leakyRelu(self.linear1_bn(self.linear1(x))))
        #print("4: ", x)
        x = self.dropbig(self.leakyRelu(self.linear2_bn(self.linear2(x))))
        #print("5: ", x)
        x = self.linear3(x)
        #print("6: ", x)
        return sigmoid(x)

def train(epochs):
    trainGenerator = Dataset_CSV_train("finalDatasetCQT/train")
    trainloader = DataLoader(trainGenerator, batch_size=32,
                            shuffle=True, num_workers=4)

    testGenerator = Dataset_CSV_test("finalDatasetCQT/test")
    testloader = DataLoader(testGenerator, batch_size=2,
                            shuffle=True, num_workers=2)


    net = longNet()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    net = net.to(device)

    print("Training the model on: ", device)

    print(repr(net))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())


    for epoch in range(epochs):  # loop over the dataset multiple times
        loss_values = []
        running_loss = 0.0
        aux = 0
        #net.train() # pytorch way to make model trainable.
        print("Training")
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            for j in range(inputs.shape[0]):
                X = inputs[j].float().to(device)
                Y = labels[j].float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(X)

                loss = criterion(outputs, Y)

                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())

                #print("The loss is: ", loss.item())

                # print statistics
                running_loss += loss.item()
                if aux % 50 == 49:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                    loss_values.append(running_loss / 50)
                    running_loss = 0.0
                aux += 1

        test_loss = []
        test_accuracy = []
        print("Testing")
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                for j in range(inputs.shape[0]):
                    X = inputs[j].float().to(device)
                    Y = labels[j].float().to(device)

                    output = net(X)
                    loss = criterion(output, Y)
                    test_loss.append(loss.item())

                    max_indices = np.where(output.cpu() > 0.51, 1, 0)
                    predicted_correctly = 0
                    total = 0
                    for i, row in enumerate(Y):
                        for j, col in enumerate(row):
                            if col.item() == 1:
                                total+= 1
                            if col.item() == 0 and max_indices[i][j].item() == 0:
                                continue
                            elif col.item() == 1 and max_indices[i][j].item() == 1:
                                print("PREDICTED CORRECTLY")
                                predicted_correctly += 1
                            else:
                                continue
                    train_acc = predicted_correctly/total
                    test_accuracy.append(train_acc)

        print("##########")
        print("EPOCH ", epoch)
        print("##########")

        print("##TRAINING STATS##\n")
        print("Loss: ", sum(loss_values)/ len(loss_values))
        #print("Acc: ", sum(test_accuracy)/ len(test_accuracy))
        print("######\n")

        print("##VALIDATION STATS##\n")
        print("Loss: ", sum(test_loss)/ len(test_loss))
        print("Acc: ", sum(test_accuracy)/ len(test_accuracy))
        print("Predicted correctly: ", predicted_correctly, "out of: ", total)
        print("######\n")

    print('Finished Training')

    torch.save(net, "LastModel.pt")

def main(epochs):
    num_epochs = train(int(epochs))

if __name__ == "__main__":
    main(sys.argv[1])
