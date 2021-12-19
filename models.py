from torch import nn,optim
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
#plt.switch_backend('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel,self).__init__()
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

    def forward(self,x):
        output = self.cnnModel(x)

        output = output.squeeze()
        return output


class SimulationDecoderModel(nn.Module):
    def __init__(self):
        super(SimulationDecoderModel, self).__init__()
        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        output = self.dnnModel(x)
        return output


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel,self).__init__()
        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )


    def forward(self,x):
        output = self.dnnModel(x)
        return output


class DecoderModel(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(DecoderModel, self).__init__()
        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        self.device = device
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.dnnModel.parameters(), lr=self.lr)
        self.dnnModel.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.privacy_list = []


    def train_decoder(self, total_epoch):
        train_loader = self.train_loader
        test_loader = self.test_loader
        device = self.device
        train_loss_list = []

        for epoch in range(total_epoch):
            print("train decoder {}/{}".format(epoch + 1, total_epoch))
            decoder_train_loss = 0.0
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(train_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
                outputs = self.dnnModel(decoder_inputs)
                loss = self.criterion(outputs, decoder_targets.long())
                train_loss_list.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                decoder_train_loss += loss.item() * decoder_inputs.size(0)
            train_loss = decoder_train_loss / float(len(train_loader.dataset))

            test_correct = 0
            with torch.no_grad():
                for batch_idx, (decoder_inputs, decoder_targets) in enumerate(test_loader):
                    decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
                    outputs = self.dnnModel(decoder_inputs)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(decoder_targets).sum().item()

            privacy = 1 - test_correct / float(len(test_loader.dataset))
            # print("test privacy = ", privacy)
            self.privacy_list.append(privacy)

        #plt.plot(train_loss_list)
        #plt.show()

        print("lowest: ", min(self.privacy_list))
        print("last: ", self.privacy_list[-1])

        return min(self.privacy_list)
        


class CloudTask(nn.Module):
    def __init__(self,train_loader,test_loader):
        super(CloudTask, self).__init__()
        self.model = TopModel()
        self.device = device
        self.lr = 0.001
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader


    def train_model(self,topModelEpoch=30):
        total_epoch = topModelEpoch
        train_loader = self.train_loader
        device = self.device
        #pdb = tqdm(range(total_epoch))

        for epoch in range(total_epoch):
            print("train top model {}/{}".format(epoch+1, total_epoch))
            train_loss = 0.0
            correct = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()*inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            train_acc = correct / float(len(train_loader.dataset))
            print("train accuracy = {:.2f}%".format(train_acc * 100))


    def test_model(self):
        test_loader = self.test_loader
        device = self.device
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        test_acc = correct/float(len(test_loader.dataset))
        print("test accuracy = {:.2f}%".format(test_acc * 100))

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + 1, 256)
        self.l2 = nn.Linear(256 , 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x