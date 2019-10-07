import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

#Load Data
with open("image", "rb") as f:
        images = np.array(pickle.load(f))
with open("angle", "rb") as f:
        angles = np.array(pickle.load(f))
        
#Shuffle Data
features, labels = shuffle(images, angles)

#Split Dataset
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,test_size=0.3)

train_x = torch.from_numpy(train_x).float()
test_x = torch.from_numpy(test_x).float()
train_y = torch.from_numpy(train_y).float()
test_y = torch.from_numpy(test_y).float()

train_x.shape, train_y.shape

#Build Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3,padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3,padding=2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3,padding=2)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=2)
        self.pool6 = nn.MaxPool2d(2, 2)
        
        x = torch.rand(100,100).view(-1,1,100,100)
        self.to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self.to_linear, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
      
    def convs(self, x):
          x = self.pool1(F.relu(self.conv1(x)))
          x = self.pool2(F.relu(self.conv2(x)))
          x = self.pool3(F.relu(self.conv3(x)))
          x = self.pool4(F.relu(self.conv4(x)))
          x = self.pool5(F.relu(self.conv5(x)))
          x = self.pool6(F.relu(self.conv6(x)))
          
          if self.to_linear is None:
              self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
          
          return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)      #Optimizer
loss_function = nn.MSELoss()


BATCH_SIZE = 32
EPOCHS = 3
for epoch in range(EPOCHS):     #Training
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)): 
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_x[i:i+BATCH_SIZE].view(-1, 1, 100, 100)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")

#Save Model
PATH = './steer_net.pth'
torch.save(net.state_dict(), PATH)
