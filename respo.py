import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from   sklearn.preprocessing import StandardScaler, LabelEncoder

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from glob import glob

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),   # Resize the image
            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
        ])
        self.diagnosis = pd.read_csv("/kaggle/input/patient-data/patient_diagnosis.csv",
                                     names = ["patient", "diagnosis"])
        unique_diags = self.diagnosis["diagnosis"].unique()
        self.diag_to_label_map = {}
        self.label_to_diag_map = {}
        label = 0
        for index, diag in enumerate(unique_diags):
            if diag in ['LRTI', 'Asthma']:
                continue
            self.diag_to_label_map[diag] = label
            self.label_to_diag_map[label] = diag
            label += 1

        print(f"There are {len(self.file_paths)} files in the dataset")
        #print("Diagnosis to Label map:", self.diag_to_label_map)
        #print("Label to Diagnosis map:", self.label_to_diag_map)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        # Find the diagnosis of this patient
        patient_id = int(os.path.basename(image_path).split('_')[0])
        diag_row_index = self.diagnosis.index[self.diagnosis.patient == patient_id][0]
        diag = self.diagnosis.at[diag_row_index, "diagnosis"]
        label = self.diag_to_label_map[diag]

        return image, label, patient_id
    
    def getDiagnosis(self, label):
        return self.label_to_diag_map[label]

file_paths = glob("/kaggle/input/respiratory-sounds/*.jpg")
custom_dataset = CustomDataset(file_paths)

# train-test split
train_ratio = 0.8
train_size = int(len(custom_dataset) * train_ratio * train_ratio)
validation_size = int((len(custom_dataset) * train_ratio) - train_size)
test_size = len(custom_dataset) - train_size - validation_size

# split indices
indices = list(range(len(custom_dataset)))
train_indices, validation_indices, test_indices = indices[:train_size], indices[train_size:(train_size+validation_size)], indices[(train_size+validation_size):]

# create subset objects for training and testing
train_dataset = Subset(custom_dataset, train_indices)
validation_dataset = Subset(custom_dataset, validation_indices)
test_dataset = Subset(custom_dataset, test_indices)

# create dataloaders for training and testing
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Dataset: %d, Train: %d, Validation: %d, Test: %d" % (
      len(custom_dataset), len(train_loader) * batch_size, len(validation_loader) * batch_size, len(test_loader) * batch_size))


import matplotlib.pyplot as plt
import numpy as np

# function to show an image
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels, patient_ids = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
print("Actual Diagnosis:", [custom_dataset.getDiagnosis(l.item()) for l in labels])


demo_data = pd.read_csv("/kaggle/input/respiratory-sound-database/demographic_info.txt", names = ["patient id", "age", "sex", "bmi", "weight", "height"], delimiter = " ")
print(demo_data.head(10))
print()

# remove patient 223 since they have no data
#demo_data.drop(index=122, inplace=True)

# figure out which are categorical vs. numerical columns
cols_cat = [col for col in demo_data.columns if demo_data[col].dtype == 'object'] # list comprehension
cols_num = [col for col in demo_data.columns if col not in cols_cat]
assert(len(cols_cat) + len(cols_num) == demo_data.shape[1])

# fill null values in numerical column with mean
demo_data[cols_num] = demo_data[cols_num].fillna(demo_data[cols_num].mean())
# fill null value in categorical columns with mode
demo_data[cols_cat] = demo_data[cols_cat].fillna(demo_data[cols_cat].mode().iloc[0])

# label encoding
encoder = LabelEncoder()
demo_data['sex'] = encoder.fit_transform(demo_data['sex'])

# standard scaler
scaler = StandardScaler()
demo_data['age'] = scaler.fit_transform(demo_data[['age']]).flatten()
scaler = StandardScaler()
demo_data['bmi'] = scaler.fit_transform(demo_data[['bmi']]).flatten()

demo_data = demo_data[['patient id', 'age', 'sex', 'bmi']]

print(demo_data.head(10))

demo_vector_by_id = {}

# build the demo vector in demo_data, turn it into a tensor, and contain it in the dictionary demo_vector_by_id
for index, row in demo_data.iterrows():
    demo_vector_by_id[int(row['patient id'])] = [row['age'], row['sex'], row['bmi']]

#print(demo_vector_by_id)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, demo_vector_by_id):
        super().__init__()
        self.demo_vector_by_id = demo_vector_by_id
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 47 * 122, 120)
        self.fc2 = nn.Linear(120 + 3, 84)
        self.fc3 = nn.Linear(84, 6)
        self.debug = False

    def forward(self, x, patient_ids):
        x = self.pool(F.relu(self.conv1(x)))
        if self.debug: print("After 1st conv layer, shape is", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        if self.debug: print("After 2nd conv layer, shape is", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        if self.debug: print("After flatten, shape is", x.shape)

        x = F.relu(self.fc1(x))
        if self.debug: print("After 1st FC layer, shape is", x.shape)
            
        # Append Demographics vector
        demo_vector = [self.demo_vector_by_id[int(id)] for id in patient_ids.tolist()]
        demo_tensor = torch.tensor(demo_vector, dtype=torch.float32, device=x.device)
        x = torch.cat((x, demo_tensor), dim=1)
        if self.debug: print("After adding demo vector, shape is", x.shape)
            
        x = F.relu(self.fc2(x))
        if self.debug: print("After 2nd FC layer, shape is", x.shape)
        x = self.fc3(x)
        if self.debug: print("After 3rd FC layer, shape is", x.shape)
            
        return x


net = Net(demo_vector_by_id)
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

dataiter = iter(train_loader)
images, labels, patient_ids = next(dataiter)
print(images.shape, labels, patient_ids)

net.debug=True
outputs = net(images, patient_ids)
net.debug=False

import time

total = 0
start_time = time.time()

# make x and y lists for the loss graph
y_train = [] # training loss
y_val = [] # validation loss

num_epochs = 20
device = 'cuda:0'

net = net.to(device)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    
    # run on training set
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, patient_ids = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        patient_ids = patient_ids.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, patient_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total += labels.size(0)
        if total % 1000 == 0:
            print(f'[Epoch: {epoch + 1}, Elapsed time: {round(time.time() - start_time):4d} seconds] ' +
                  f'  Processed {total} images so far')
            
    y_train.append(running_loss / len(train_dataset))

    print(f'[Epoch: {epoch + 1}, Elapsed time: {round(time.time() - start_time):4d} seconds] ' +
          f'Trained on {total} images, Total loss: {running_loss}\n')

    running_loss = 0.0
    total = 0
    
    # predict on validation set
    for i, data in enumerate(validation_loader, 0): 
        inputs, labels, patient_ids = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        patient_ids = patient_ids.to(device)
        
        # forward pass
        outputs = net(inputs, patient_ids)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        total += 1
        
    y_val.append(running_loss / len(validation_dataset))
        
    
    print(f'[Epoch: {epoch + 1}, Elapsed time: {round(time.time() - start_time):4d} seconds] ' +
          f'Validated on {total} images, Total loss: {running_loss}\n')

print('y_train is', y_train)
print('y_val is', y_val)

for data in validation_loader:
    print(data)
    break

MODEL_PATH = "cnn_model"
torch.save(net.state_dict(), MODEL_PATH)

x = list(range(1, len(y_train) + 1))
print(x)
print(y_train)

plt.plot(x, y_train, label='Training', marker='o', linestyle='-', color='blue')
plt.plot(x, y_val, label='Validation', marker='s', linestyle='--', color='red')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph with Training and Validation Losses')

plt.legend()

plt.show()

# get some random training images
dataiter = iter(test_loader)
images, labels, patient_ids = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
print("Actual Diagnosis:    ", [custom_dataset.getDiagnosis(l.item()) for l in labels])


# Run the images through the model
outputs = net(images, patient_ids)
_, predicted = torch.max(outputs, 1)

print("Predicted Diagnosis: ", [custom_dataset.getDiagnosis(l.item()) for l in predicted])

correct = 0
total = 0

y_true = []
y_pred = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs, labels, patient_ids = data
        
        # calculate outputs by running images through the network
        outputs = net(inputs, patient_ids)
        
        # prepare for confusion matrix
        for label in labels:
            y_true.append(int(label))
        for pred in predicted:
            y_pred.append(int(pred))
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy of the network on {total} images: {100 * correct // total} %')

