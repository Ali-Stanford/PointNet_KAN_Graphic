# In The Name of God
##### PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets #####

#Author: Ali Kashefi (kashefi@stanford.edu)

##### Citation #####
#If you use the code, please cite the following journal paper: 
#[PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets]
# https://arxiv.org/abs/2410.10084

#@article{kashefi2024PointNetKAN,
#title={PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets},
#author={Kashefi, Ali},
#journal={arXiv preprint arXiv:2410.10084},
#year={2024}}

# Libraries
import torch
import torch.utils.data as data
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import linecache
from operator import itemgetter
from numpy import zeros
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

#Parameter setup

NUM_POINTS = 1024 
NUM_CLASSES = 40 # ModelNet40
BATCH_SIZE = 64 
poly_degree = 4 # Polynomial degree of Jacaboi Polynomial
FEATURE = 6 # (x,y,z) + (nx,ny,nz)
ALPHA = 1.0 # \alpha in Jacaboi Polynomial
BETA = 1.0 # \beta in Jacaboi Polynomial
SCALE = 3.0 # To control the size of tensor A in the manuscript
MAX_EPOCHS = 300
direction = './ModelNet40'

###### Function: parse_dataset ######
def parse_dataset(num_points=NUM_POINTS):
    train_points_with_normals = []
    train_labels = []
    test_points_with_normals = [] 
    test_labels = []
    class_map = {}

    DATA_DIR = direction 
    folders = glob.glob(os.path.join(DATA_DIR, "*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            mesh = trimesh.load(f)
            points, face_indices = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_indices]
            
            points_with_normals = np.concatenate([points, normals], axis=1)
            
            train_points_with_normals.append(points_with_normals)
            train_labels.append(i)

        for f in test_files:
            mesh = trimesh.load(f)
            points, face_indices = mesh.sample(num_points, return_index=True)
            normals = mesh.face_normals[face_indices]  
            
            points_with_normals = np.concatenate([points, normals], axis=1)
            
            test_points_with_normals.append(points_with_normals)
            test_labels.append(i)

    train_points = torch.tensor(np.array(train_points_with_normals), dtype=torch.float32)
    test_points = torch.tensor(np.array(test_points_with_normals), dtype=torch.float32)
    train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
    test_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

    return train_points, test_points, train_labels, test_labels, class_map

###### Object: PointCloudDataset ######
class PointCloudDataset(Dataset):
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels
        self.normalize()

    def normalize(self):
        
        for i in range(self.points.shape[0]):
            
            spatial_coords = self.points[i, :, :3]  
            normals = self.points[i, :, 3:] 
            
            centroid = spatial_coords.mean(axis=0, keepdims=True)
            spatial_coords -= centroid

            furthest_distance = torch.max(torch.sqrt(torch.sum(spatial_coords ** 2, axis=1, keepdims=True)))
            spatial_coords /= furthest_distance

            self.points[i] = torch.cat((spatial_coords, normals), dim=1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        label = self.labels[idx]
        return point, label

###### Object: KANshared (i.e., shared KAN) ######
class KANshared(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KANshared, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous() 
        x = torch.tanh(x) 

        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        jacobi = jacobi.permute(0, 2, 3, 1)  
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs) 
        return y

###### Object: KAN ######
class KAN(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(KAN, self).__init__()
        self.inputdim = input_dim
        self.outdim   = output_dim
        self.a        = a
        self.b        = b
        self.degree   = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim)) 
        
        x = torch.tanh(x)
        
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, i] = (A*x + B)*jacobi[:, :, i-1].clone() + C*jacobi[:, :, i-2].clone()

        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs) 
        y = y.view(-1, self.outdim)
        return y

###### Object: PointNetKAN for classification (i.e., PointNet-KAN) ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()

        self.jacobikan5 = KANshared(input_channels, int(1024 * scaling), poly_degree)
        self.jacobikan6 = KAN(int(1024 * scaling), output_channels, poly_degree)

        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))

    def forward(self, x):

        x = self.jacobikan5(x)
        x = self.bn5(x)

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        x = self.jacobikan6(global_feature)
        return x

###### Loading data, setting devices ######

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

train_dataset = PointCloudDataset(train_points, train_labels)
test_dataset = PointCloudDataset(test_points, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Model setup ######

model = PointNetKAN(input_channels=FEATURE, output_channels=NUM_CLASSES, scaling = SCALE).to(device) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

###### Training ######
for epoch in range(MAX_EPOCHS):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0

    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)

        points = points.transpose(1, 2)

        optimizer.zero_grad()

        outputs = model(points)
      
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * points.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct / total


    model.eval() 
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    class_correct_val = np.zeros(NUM_CLASSES) 
    class_total_val = np.zeros(NUM_CLASSES)  

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)

            points = points.transpose(1, 2)

            outputs = model(points)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * points.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total_val[label] += 1
                class_correct_val[label] += (predicted[i] == label).item()

    val_loss /= len(test_loader.dataset)
    val_accuracy = 100 * val_correct / val_total

    class_accuracy_val = 100 * np.divide(class_correct_val, class_total_val, out=np.zeros_like(class_correct_val), where=class_total_val != 0)
    average_class_accuracy_val = np.mean(class_accuracy_val)

    scheduler.step()

    print(f"Epoch {epoch+1}/{MAX_EPOCHS}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, "
          f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%, "
          f"Avg Class Accuracy Test: {average_class_accuracy_val:.2f}%")
