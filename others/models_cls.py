# Libraries
import torch
import torch.utils.data as data
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import linecache
from operator import itemgetter
from numpy import zeros

#Parameter setup
NUM_POINTS = 1024 
NUM_CLASSES = 40 # ModelNet40
BATCH_SIZE = 64 
poly_degree = 4 # Polynomial degree of Jacaboi Polynomial
FEATURE = 6 # (x,y,z) + (nx,ny,nz)
ALPHA = 1.0 # \alpha in Jacaboi Polynomial
BETA = 1.0 # \beta in Jacaboi Polynomial
SCALE = 3.0 # To control the size of tensor A in the manuscript

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

##### Deep Point Net KAN without T-Nets ##### 
class DeepPointNetKAN1(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()

        #Shared KAN (64, 64)
        self.jacobikan1 = KANshared(input_channels, int(64 * scaling), poly_degree)
        self.jacobikan2 = KANshared(int(64 * scaling), int(64 * scaling), poly_degree)

        #Shared KAN (64, 128, 1024)
        self.jacobikan3 = KANshared(int(64 * scaling), int(64 * scaling), poly_degree)
        self.jacobikan4 = KANshared(int(64 * scaling), int(128 * scaling), poly_degree)
        self.jacobikan5 = KANshared(int(128 * scaling), int(1024 * scaling), poly_degree)

        #Shared KAN (512, 256, num_classes)
        self.jacobikan6 = KAN(int(1024 * scaling), int(512 * scaling), poly_degree)
        self.jacobikan7 = KAN(int(512 * scaling), int(256 * scaling), poly_degree)
        self.jacobikan8 = KAN(int(256 * scaling), output_channels, poly_degree)

        #Batch Normalization
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))

        #DropOut Layer
        self.dropout2 = nn.Dropout(0.3) 

    def forward(self, x):

        # Shared KAN (64, 64)
        x = self.jacobikan1(x)
        x = self.bn1(x)
        x = self.jacobikan2(x)
        x = self.bn2(x)

        # Shared KAN (64, 128, 1024)
        x = self.jacobikan3(x)
        x = self.bn3(x)
        x = self.jacobikan4(x)
        x = self.bn4(x)
        x = self.jacobikan5(x)
        x = self.bn5(x)

        # Max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)

        # KAN (512, 256, num_classes)
        x = self.jacobikan6(global_feature)
        x = self.bn6(x)
        x = self.jacobikan7(x)
        x = self.bn7(x)
        x = self.dropout2(x)
        x = self.jacobikan8(x)
  
        return x

