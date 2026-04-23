# In The Name of God
##### PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets #####

#Author: Ali Kashefi (kashefi@stanford.edu)

##### Citation #####
#If you use the code, please cite the following journal paper: 
#[PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets]
# https://doi.org/10.1016/j.cag.2025.104319

#@article{kashefi2025PointNetKANgraphics,
#title={PointNet with KAN versus PointNet with MLP for 3D classification and segmentation of point sets},
#author={Kashefi, Ali},
#journal={Computers \& Graphics},
#pages={104319},
#year={2025},
#publisher={Elsevier}}

###### Libraries ######
import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###### Parameter setup ######
num_points = 2048
input_channels = 3 # (x,y,z)
output_channels = 50  # totla number of parts in ShapeNet part
num_objects = 16 # number of objects in ShapeNet part
SCALE = 5.0 # To control the size of tensor A in the manuscript
ALPHA = -0.5 # \alpha in Jacaboi Polynomial
BETA = -0.5 # \beta in Jacaboi Polynomial
poly_degree = 2 # Polynomial degree of Jacaboi Polynomial
batch_size = 32
max_epochs = 200

###### Data loading and data preparation ######
hdf5_data_dir = '/shapenet_part_seg_hdf5_data'
TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
VALIDATION_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

def get_data_files(file_list):
    with open(file_list, 'r') as f:
        return [line.rstrip() for line in f]

def load_h5(h5_filename):
    full_path = os.path.join(hdf5_data_dir, h5_filename)
    print(f"Loading file: {full_path}")
    try:
        f = h5py.File(full_path, 'r')
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        f.close()
        return data, label, seg
    except Exception as e:
        print(f"Error loading file {full_path}: {e}")
        raise

def load_data_files(file_list):
    all_data = []
    all_label = []
    all_seg = []
    for h5_filename in file_list:
        data, label, seg = load_h5(h5_filename)
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

def uniformly_sample_points(data, seg, num_points=2048):
    sampled_data = []
    sampled_seg = []
    for i in range(data.shape[0]):
        num_shape_points = data[i].shape[0]
        if num_shape_points > num_points:
            sampled_indices = np.random.choice(num_shape_points, num_points, replace=False)
        else:
            sampled_indices = np.random.choice(num_shape_points, num_points, replace=True)
        sampled_data.append(data[i][sampled_indices, :])
        sampled_seg.append(seg[i][sampled_indices])
    return np.array(sampled_data), np.array(sampled_seg)

train_files = get_data_files(TRAINING_FILE_LIST)
train_data, train_labels, train_seg = load_data_files(train_files)
train_data_sampled, train_seg_sampled = uniformly_sample_points(train_data, train_seg, num_points=2048)

validation_files = get_data_files(VALIDATION_FILE_LIST)
validation_data, validation_labels, validation_seg = load_data_files(validation_files)
validation_data_sampled, validation_seg_sampled = uniformly_sample_points(validation_data, validation_seg, num_points=2048)

train_dataset = TensorDataset(torch.tensor(train_data_sampled, dtype=torch.float32),
                              torch.tensor(train_seg_sampled, dtype=torch.long),
                              torch.tensor(train_labels, dtype=torch.long))

val_dataset = TensorDataset(torch.tensor(validation_data_sampled, dtype=torch.float32),
                            torch.tensor(validation_seg_sampled, dtype=torch.long),
                            torch.tensor(validation_labels, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

###### Mapping for numerical labels (assuming 0-15 correspond to object categories) ######
object_part_mapping_numeric = {
    0: {'num_parts': 4, 'parts': [0, 1, 2, 3]},    # Aero
    1: {'num_parts': 2, 'parts': [4, 5]},          # Bag
    2: {'num_parts': 2, 'parts': [6, 7]},          # Cap
    3: {'num_parts': 4, 'parts': [8, 9, 10, 11]},  # Car
    4: {'num_parts': 4, 'parts': [12, 13, 14, 15]},# Chair
    5: {'num_parts': 3, 'parts': [16, 17, 18]},    # Earphone
    6: {'num_parts': 3, 'parts': [19, 20, 21]},    # Guitar
    7: {'num_parts': 2, 'parts': [22, 23]},        # Knife
    8: {'num_parts': 4, 'parts': [24, 25, 26, 27]},# Lamp
    9: {'num_parts': 2, 'parts': [28, 29]},        # Laptop
    10: {'num_parts': 6, 'parts': [30, 31, 32, 33, 34, 35]}, # Motorbike
    11: {'num_parts': 2, 'parts': [36, 37]},       # Mug
    12: {'num_parts': 3, 'parts': [38, 39, 40]},   # Pistol
    13: {'num_parts': 3, 'parts': [41, 42, 43]},   # Rocket
    14: {'num_parts': 3, 'parts': [44, 45, 46]},   # Skateboard
    15: {'num_parts': 3, 'parts': [47, 48, 49]}    # Table
}

###### Object: KANshared (i.e., shared KAN) ######
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=ALPHA, b=BETA):
        super(JacobiKANLayer, self).__init__()
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

###### Object: PointNetKAN for segmentation (i.e., PointNet-KAN) ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()

        self.jacobikan4 = JacobiKANLayer(input_channels, int(128 * scaling), poly_degree)
        self.jacobikan5 = JacobiKANLayer(int(128 * scaling), int(1024 * scaling), poly_degree)

        self.jacobikan9 = JacobiKANLayer(int(1024 * scaling) + int(128 * scaling) + int(num_objects), int(128 * scaling), poly_degree)
        self.jacobikan10 = JacobiKANLayer(int(128 * scaling), output_channels, poly_degree)

        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))

    def forward(self, x, class_label):

        x = self.jacobikan4(x)
        x = self.bn4(x)

        local_4 = x

        x = self.jacobikan5(x)
        x = self.bn5(x)

        class_label = class_label.view(-1, num_objects, 1)
        class_label = class_label.view(-1, class_label.size(1), 1).expand(-1, -1, num_points)

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, num_points)

        x = torch.cat([local_4, global_feature, class_label], dim=1)

        x = self.jacobikan9(x)
        x = self.bn9(x)
        x = self.jacobikan10(x)

        return x

####### Function: to one-hot encode class labels ######
def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels.long()]

###### Function: compute loss ######
def compute_loss_for_relevant_parts(outputs, batch_seg_labels, batch_obj_labels, object_part_mapping_numeric, criterion):
    total_loss = 0 

    batch_obj_labels = batch_obj_labels.cpu()

    assert torch.all(batch_obj_labels >= 0) and torch.all(batch_obj_labels < len(object_part_mapping_numeric)), \
        f"Invalid object labels in batch_obj_labels: {batch_obj_labels}"

    for i, obj_label in enumerate(batch_obj_labels):
        object_category = obj_label.item() 

        assert object_category in object_part_mapping_numeric, f"Invalid object category: {object_category}"

        relevant_parts = object_part_mapping_numeric[object_category]['parts']  

        assert all(rp < outputs.shape[-1] for rp in relevant_parts), "Relevant part index out of bounds!"

        part_label_mapping = {orig_label: idx for idx, orig_label in enumerate(relevant_parts)}

        remapped_labels = torch.clone(batch_seg_labels[i])
        for orig_label, idx in part_label_mapping.items():
            remapped_labels[batch_seg_labels[i] == orig_label] = idx

        assert torch.all(remapped_labels >= 0) and torch.all(remapped_labels < len(relevant_parts)), \
            f"Invalid remapped part labels in batch_seg_labels for object {i}: {remapped_labels}"

        masked_output = outputs[i, :, relevant_parts]

        loss = criterion(masked_output.reshape(-1, len(relevant_parts)), remapped_labels.reshape(-1))

        total_loss += loss

    return total_loss

###### Model setup ######
model = PointNetKAN(input_channels, output_channels, scaling=SCALE)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

best_val_loss = float('inf')
best_model_path = os.path.join(os.getcwd(), 'best_model.pth')

###### Training Loop ######
for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0

    for batch_data, batch_seg_labels, batch_obj_labels in train_loader:
        batch_data, batch_seg_labels, batch_obj_labels = (
            batch_data.to(device),
            batch_seg_labels.to(device),
            batch_obj_labels.to(device),
        )
        
        batch_data = batch_data.transpose(1, 2)

        batch_obj_one_hot = one_hot_encode(batch_obj_labels, num_objects).to(device)

        optimizer.zero_grad()
        outputs = model(batch_data, batch_obj_one_hot)
        
        outputs = outputs.permute(0, 2, 1)

        loss = compute_loss_for_relevant_parts(outputs, batch_seg_labels, batch_obj_labels, object_part_mapping_numeric, criterion)

        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []
    val_obj_labels = []
    with torch.no_grad():
        for batch_data, batch_seg_labels, batch_obj_labels in val_loader:
            batch_data, batch_seg_labels, batch_obj_labels = (
                batch_data.to(device),
                batch_seg_labels.to(device),
                batch_obj_labels.to(device),
            )
            
            batch_data = batch_data.transpose(1, 2)

            batch_obj_one_hot = one_hot_encode(batch_obj_labels, num_objects).to(device)

            outputs = model(batch_data, batch_obj_one_hot)
            
            outputs = outputs.permute(0, 2, 1)

            loss = compute_loss_for_relevant_parts(outputs, batch_seg_labels, batch_obj_labels, object_part_mapping_numeric, criterion)
            
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{max_epochs}], validation loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss and epoch > 9:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation loss decreased ({best_val_loss:.4f}). Saving model...")
    
    scheduler.step()
