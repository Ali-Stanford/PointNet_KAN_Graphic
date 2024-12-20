# Libraries
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
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter Setting
input_channels = 3  # x, y, z
output_channels = 50  # total number of parts in ShapeNet part
num_objects = 16 # number of objects in ShapeNet part
scaling = 5.0 # To control the size of tensor A in the manuscript
ALPHA = -0.5 # \alpha in Jacaboi Polynomial
BETA = -0.5 # \beta in Jacaboi Polynomial
poly_degree = 2 # Polynomial degree of Jacaboi Polynomial


###### Data loading and data preparation ######
hdf5_data_dir = '/shapenet_part_seg_hdf5_data'
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'test_hdf5_file_list.txt')

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


test_files = get_data_files(TESTING_FILE_LIST)
test_data, test_labels, test_seg = load_data_files(test_files)

###### part color mapping ######
part_color_mapping = {
    0: [1.0, 0.0, 0.0],      # Bright Red
    1: [0.0, 1.0, 0.0],      # Bright Green
    2: [0.0, 0.0, 1.0],      # Bright Blue
    3: [1.0, 1.0, 0.0],      # Yellow
    4: [1.0, 0.0, 1.0],      # Magenta
    5: [0.0, 1.0, 1.0],      # Cyan
    6: [0.5, 0.0, 0.5],      # Dark Purple
    7: [0.0, 0.5, 0.0],      # Dark Green
    8: [0.5, 0.5, 0.0],      # Olive
    9: [1.0, 0.65, 0.0],     # Orange
   10: [0.0, 0.5, 0.5],      # Teal
   11: [0.5, 0.0, 0.0],      # Dark Red
   12: [0.8, 0.2, 0.0],      # Burnt Orange
   13: [0.0, 0.8, 0.2],      # Lime Green
   14: [0.2, 0.8, 0.0],      # Leaf Green
   15: [0.0, 0.2, 0.8],      # Deep Blue
   16: [0.8, 0.0, 0.2],      # Pinkish Red
   17: [0.2, 0.0, 0.8],      # Violet
   18: [0.0, 0.8, 0.8],      # Light Teal
   19: [0.8, 0.8, 0.0],      # Light Yellow
   20: [0.6, 0.4, 0.2],      # Brown
   21: [0.2, 0.6, 0.4],      # Sea Green
   22: [0.4, 0.2, 0.6],      # Lavender
   23: [0.6, 0.2, 0.4],      # Rose
   24: [0.4, 0.6, 0.2],      # Moss Green
   25: [0.2, 0.4, 0.6],      # Sky Blue
   26: [0.9, 0.1, 0.1],      # Bright Coral
   27: [0.1, 0.9, 0.1],      # Bright Lime
   28: [0.1, 0.1, 0.9],      # Deep Blue
   29: [0.8, 0.1, 0.6],      # Fuchsia
   30: [0.6, 0.8, 0.1],      # Chartreuse 
   31: [0.1, 0.6, 0.8],      # Turquoise
   32: [0.7, 0.3, 0.7],      # Orchid
   33: [0.3, 0.7, 0.3],      # Mint Green
   34: [0.7, 0.3, 0.3],      # Salmon
   35: [0.3, 0.7, 0.7],      # Pale Cyan
   36: [1.0, 0.5, 0.0],      # Bright Orange
   37: [0.5, 0.5, 1.0],      # Light Blue
   38: [1.0, 0.5, 0.5],      # Light Coral
   39: [0.5, 1.0, 0.5],      # Pale Green
   40: [0.5, 0.0, 1.0],      # Deep Purple
   41: [0.5, 0.5, 0.0],      # Mustard
   42: [1.0, 0.0, 0.5],      # Hot Pink
   43: [0.5, 1.0, 0.0],      # Light Lime
   44: [0.0, 1.0, 0.5],      # Emerald
   45: [1.0, 0.8, 0.2],      # Golden Yellow
   46: [0.5, 0.5, 0.8],      # Lavender Blue
   47: [0.8, 0.5, 0.5],      # Dusty Rose
   48: [0.5, 0.8, 0.5],      # Light Olive
   49: [0.8, 0.5, 0.8]       # Light Magenta
}


class VariableLengthPointCloudDataset(Dataset):
    def __init__(self, data, seg_labels, obj_labels):
        self.data = data
        self.seg_labels = seg_labels
        self.obj_labels = obj_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx] 
        seg_label = self.seg_labels[idx] 
        obj_label = self.obj_labels[idx]  

        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        seg_label = torch.tensor(seg_label, dtype=torch.long)
        obj_label = torch.tensor(obj_label, dtype=torch.long)

        return point_cloud, seg_label, obj_label

def variable_length_collate_fn(batch):
    point_clouds = []
    seg_labels = []
    obj_labels = []

    for data, seg_label, obj_label in batch:
        point_clouds.append(data)
        seg_labels.append(seg_label)
        obj_labels.append(obj_label)

    obj_labels = torch.stack(obj_labels)

    return point_clouds, seg_labels, obj_labels

test_dataset = VariableLengthPointCloudDataset(test_data, test_seg, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=variable_length_collate_fn)

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
        x = torch.tanh(x)  # Normalize x to [-1, 1]

        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        # Compute the Jacobi interpolation
        jacobi = jacobi.permute(0, 2, 3, 1) 
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs) 

        return y

###### Object: PointNetKAN for segmentation (i.e., PointNet-KAN) ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=1.0):
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
        class_label = class_label.view(-1, class_label.size(1), 1).expand(-1, -1, x.size(2))

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, x.size(2))

        x = torch.cat([local_4, global_feature, class_label], dim=1)

        x = self.jacobikan9(x)
        x = self.bn9(x)
        x = self.jacobikan10(x)

        return x

#### plotting tools and functions ####
def rotate_90_degrees_x(points):
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    return np.dot(points, rotation_matrix.T)

def create_category_folder(category_number):
    category_folder = os.path.join(save_dir, str(category_number))
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)
    return category_folder

def plot_point_cloud_with_labels(points, labels, save_path, obj_category, num_parts, dpi=300):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rotated_points = rotate_90_degrees_x(points)

    relevant_parts = object_part_mapping_numeric[obj_category]['parts']

    colors = np.array([part_color_mapping[relevant_parts[label]] for label in labels])

    scatter = ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], 
                         c=colors, marker='o')

    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Save the figure with 300 DPI
    plt.savefig(f"{save_path}.png", format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# One-hot encode class labels
def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels.long()]

# Mapping for numerical labels (assuming 0-15 correspond to object categories)
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

# Evaluate and save all samples for all categories
def evaluate_and_save_all_samples(loader, dataset_name="test"):
    model.eval()
    sample_count = {i: 0 for i in range(16)}  # To track samples for each category

    with torch.no_grad():
        for batch_data_list, batch_labels_list, batch_obj_labels in loader:
            batch_obj_labels = batch_obj_labels.to(device)

            for data, labels, obj_label in zip(batch_data_list, batch_labels_list, batch_obj_labels):
                data = data.to(device).unsqueeze(0).transpose(1, 2)  # [1, 3, num_points]
                labels = labels.to(device)
                obj_label = obj_label.unsqueeze(0)

                obj_one_hot = one_hot_encode(obj_label, num_objects).to(device)

                outputs = model(data, obj_one_hot)
                outputs = outputs.permute(0, 2, 1)  # [1, num_points, output_channels]
                outputs = outputs.squeeze(0)  # [num_points, output_channels]

                obj_category = obj_label.item()
                category_folder = create_category_folder(obj_category)

                relevant_parts = object_part_mapping_numeric[obj_category]['parts']
                num_parts = len(relevant_parts)

                masked_output = outputs[:, relevant_parts].cpu().numpy()

                part_label_mapping = {orig_label: idx for idx, orig_label in enumerate(relevant_parts)}
                remapped_labels = labels.cpu().numpy()

                for orig_label, idx in part_label_mapping.items():
                    remapped_labels[remapped_labels == orig_label] = idx  # Remap the labels

                pred_labels_for_obj = np.argmax(masked_output, axis=1)

                points = data.squeeze(0).transpose(0, 1).cpu().numpy()
                gt_labels = remapped_labels

                sample_index = sample_count[obj_category]
                gt_save_path = os.path.join(category_folder, f"GroundTruth_{sample_index}")
                pred_save_path = os.path.join(category_folder, f"Prediction_{sample_index}")

                plot_point_cloud_with_labels(points, gt_labels, gt_save_path, obj_category, num_parts)

                plot_point_cloud_with_labels(points, pred_labels_for_obj, pred_save_path, obj_category, num_parts)

                sample_count[obj_category] += 1  

###### Load the model and evaluate ######
######  PointNet-KAN Model ######
model = PointNetKAN(input_channels, output_channels, scaling=scaling).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

save_dir = os.getcwd()
evaluate_and_save_all_samples(test_loader, "test")
