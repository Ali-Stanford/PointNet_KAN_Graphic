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
import h5py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

torch.backends.cudnn.benchmark = True

###### Parameter Setup ######
BASE_DIR       = '/S3D'
H5_SUBDIR      = 'indoor3d_sem_seg_hdf5_data'
DATA_DIR       = os.path.join(BASE_DIR, H5_SUBDIR)
ALL_H5_LIST    = os.path.join(DATA_DIR, 'all_files.txt')
ROOM_LIST_FILE = os.path.join(DATA_DIR, 'room_filelist.txt')
NUM_CLASSES    = 13 # Stanford 3D Semantic Parsing Data Set
BATCH_SIZE     = 16 # or 32 if memory permits
EPOCHS         = 100
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE = 5.0 # To control the size of tensor A in the manuscript
ALPHA = -0.5 # \alpha in Jacaboi Polynomial
BETA = -0.5 # \beta in Jacaboi Polynomial
poly_degree = 2 # Polynomial degree of Jacaboi Polynomial
 
###### Function ######
def read_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

###### Object ######
class S3DISHDF5Dataset(Dataset):
    def __init__(self, h5_list_txt, room_list_txt, areas):
        basenames = read_list(h5_list_txt)
        self.paths = [os.path.join(BASE_DIR, fn) for fn in basenames]
        rooms = read_list(room_list_txt)

        self.data = []
        self.labels = []

        counter = 0
        for path in self.paths:
            with h5py.File(path, 'r') as f:
                data_all = f['data'][:]  
                label_all = f['label'][:]

            for i in range(data_all.shape[0]):
                room = rooms[counter]
                counter += 1
                area = '_'.join(room.split('_')[:2])
                if area in areas:
                    self.data.append(torch.from_numpy(data_all[i]).float())         
                    self.labels.append(torch.from_numpy(label_all[i].squeeze()).long()) 

        assert counter == len(rooms), (
            f"Mismatch: expected {len(rooms)} room entries, but got {counter}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

###### Object: PointNet-KAN for semantic segmentation ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=SCALE):
        super(PointNetKAN, self).__init__()

        self.jacobikan4 = JacobiKANLayer(input_channels, int(128 * scaling), poly_degree)
        self.jacobikan5 = JacobiKANLayer(int(128 * scaling), int(1024 * scaling), poly_degree)

        self.jacobikan9 = JacobiKANLayer(int(1024 * scaling) + int(128 * scaling), int(128 * scaling), poly_degree)
        self.jacobikan10 = JacobiKANLayer(int(128 * scaling), output_channels, poly_degree)

        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))

    def forward(self, x):

        num_points = x.size(-1)

        x = self.jacobikan4(x)
        x = self.bn4(x)

        local_4 = x

        x = self.jacobikan5(x)
        x = self.bn5(x)

        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, num_points)

        x = torch.cat([local_4, global_feature], dim=1)

        x = self.jacobikan9(x)
        x = self.bn9(x)
        x = self.jacobikan10(x)

        return x

###### Function ######
def compute_mIoU_torch(preds, labels, num_classes):
    p = preds.view(-1)
    l = labels.view(-1)
    ious = []
    for cls in range(num_classes):
        pm = (p == cls)
        lm = (l == cls)
        inter = (pm & lm).sum().float()
        union = (pm | lm).sum().float()
        ious.append(inter/union if union > 0 else torch.tensor(1.0, device=p.device))
    return torch.stack(ious).mean().item()

###### Function ######
def train(train_areas):
    ds = S3DISHDF5Dataset(ALL_H5_LIST, ROOM_LIST_FILE, train_areas)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    model = PointNetKAN(9,NUM_CLASSES,SCALE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for ep in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        start_time = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            x = x.permute(0, 2, 1) 

            pred = model(x)  # [B, N, C]

            pred = pred.permute(0, 2, 1) 
            
            loss = criterion(
                pred.reshape(-1, NUM_CLASSES),
                y.reshape(-1)
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lbl = pred.argmax(dim=2)
            correct += (lbl == y).sum().item()
            total += y.numel()
            all_preds.append(lbl)
            all_labels.append(y)

        scheduler.step()

        acc = correct / total
        miou = compute_mIoU_torch(torch.cat(all_preds), torch.cat(all_labels), NUM_CLASSES)
        print(f"Epoch {ep}/{EPOCHS} | Time {time.time() - start_time:.1f}s | "
              f"Loss {running_loss:.4f} | Acc {acc:.4f} | mIoU {miou:.4f}")

    return model

###### Function ######
def evaluate(model, test_areas):
    ds = S3DISHDF5Dataset(ALL_H5_LIST, ROOM_LIST_FILE, test_areas)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.permute(0, 2, 1) 
            pred = model(x)
            pred = pred.permute(0, 2, 1) 
            lbl = pred.argmax(dim=2)
            correct += (lbl == y).sum().item()
            total += y.numel()
            all_preds.append(lbl)
            all_labels.append(y)

    acc = correct / total
    miou = compute_mIoU_torch(torch.cat(all_preds), torch.cat(all_labels), NUM_CLASSES)
    print(f"\nEvaluation ? Acc: {acc:.4f} | mIoU: {miou:.4f}")

###### Function ######
# -----------------------------
# Main: Train on Areas 1,2,3,4,6; Test on Area 5 [to use k-fold strategy, one should change the train_areas and test_areas]
# -----------------------------
if __name__ == '__main__':
    train_areas = {'Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'}
    test_areas = {'Area_5'}

    model = train(train_areas)
    torch.save(model.state_dict(), 'model.pth')
    evaluate(model, test_areas)
