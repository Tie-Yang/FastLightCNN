import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
from thop import profile

# ====================== 1. Configuration ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
total_epochs = 10
initial_lr = 0.001

save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====================== 2. Model Definition ======================
class LightChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FastDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw_pw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.attention = LightChannelAttention(out_channels)

    def forward(self, x):
        x = self.dw_pw(x)
        x = self.attention(x)
        return x

class FastLightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block1 = FastDepthwiseSeparableConv(8, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ====================== 3. Training & Testing ======================
def train_one_epoch(model, optimizer, criterion, epoch, model_name):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            progress = 100.0 * batch_idx / len(train_loader)
            print(f"[{model_name}] Epoch {epoch:2d}/{total_epochs} | Batch {batch_idx:3d}/{len(train_loader)} ({progress:5.1f}%) | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"[{model_name}] Epoch {epoch:2d} Train Done | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    return avg_loss

def test(model, model_name, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    print(f"[{model_name}] Epoch {epoch:2d} Test | Accuracy: {acc:.2f}%")
    return acc

# ====================== 4. Metrics Calculation ======================
def get_metrics(model, model_name, best_acc):
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(1, 1, 28, 28).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    flops_m = flops / 1e6

    model_jit = torch.jit.script(model)
    with torch.no_grad():
        for _ in range(20):
            model_jit(dummy)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(200):
            model_jit(dummy)
    t_avg = (time.time() - t0) / 200 * 1000

    return {
        "model": model_name,
        "acc": f"{best_acc:.2f}%",
        "params": params,
        "flops": f"{flops_m:.2f}M",
        "time": f"{t_avg:.2f}ms"
    }

# ====================== 5. Main Program ======================
if __name__ == "__main__":

    def run_experiment(model_class, name):
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        best_acc = 0.0
      
        best_model_path = os.path.join(save_dir, f"best_{name}.pth")

        print(f"\n" + "="*60 + f" Training {name} " + "="*60)

        for epoch in range(1, total_epochs + 1):
            train_one_epoch(model, optimizer, criterion, epoch, name)
            acc = test(model, name, epoch)
            
            if acc > best_acc:
                best_acc = acc
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, best_model_path)
                print(f"[{name}] Saved best model to {best_model_path} (Acc: {best_acc:.2f}%)")
            
            scheduler.step()
            print("-" * 85)

        print(f"\n[{name}] Best Accuracy: {best_acc:.2f}%")
        return model, best_acc

    # Model training
    light_model, light_best = run_experiment(FastLightCNN, "FastLightCNN")
    lenet_model, lenet_best = run_experiment(LeNet5, "LeNet-5")

    # Metrics calculation
    m1 = get_metrics(light_model, "FastLightCNN", light_best)
    m2 = get_metrics(lenet_model, "LeNet-5", lenet_best)

    # Print results
    print("\n" + "="*90)
    print("          Model | Accuracy | Params | FLOPs  | Inference Time")
    print("-"*90)
    print(f"{m1['model']:>16} | {m1['acc']:>8} | {m1['params']:>6,} | {m1['flops']:>6} | {m1['time']:>15}")
    print(f"{m2['model']:>16} | {m2['acc']:>8} | {m2['params']:>6,} | {m2['flops']:>6} | {m2['time']:>15}")
    print("="*90)