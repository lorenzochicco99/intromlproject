# Metti il nome del modello che vai a salvare qualcosa di sto tipo pls
model_name = 'models/cub_uuuaaa_2.pth'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import time
from torch.optim.lr_scheduler import StepLR
from torchvision.models.resnet import Bottleneck, conv1x1

# devo ancora capire bene sta rete ma 
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class SEResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(SEResNet50, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # Load pre-trained ResNet-50
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # Modify the fully connected layer to match the number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1, reduction=8):  # Adjusted reduction ratio
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # scegli tra le due
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction))  # Adjusted reduction ratio
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction))  # Adjusted reduction ratio

        return nn.Sequential(*layers)




def get_data_loaders(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda', model_name='best_model.pth'):

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print("\n", '-'*10)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr}')     
        e_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.sampler)
        train_acc = evaluate_model(model, train_loader, device)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print('Epoch time: ', time.time() - e_start)
        
        # Check for best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)

        scheduler.step()
        print('-'*10, "\n")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break
    
    print('Training complete. Best validation accuracy: {:.4f}'.format(best_val_acc))



def evaluate_model(model, data_loader, criterion=None, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    if criterion:
        avg_loss = running_loss / len(data_loader.sampler)
        return avg_loss, accuracy
    return accuracy



data_dir = 'Images'  
batch_size = 64
num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
num_epochs = 10
dropout = nn.Dropout(p=0.5)


train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size)

print("Data loader done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEResNet50(num_classes=num_classes).to(device)

print("Pre trained model loaded")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

print("Start training")
start = time.time()

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, dropout, num_epochs, device)

print("End training, time: ", time.time()-start)

test_accuracy = evaluate_model(model, test_loader, device)
print('Test Accuracy: {:.4f}'.format(test_accuracy))

model.load_state_dict(torch.load(model_name))