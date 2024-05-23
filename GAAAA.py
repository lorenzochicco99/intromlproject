# qui provo il dropout neuron


# robe da impostare 
file_name = 'cub_5e_dn05_vit.pth' # deve essere di sto formato qua !!!
freeze_layers_except_last = False
num_epochs = 5
data_root = "intromlproject/cub"
dn = 0.5



# misuriamo il tempo
import time

start_time = time.time()

# Funzioni rubate ihihi
from intromlproject.utils.read_dataset import read_dataset, transform_dataset, get_data_loader

train_data, val_data, test_data = read_dataset(data_root, transform_dataset())
train_loader, val_loader, test_loader = get_data_loader(train_data, val_data, test_data, batch_size=32)

dataset_sizes = {
    'train': len(train_loader.dataset),
    'val': len(val_loader.dataset),
    'test': len(test_loader.dataset)
}

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

class_names = train_data.classes
print("Nomi delle classi: ", class_names)
num_classes = len(class_names)
print("Numero di classi: ", num_classes)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import copy


model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True)


class ModifiedViT(nn.Module):
    def __init__(self, original_model, num_classes, dropout_prob=0.5):
        super(ModifiedViT, self).__init__()
        self.features = nn.Sequential(
            original_model.patch_embed,
            original_model.pos_drop,
            *original_model.blocks,
            original_model.norm
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(original_model.head.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x[:, 0]
        x = self.dropout(x)
        x = self.head(x)
        return x


model = ModifiedViT(model, num_classes, dropout_prob=dn)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, criterion, optimizer, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer, num_epochs=num_epochs)


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

    return total_loss, total_acc

test_loss, test_acc = evaluate_model(model_ft, dataloaders['test'], criterion)


torch.save(model_ft.state_dict(), file_name)

elapsed_time = time.time() - start_time


print(f"Code executed in {elapsed_time} seconds.")