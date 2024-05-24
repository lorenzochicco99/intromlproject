import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from utils.logger import *
from utils.models_init import *
from utils.training import *
from utils.optimizers import *

# Configuration

"""
Image Sizes:
    - AlexNet: 224x224
    - DenseNet: 224x224
    - Inception: 299x299
    - ResNet: 224x224
    - VGG: 224x224

Mean, Std, number of classes for Datasets:
    - CUB-200-2011: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=200
    - Stanford Dogs: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=120
    - FGVC Aircraft: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
    - Flowers102: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
"""
config = {
    # Path and directory stuff
    'data_dir': '/home/disi/machinelearning/datasets',  # Directory containing the dataset
    'dataset_name' : 'aerei', # Name of the dataset you are using, doesn't need to match the real name, just a word to distinguish it
    # leave checkpoint = None if you don't have one
    'checkpoint': '/home/disi/machinelearning/checkpoints/alexnet/alexnet_aerei_epoch2.pth',  # Path to a checkpoint file to resume training
    'save_dir': '/home/disi/machinelearning/checkpoints/alexnet',  # Directory to save logs and model checkpoints
    'project_name': 'alexnet_test',  # Weights and Biases project name
    
    
    # Image transformation 
    'image_size': 224,  # Size of the input images (default: 224)
    'num_classes': 102,  # Number of classes in the dataset
    'mean': [0.485, 0.456, 0.406],  # Mean for normalization
    'std': [0.229, 0.224, 0.225],  # Standard deviation for normalization

    # Training loop
    'model_name': 'alexnet',  # Name of the model to use
    'batch_size': 32,  # Batch size (default: 32)
    'epochs': 20,  # Number of epochs to train (default: 10)
    'optimizer': 'Adam',  # Optimizer to use (default: Adam)
    'optimizer_type': 'simple',  # Type of optimizer to use (default: simple)
    'learning_rate': 0.001,  # Learning rate (default: 0.001)
    'weight_decay': 0,  # Weight decay for optimizer (default: 0)
    'momentum': 0,  # Momentum for optimizer (default: 0)
    'criterion': 'CrossEntropyLoss',  # Criterion for the loss function (default: CrossEntropyLoss)

    #Irrelevant
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for training
}

### NON MODIFICAR EQUA SOTTO O MARTINA SI INCAZZA (MI STA PUNTANDO U NCOLTELLO ALLA GOLA)

def main(config):
    # Initialize wandb
    wandb.init(project=config['project_name'])

    # Setup logger
    logger = setup_logger(log_dir=config['save_dir'])

    # Initialize the model
    model = init_model(config['model_name'], config['num_classes'])
    model.to(config['device'])

    # Define the optimizer
    if config['optimizer_type'] == 'custom':
        optimizer = create_custom_optimizer(model.parameters(), {
            'type': config['optimizer'],
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'momentum': config['momentum']
        })
    else:
        optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])

    # Define the loss function
    criterion = getattr(torch.nn, config['criterion'])()

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Load checkpoint if specified
    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            cost_function=criterion,
            epoch=epoch,
            model_name=config['model_name'],
            dataset_name=config['dataset_name'],
            save_dir=config['save_dir'],
            device=config['device']
        )
        print(f"Epoch {epoch} completed. Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

if __name__ == "__main__":
    main(config)