#Imports

#SSL Fix for macOS
import ssl

#Pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

#Utility
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split

#Dataset Loading
dataset_path = Path(__file__).parent / 'plant_dataset' / 'segmented' #creates the path to the dataset
if dataset_path.exists():
    print(f"Dataset found at: {dataset_path}")
else:
    print(f"Dataset not found at: {dataset_path}")

print("Dataset Structure")

classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
print(f"Number of classes: {len(classes)}")
print(f"Classes: {classes}")

print("Images per Class")
for class_name in classes:
    class_path = dataset_path / class_name
    image_files = list(class_path.rglob("*.jpg")) + list(class_path.rglob("*.jpng")) + \
                    list(class_path.rglob("*.png")) + list(class_path.rglob("*.JPG"))
    print(f"{class_name}: {len(image_files)} images")

print("Sample Image Info")
if classes:
    first_class = classes[0]
    sample_images = list((dataset_path / first_class).rglob('*.jpg'))
    if sample_images:
        sample_img = Image.open(sample_images[0])
        print(f"Sample image path: {sample_images[0].name}")
        print(f"Sample image size: {sample_img.size}")
        print(f"Sample image mode: {sample_img.mode}")
    else:
        print("No .jpg images found, trying other formats...")

#Data Transformations
train_transforms = transforms.Compose([ #pipeline of transformations
    transforms.Resize((224, 224)), #resize image to 224x224 because standard for ResNet
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10), #rotates max 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #chance to chang image color
    transforms.ToTensor(), #makes the image a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalize?
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Divideding Dataset into Train, Validation, and Test Sets
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms) #gets images with transformations
train_size = int(0.7 * len(full_dataset)) #divides images into 0.7 train, 0.15 val, 0.15 test
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(67)
)

print(f"Training set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")
print(f"Test set: {len(test_dataset)} images")

#Data Loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Model Setup
ssl._create_default_https_context = ssl._create_unverified_context
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #loads ResNet18 with pretrained weights 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of trainable parameters: {trainable_params}")
print(f"Total number of parameters: {total_params:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #checks for GPU
model = model.to(device) #moves model to GPU

#Training

criterion = nn.CrossEntropyLoss() #loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam optimizer, automatically adjusts learning rate

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader: #iterates through the batches
        images, labels = images.to(device), labels.to(device) #move data to GPU

        outputs = model(images) #forward pass = gets predictions
        loss = criterion(outputs, labels) #compute loss 
        optimizer.zero_grad() #clears the gradients PyTorch accumulates
        loss.backward() #backward pass = calculates gradients to adjust weights
        optimizer.step() #updates weights, where the learning actually happens

        train_loss += loss.item() #.item() turns the tensor into a number 
        _, predicted = torch.max(outputs.data, 1) #gets the class with highest score, the _ gets the index
        train_total += labels.size(0) 
        train_correct += (predicted == labels).sum().item()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): #turns off gradients since weights dont need to be updated and it makes it faster
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
    print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')

torch.save(model.state_dict(), 'plant_classifier.pth')