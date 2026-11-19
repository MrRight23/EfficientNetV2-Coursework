import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

#42 seed to reproduce the same random numbers for a fair comparison. 
torch.manual_seed(42)

# This is for my mac so it will use the GPU if available.
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)




transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
])

dataset = CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)


model = efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.DEFAULT
    )

for parameter in model.features.parameters():
    parameter.requires_grad = False

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10)

model = model.to(device)
print("Using device:", device)


generator = torch.Generator().manual_seed(42) #42 let us reproduce the same random numbers

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

#5000 images for the training and 1000 for validation
train_indices = indices[:5000]

val_indices = indices[5000:6000]



#subset is for the dataset that we split
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))



#This is the data loader part to process the images into smaller batches

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


criterion = nn.CrossEntropyLoss() 
#measure how well the prediction matches the target labels.

#tells adam to only update the new final layer
optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)
#print("Loss function:", criterion)
#print("Optimizer:", optimizer)




model.train()
#we want to keep the model features in evaluation mode. 
#This is so stored avereage that is learned from imagenet is not updated with the CIFAR10 dataset.
model.features.eval()

#performance variables to see how the model performed during training.
total_loss = 0.0
correct_predictions = 0
total_images = 0

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * labels.size(0)
    predictions = outputs.argmax(dim=1)

    correct_predictions += (predictions == labels).sum().item()
    total_images += labels.size(0)

#results calculation after the batches is processed.
average_loss = total_loss / total_images
training_accuracy = correct_predictions / total_images



print("Average training loss:", round(average_loss, 4))
print("Accuracy of training set:", training_accuracy * 100, "%")




model.eval()
validation_correct = 0
validation_total = 0

#validation loop and torch.inference_mode() to disable gradient calculations 
# because we are only making predictions. 
with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        validation_correct += (predictions == labels).sum().item()
        validation_total += labels.size(0)

validation_accuracy = validation_correct / validation_total
print("Accuracy of validation set:", validation_accuracy * 100, "%")


from pathlib import Path

Path("results").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "results/trained_model_160.pth"
)
print("Model saved to results/trained_model_160.pth")