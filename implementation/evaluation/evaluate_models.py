import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
val_data = Subset( dataset, val_indices)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("number of evaluation batches:", len(val_loader))


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else torch.device("cpu")
)

#weight is none because we are loading the weights that i already saved. 
model = efficientnet_v2_s(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10)

model.load_state_dict(
    torch.load(
        "results/trained_model_224.pth",
        map_location="cpu",
        weights_only=True
    )
)


 

model = model.to(device)
model.eval()

print("Using device:", device)
print("model loaded successfully.")


all_predictions = []
all_labels = []

with torch.inference_mode():
    for images, labels in val_loader:

        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


accuracy = accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

print(f"Validation accuracy: {accuracy *100}%")
print(f"Macro F1-score: {macro_f1:.4f}")

matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion matrix: ")
print(matrix)




