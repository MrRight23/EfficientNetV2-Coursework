import torch
from collections import Counter
from torchvision.datasets import CIFAR10
from sklearn.metrics import accuracy_score, f1_score

dataset = CIFAR10(
    root="./data",
    train=True,
    download=False,
)

generator = torch.Generator().manual_seed(42) #42 let us reproduce the same random numbers

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

#5000 images for the training and 1000 for validation
train_indices = indices[:5000]

val_indices = indices[5000:6000]

train_labels = [dataset.targets[i] for i in train_indices]
val_labels = [dataset.targets[i] for i in val_indices]

class_counts = Counter(train_labels)

majority_class = class_counts.most_common(1)[0][0]

print("The most common class is: ",dataset.classes[majority_class])

predictions = [majority_class] * len(val_labels)


#Evaluation metrics calculation 

accuracy = accuracy_score(val_labels, predictions)

macro_f1 = f1_score(
    val_labels,
    predictions,
    labels=list(range(10)),
    average="macro",
    zero_division=0
)

print(f"Baseline accuracy is: {accuracy * 100}%")
print(f"Baseline macro F1-score is: {macro_f1:.4f}")