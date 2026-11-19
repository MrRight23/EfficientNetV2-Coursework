import torch
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.neighbors import KNeighborsClassifier
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

#Converting images to numbers and reshaping it
#  because the model expects a flat table. 
x_train = dataset.data[train_indices].reshape(5000,-1)
x_val = dataset.data[val_indices].reshape(1000, -1)

x_train = x_train.astype(np.float32) / 255
x_val = x_val.astype(np.float32) / 255

y_train = [dataset.targets[i] for i in train_indices]
y_val = [dataset.targets[i] for i in val_indices]

print("Training shape:", x_train.shape)
print("Validation shape:", x_val.shape)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

predictions = knn.predict(x_val)


accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    labels=list(range(10)),
    average="macro",
    zero_division=0
)


print(f"The k-NN accuracy is: {accuracy * 100}%")
print(f"The k-NN macro F1-score is: {macro_f1:4f}")