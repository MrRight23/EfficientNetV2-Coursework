This project investigates image classification using the EfficientNetV2-S architecture. This is based on the research paper EfficientNetv2 by Tan and le 2021.

The implementation for this uses an ImageNet pretrained EfficientNetV2 model, and replaces its final classification layer to recognise the 10 CIFAR-10 classes. The feature extraction layers are frozen while training the new classifier. 

The experiment compares fixed image resolutions, progressive image resizing and different data augmentation. 




REQUIREMENTS to run the EfficientNetV2 code:

Download the following libraries with the right versions. 

PyTorch -> 2.15.0
TorchVision -> 0.29.2
scikit-learn -> 1.9.1
NumPy -> 2.5.4


---DATASET---

The project uses the CIFAR-10 dataset. use this link:
https://cave.cs.toronto.edu/kriz/cifar.html

The dataset contains 60,000 images across 10 classes, with 50,000 official training images and 10,000 test images. 

To install the dataset open the terminal and do:

python3 -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', train=True, download=True); CIFAR10(root='./data',train=False, download=True)"

change python3 to .venv/bin/python if using the virtual environment.




The EfficientNetV2 paper evaluates transfer learning on CIFAR-10 among the other datasets, so i used it as a small and fast benchmark to do my progressive training. 

For the training and validation, i selected 5000 training images and 1000 validation images from the CIFAR-10 training partition while using random seed 42. 

The final evaluation uses all the 10,000 images from the seperate CIFAR-10 test partition. 

The dataset is stored inside of the data directory which is inside of the main directory. 

The experiments directory is what contains the training scripts. Baseline and model evaluation can be found inside of the evaluation directory. 

Saved model checkpoints can be found inside of the results directory. 


---The project includes six EfficientNetV2-S experiments---


-Experiment A - One epoch with a fixed 160x160 training resolution. - my_training.py

-Experiment B - One epoch with a fixed 224x224 training resolution. - my_training224.py

-Experiment C - Three epoch with a fixed 224x224 training resolution. - my_training_3_epochs.py

-Experiment D - Three epochs that uses progressive training resolution that goes from 128, 160 and 224. - progressive_training.py

-Experiment E - Progressive resizing with an increasing RandAugment magnitude. - adaptive_training.py

-Experiment F - Progressive resizing with a constant RandAugment magnitude. - same_augmentation.py

For all the experiments, validaton images remain the same - 224x224 resolution.


---BASELINE EVALUATION---

Two classification baselines are included. 

To run the majority class baseline run the baseline.py file inside of the evaluation directory.

that would be: implementation/evaluation/baseline.py

To run the 3 nearest neighbours baseline run the knn_baseline.py file that is inside of the evaluation directory.

implementation/evaluation/knn_baseline.py


---MODEL EVALUATION---

Experiment D uses the saved checkpoint - progressive_training.pth

this can be found inside of the results directory. 

results/progressive_training.pth

If you want to evaluate this checkpoint on the official CIFAR-10 test set, then run the test model. - implementation/evaluation/testing_model.py.


The scripts calculates classification accuracy, the macro F1-score and also a confusion matrix. 

The implementation/evaulation/evaluate_models.py script evaluates the saved model checkpoints.

To evaluate a different experiment, simply change the checkpoint path inside of the evaluate_models.py.

By doing that we can load the already trained checkpoint and calculate the evaluation metrics without training the model again.

Make sure to run all of the commands from the Main directory - EfficientNetV2-Coursework directory. 


Validation and Final Test Results:

Experiment B, the one-epoch fixed resolution model managed to achieve:

91.20% validation accuracy with validation images of 1000 and a Macro F1-score of 0.9117.


Experiment D, the progressive resizing model managed to achieve:

90.90% validation accuracy and a macro F1-score of 0.9087 on the same 1000 validation images.

---Final Test Results---

Experiment D managed to achieve:

91.60% test accuracy with test images of 10,000 and a Macro F1-score of 0.9162. 
 
 Experiment B was not evaluated on the official test set. its reported results are from the validation set. 


---Saved Model Checkpoints---

The six trained model checkpoints are available in the GitHub Release section under the v1.0.

Download the result zip and extract into the main project directory. 

The evaluation scripts load these saved checkpoints in order to calculate validation accuracy, macro F1-score and confusion matrices without having to retrain the mdoels.


 



If you want a link to the paper itself then it is this: 

https://proceedings.mlr.press/v139/tan21a.html
