# Deep-Learning-Model-Implementation-and-Training
CIFAR-10 Image Classification with Custom Deep CNN
A PyTorch implementation of a custom 6-layer deep Convolutional Neural Network (CNN) for 10-class image classification on the CIFAR-10 benchmark dataset. The architecture utilizes progressive spatial downsampling, Batch Normalization, layered Dropout regularization, and a Cosine Annealing learning rate schedule to achieve ~82.4% validation accuracy with minimal overfitting.
#Model Architecture Overview
Input (3x32x32)
   │
   ├── [Conv Block 1]  (Conv2D 3->32 -> BN -> ReLU) x 2 ➔ MaxPool(2x2) ➔ Dropout(0.2)
   ├── [Conv Block 2]  (Conv2D 32->64 -> BN -> ReLU) x 2 ➔ MaxPool(2x2) ➔ Dropout(0.3)
   ├── [Conv Block 3]  (Conv2D 64->128 -> BN -> ReLU) x 1 ➔ MaxPool(2x2) ➔ Dropout(0.4)
   │
   ├── Flatten (2048 units)
   ├── [Dense Layer]   Linear(2048 -> 512) ➔ BatchNorm1D ➔ ReLU ➔ Dropout(0.5)
   └── [Output Layer]  Linear(512 -> 10) ➔ Raw Logits
   #Per-Class Performance InsightsStrongest Categories ($>88\%$ Accuracy): Distinct macro-features allow high performance on Automobile, Ship, Truck, and Frog.Fine-Grained Challenges: Primary confusion occurs among animal categories with overlapping background and structural features (e.g., Cat vs. Dog, Bird vs. Deer).
   #Performance SummaryMetricTraining SetValidation SetTop-1 Accuracy~88.5%~82.4%Final Loss< 0.35~0.52Generalization Gap—~6.1%
