# README: Implementation of GoogleNet in PyTorch

## Overview

This project provides a PyTorch implementation of **GoogleNet**, also known as **Inception v1**, based on the original research paper ["Going Deeper with Convolutions"](https://arxiv.org/abs/1409.4842) by Szegedy et al. GoogleNet introduced the Inception module, a novel architecture for efficiently utilizing computational resources while achieving state-of-the-art performance in image recognition tasks.

---

## Key Features

- **Custom Convolutional Block**: Simplifies layers with Conv2D, BatchNorm, and ReLU activation.
- **Inception Module**: Incorporates 1x1, 3x3, and 5x5 convolutions with dimensionality reduction.
- **Auxiliary Classifiers**: Improves gradient flow during training.
- **CIFAR-10 Dataset Integration**: Prepares, trains, and validates the model on the CIFAR-10 dataset.

---

## File Structure

- **GoogleNet.ipynb**: Main script containing the implementation, training pipeline, and testing code.
- **/content/drive/MyDrive**: Directory for storing dataset and model checkpoints.

---

## Implementation Details

### 1. **Convolutional Block**
Defines a reusable block comprising:
- Convolution layer (`Conv2d`)
- Batch Normalization (`BatchNorm2d`)
- ReLU activation

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out
```

---

### 2. **Inception Module**
Combines multiple convolution operations (1x1, 3x3, 5x5) and max pooling in parallel, concatenating their outputs. 

- Dimensionality reduction with 1x1 convolutions reduces computational complexity.
- Max pooling includes a 1x1 projection layer.

```python
class Inceptions(nn.Module):
    def __init__(self, in_channels, num1x1, num3x3_reduce, num3x3, num5x5_reduce, num5x5, pool_proj):
        super(Inceptions, self).__init__()
        self.block1 = ConvBlock(in_channels, num1x1, kernel_size=1, stride=1, padding=0)
        self.block2 = nn.Sequential(
            ConvBlock(in_channels, num3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num3x3_reduce, num3x3, kernel_size=3, stride=1, padding=1)
        )
        self.block3 = nn.Sequential(
            ConvBlock(in_channels, num5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num5x5_reduce, num5x5, kernel_size=5, stride=1, padding=2)
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        return torch.cat([out1, out2, out3, out4], 1)
```

---

### 3. **Auxiliary Classifier**
Used during training to provide auxiliary gradients, ensuring better optimization. It is later removed during inference.

```python
class Auxillary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxillary, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)
```

---

### 4. **GoogleNet Architecture**
Integrates the Convolutional Block, Inception modules, and Auxiliary Classifiers.

---

### 5. **Training the Model**
- **Loss Function**: Cross-Entropy Loss combines the main output and auxiliary outputs.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **DataLoader**: CIFAR-10 dataset is preprocessed and split into training, validation, and test sets.

---

## Usage Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Google Colab (optional, for cloud-based execution)

### Steps

1. Clone this repository or download the `.ipynb` file.
2. Upload the notebook to Google Colab or run it locally.
3. Mount Google Drive to store datasets and model checkpoints.
4. Execute all cells to:
   - Define the architecture
   - Load CIFAR-10 dataset
   - Train the model
   - Save and evaluate the trained model

---

## Results

### Test Accuracy
The model achieves **~80% accuracy** on CIFAR-10 after 15 epochs.

---

## References
- Szegedy, C., et al. "Going Deeper with Convolutions." [arXiv:1409.4842](https://arxiv.org/abs/1409.4842).
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

Feel free to reach out for questions or improvements!
