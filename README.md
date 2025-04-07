# Neural Network from Scratch for MNIST Digit Classification

## Overview
This project implements a two-layer neural network from scratch in Python to classify handwritten digits from the MNIST dataset. The goal is to demonstrate a foundational understanding of neural network concepts, including forward and backward propagation, gradient descent, and activation functions, without relying on high-level libraries like TensorFlow or PyTorch.

## Key Features
- **Architecture**: 
  - Input layer: 784 pixels (28x28 MNIST images).
  - Hidden layer: 10 neurons with ReLU or sigmoid activation.
  - Output layer: 10 neurons (one per digit) with softmax activation.
- **Concepts Demonstrated**:
  - Forward propagation with matrix operations.
  - Backward propagation for gradient computation.
  - Gradient descent optimization.
  - Activation functions: ReLU, sigmoid, and softmax.
  - Cross-entropy loss calculation.
- **Evaluation**: Tracks accuracy and loss during training.
- **Prediction**: Generates digit predictions for test images with visualization.

## Dataset
- **MNIST**: 
  - Training set: 42,000 labeled images (`data/train.csv`).
  - Test set: 28,000 unlabeled images (`data/test.csv`).
- Data is preprocessed by normalizing pixel values (divided by 255).
- **Source**: Download `train.csv` and `test.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data) and place them in the `data/` folder in this repository.

## Requirements
- Python 3.x
- Libraries:
  - `numpy` (for matrix operations)
  - `pandas` (for data loading)
  - `matplotlib` (for visualization)

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sprobulski/neural-network-from-scratch
   cd neural-network-from-scratch

2. **Get Data**:
Download train.csv and test.csv from Kaggle and place them in the data/ folder.

3. **Output**:
Training prints accuracy and loss every 50 iterations.
Test predictions are saved to a DataFrame and visualized for a sample image.

## Results
- After 1000 iterations with a learning rate of 0.1 and ReLu activation:
  - Training accuracy: ~85.62%.
  - Cross-entropy loss: ~0.4795.
- Note: The focus is on understanding, not optimizing for high accuracy.

## Files
- `neural_network_scratch.ipynb`: Main Jupyter notebook with all code.
- `data/` folder: Contains a placeholder. Add `train.csv` and `test.csv` from Kaggle here.

## Limitations
- Simple architecture (two layers, 10 neurons each) limits performance.
- Full-batch gradient descent (no mini-batching).
- No hyperparameter tuning or regularization.
