# MNIST Digit Classification using MLP + Optimizer Comparison
This project focuses on building a **Multilayer Perceptron (MLP)** neural network model to classify handwritten digits from the **MNIST dataset**, while comparing different **training optimization techniques** including variants of **Stochastic Gradient Descent (SGD)**, **Adam**, and **RMSProp**.
## Objectives
- Load and preprocess MNIST dataset
- Build an MLP classifier using TensorFlow/Keras
- Train using multiple optimization strategies
- Compare:
  - Training and validation performance
  - Convergence speed
  - Execution time
    - Select the best model and evaluate on test data
    - Visualize learning curves + prediction samples
## Dataset
**MNIST** : 70,000 grayscale handwritten digits\
**Images** : 28×28 resolution\
**Training** : 60,000\
**Testing** : 10,000

## MLP Architecture
| Layer | Type                 | Units | Activation |
| ----- | -------------------- | ----- | ---------- |
| 1     | Dense (input layer)  | 128   | ReLU       |
| 2     | Dense (hidden layer) | 64    | ReLU       |
| 3     | Dense (output layer) | 10    | Softmax    |

- Loss Function : **Sparse Categorical Crossentropy**
- Evaluation Metric : **Accuracy**

## Compared Optimization Strategies
| Optimizer       | Batch Size | Epochs | Extra Parameters |
| --------------- | ---------- | ------ | ---------------- |
| SGD             | 1          | 5      | -                |
| Mini-Batch SGD  | 64         | 50     | -                |
| Full-Batch SGD  | Full set   | 50     | -                |
| SGD w/ Decay    | 64         | 50     | Decay = 1e-6     |
| SGD w/ Momentum | 64         | 50     | Momentum = 0.9   |
| Adam            | 64         | 50     | lr = 0.001       |
| RMSProp         | 64         | 50     | lr = 0.001       |

## Model Selection
After training, all models were evaluated on validation set and the best performer was automatically selected and saved as:
