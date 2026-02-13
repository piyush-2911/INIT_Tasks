# Task 3: Neural Networks on CIFAR-10

## Overview

This task implements and compares Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for image classification on CIFAR-10 dataset.

---

## 1. Loss Function: Cross Entropy Loss

The model uses cross-entropy loss as the values are normalized to 0-1, this loss function will punish the more confidently wrong predictions more severely in this range.

$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where $C$ = number of classes, $y_i$ = ground truth (one-hot), $\hat{y}_i$ = predicted probability

**Softmax:**

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

## 2. Gradient Descent Update Rule

The optimization starts with basic gradient descent where the model walks down the loss hill by following the steepest slope at each step with no memory. This works but can be slow and get stuck in small dips. To fix this, momentum is added so the model remember which direction it was going in and keep that momentum, making it faster and able to roll past obstacles. 

**Basic SGD (Part 1):**
$$\theta_t = \theta_{t-1} - \alpha \cdot \nabla L(\theta_{t-1})$$

**SGD with Momentum (Parts 2-3):**
$$v_t = \beta \cdot v_{t-1} + \nabla L(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} - \alpha \cdot v_t$$

## 3. ANN vs CNN Accuracy Comparison

| Model | Architecture | Test Accuracy | Epochs |
|-------|-------------|---------------|--------|
| **ANN** | 3072 → 512 → 256 → 10 | ~40-45% | 5 |
| **CNN** | Conv(3→12) → Conv(12→24) → FC | **~65-72%** | 30 |

---

## Key Findings

 **ANN Inputs**: we flatten the images into a vector of singular dimension as Dense layers require a 1D input.

**Dropout**: It helped the model generalize better. Even if the traing accuracy decreases, it does improve the validation accuracy.

**Gradient Descent**: Basic SGD walks the loss landscape directly, while momentum-based SGD remembers direction and converges faster by rolling past obstacles.

**CNN vs ANN**: CNNs dramatically outperform ANNs (65-72% vs 40-45%) by capturing spatial image structure with convolutional filters using 50x fewer parameters.

