# Image Classification with K-NN, NCC, and MLP on CIFAR-10

This repository contains the implementation and comparison of various machine learning and deep learning algorithms for image classification, using the popular **CIFAR-10** dataset.

The project is divided into two main parts:

1. The implementation "from scratch" of the **K-Nearest Neighbors (K-NN)** and **Nearest Centroid Classifier (NCC)** algorithms.
2. The implementation, training, and optimization of a **Multilayer Perceptron (MLP)** (Feedforward Neural Network).

## 🎯 Project Purpose

The main goal of this project is to evaluate and compare the performances of classification algorithms, specifically studying:

* The accuracy and execution time (training & prediction time) between traditional algorithms (K-NN, NCC).
* The impact of various hyperparameters on the performance of a neural network (MLP), analyzing Underfitting/Overfitting phenomena, and finding the optimal model that achieves good generalization (Good Fitting).

## 📊 Dataset

The project uses the [**CIFAR-10**](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

* **Size:** 60,000 color images with a resolution of 32x32 pixels.
* **Split:** 50,000 images for training (Training set) and 10,000 images for testing (Test set).
* **Classes:** 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

The dataset is downloaded and prepared automatically (using standard scaling in the first part and `torchvision.transforms` in the second).

## 🛠️ Technologies & Libraries

The project was developed in a **Python** environment (using Google Colab with a T4 GPU is highly recommended for faster execution of the MLP).

**Core Libraries:**

* `numpy`: For fast array computations (K-NN and NCC implementation).
* `torch`, `torch.nn`, `torch.optim`: For building and training the MLP (PyTorch).
* `torchvision`: For loading the CIFAR-10 dataset and applying data augmentation.
* `matplotlib.pyplot`: For visualizing results (loss/accuracy curves).
* `time`, `pickle`, `urllib`, `tarfile`: Utilities for measuring execution time and managing files.

## 🗂️ Project Structure

### Part 1: K-NN & NCC (From Scratch)

* Implementation of `KNNClassifier` (with support for choosing `k`) and `NCCClassifier` classes.
* Implementation of utility functions: Feature Standardization, Confusion Matrix calculation, and performance metrics (Accuracy, Precision, Recall, F1-Score).
* **Key Findings:** K-NN ($k=1, k=3$) achieves better accuracy (~35.6%) compared to NCC (~28.1%), but it is significantly slower during prediction (it requires calculating distances across the entire training set, unlike the 10 centroids used in NCC).

### Part 2: Feedforward Neural Network (MLP)

* Implementation of an `MLP` class (inheriting from PyTorch's `nn.Module`) with a dynamic architecture.
* **Experimentation Phases:**
  1. **Architecture (Simple Models):** Evaluating small networks with 1-2 hidden layers (underfitting was observed, indicating the need for deeper networks).
  2. **Regularization:** Using Dropout and Weight Decay (L2) to manage overfitting.
  3. **Activation Functions:** Comparing ReLU, Leaky ReLU, Tanh, and Sigmoid (ReLU and Leaky ReLU yielded the best results).
  4. **Optimizers:** Comparing Adam, SGD (with momentum), and RMSprop (RMSprop struggled in this setup, while SGD+Momentum and Adam provided the best results).
  5. **Road to Optimized Model:** Combining the best hyperparameters (e.g., [1024, 512, 256] neurons, ReLU, SGD with Momentum, low weight decay) to achieve ~55-60% accuracy with good fitting.

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have the necessary libraries installed (`pip install numpy torch torchvision matplotlib`).
3. It is recommended to run the code (`.ipynb` file or Jupyter Notebook) on [Google Colab](https://colab.research.google.com/) with the **Hardware Accelerator: GPU (T4)** enabled, so the neural network training (Part 2) completes quickly.
4. Execute the cells sequentially. The CIFAR-10 dataset will be downloaded automatically.

## 📈 Evaluation & Insights (Per Class Analysis)

The project includes detailed reporting for per-class accuracy, as well as the generation of correct and incorrect prediction examples. This provides insights into which images frequently confuse the algorithms (e.g., misclassifying a cat as a dog, or an automobile as a truck). Additionally, detailed Loss and Accuracy curves are generated automatically to visualize the training process.

Check out the next part: [Part 2: SVM & Hinge-Loss MLP](https://github.com/BB-lab24/svm-hinge_mlp-knn-ncc-cifar10)
