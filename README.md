BINARY CLASSIFICATION OF CIFAR-10 IMAGES
1. Introduction
This report presents the development and evaluation of three different Machine Learning Algorithms for a binary
classification task on the CIFAR-10 dataset. The goal is to classify images into two categories: vehicles (labels 0, 1, 8,
9) and animals (all other labels). Each model applies different preprocessing techniques and neural network
architectures to optimize accuracy.
2. Dataset Description
● Dataset: CIFAR-10
● Images: 60,000 color images of size 32×32 pixels, 10 classes
● Classes for binary classification: Vehicles (airplane, automobile, ship, truck) vs Animals (others)
● Training samples: 50,000
● Test samples: 10,000
3. Preprocessing Steps Common to All Models
● Label transformation: Converted multi-class labels to binary (vehicles = 1, animals = 0) using NumPy’s isin.
● Normalization: Images scaled to [0, 1] by dividing pixel values by 255.
● Flattening: 32×32×3 images reshaped to 1D vectors of length 3072.
● Train-validation split: For model training, data was split into training and validation sets using stratified
splitting to maintain class distribution.
● Dimensionality Reduction (PCA): Applied PCA for dimensionality reduction on CIFAR-10’s
3072-dimensional colored images, reducing features to 100 componens to lower complexity and enhance the
performance of shallow algorithms.
4. Model Architectures and Methodologies
To evaluate and compare different learning approaches, we trained a total of nine models across three categories: a
non-parametric model (K-Nearest Neighbors), a parametric model (Logistic Regression), and an artificial neural
network (ANN). For each category, three variations were implemented to observe how performance changes with
different training strategies and evaluation techniques. The following sections provide detailed descriptions and
results for each model and its respective variations.
4.1 Parametric Algorithm - K Nearest Neighbours (KNN)
Using K-Nearest Neighbors (KNN) for the CIFAR-10 image dataset presents both strengths and limitations. KNN is a
simple, instance-based learning algorithm that works well for small datasets and low-dimensional data. However,
CIFAR-10 images are high-dimensional (3072 features per image), which makes KNN computationally expensive
and sensitive to the "curse of dimensionality." To address this, we applied dimensionality reduction using PCA, which
significantly improves efficiency and classification performance. While KNN can capture patterns in the data without
training a model in the traditional sense, it lacks the representational power of neural networks and doesn't scale well
to large datasets. Nevertheless, when optimized carefully (e.g., through hyperparameter tuning, train-test split
evaluation, and cross-validation), KNN can still offer competitive baseline results for binary classification tasks like
