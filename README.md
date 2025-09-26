# 40-Daya-DL-MLOps
🚀 40-Day Deep Learning + MLOps Journey | Implementations, notes, and projects based on Hands-On Machine Learning (3rd Ed.). Covers neural nets, CNNs, transfer learning, and MLOps practices like versioning, experiment tracking &amp; deployment.

## Day 1 - First Neural Network (MNIST)

- Built a simple fully connected neural network using TensorFlow & Keras.
- Dataset: MNIST (handwritten digits).
- Model: Flatten → Dense(128, ReLU) → Dense(10, Softmax).
- Achieved ~97% test accuracy.

## Day 2 — Activation Functions
- Compared Sigmoid vs ReLU on MNIST.
- ReLU trained faster and achieved higher accuracy.
- Learned why activation functions introduce non-linearity and prevent collapse into a linear model.

## Day 3 - Loss Functions & Optimizers
- Learned about common loss functions (MSE, Cross-Entropy).  
- Studied optimizers: SGD, Momentum, RMSProp, Adam.  
- Implemented comparison on MNIST.  
- Adam achieved the best test accuracy (~98%).  

## Day 4 - Overfitting & Regularization

- Studied overfitting and techniques to improve generalization.  
- Implemented L2 regularization, Dropout, and Early Stopping.  
- Dataset: MNIST.  
- Achieved ~98% accuracy with reduced overfitting.  

## Day 5 - Intro to CNNs

- Learned about convolutional and pooling layers.  
- Implemented first CNN on CIFAR-10 dataset.  
- Model: Conv → Pool → Conv → Pool → Dense.  
- Achieved ~70% test accuracy after 5 epochs.  

## Day 6 - CNN on CIFAR-10 (Deeper Architectures)

- Explored evolution of CNN architectures: LeNet, AlexNet, GoogLeNet, ResNet.  
- Implemented a deeper CNN with Conv → Conv → Pool → Dropout blocks.  
- Dataset: CIFAR-10.  
- Achieved ~80% test accuracy after 15 epochs.  

## Day 7 - Batch Normalization & Dropout in CNNs

- Studied how BatchNorm stabilizes training and Dropout prevents overfitting.  
- Implemented CNN with Conv → BN → ReLU + Dropout in Dense layers.  
- Dataset: CIFAR-10.  
- Achieved ~82% test accuracy with improved generalization.  
