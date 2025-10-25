# 40-Daya-DL-MLOps
40-Day Deep Learning + MLOps Journey | Implementations, notes, and projects based on Hands-On Machine Learning (3rd Ed.). Covers neural nets, CNNs, transfer learning, and MLOps practices like versioning, experiment tracking &amp; deployment.

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

## Day 8 - Data Augmentation in CNNs

- Learned how to apply random image transformations to expand training data.  
- Techniques used: flipping, rotation, zoom, contrast.  
- Integrated augmentation inside CNN pipeline.  
- Dataset: CIFAR-10.  
- Achieved ~65% test accuracy after 10 epochs.  
- Note: Accuracy can improve significantly with deeper models or longer training.  

## Day 9 - Transfer Learning

- Learned how to reuse pretrained CNNs for new tasks.  
- Used MobileNetV2 pretrained on ImageNet as a feature extractor.  
- Added classifier head (Dense + Dropout + Softmax).  
- Dataset: CIFAR-10.  
- Achieved ~85–90% test accuracy in just 10 epochs.  

## Day 10 - Mini Project: Transfer Learning on Flowers Dataset

- Applied all previous learnings: CNNs, regularization, augmentation, transfer learning.  
- Used TensorFlow Flowers dataset with 5 classes.  
- Base model: MobileNetV2 pretrained on ImageNet.  
- Techniques: Data Augmentation + Dropout + EarlyStopping.  
- Achieved 92%+ test accuracy.  
- Perfect example of practical transfer learning for real-world data.  

## Day 11 - Introduction to Deep Architectures

- Learned how increasing network depth improves representation power.  
- Understood issues like vanishing gradients and their solutions (ReLU, BatchNorm, Residuals).  
- Compared shallow vs deep CNNs on CIFAR-10.  
- Deep model achieved ~80–85% accuracy vs ~70% for shallow.  

## Day 12 - ResNet Architecture

- Learned about residual connections and how they solve the vanishing gradient problem.  
- Implemented a simplified ResNet model on CIFAR-10 using Keras.  
- Understood skip connections, identity mapping, and bottleneck structure.  
- Compared ResNet with previous deep CNN — faster convergence and better generalization.  
- Achieved ~86–88% test accuracy after 20 epochs.  

## Day 13 - Inception & GoogLeNet

- Learned how Inception modules enable multi-scale feature extraction efficiently.  
- Implemented a simplified GoogLeNet (Inception v1) on CIFAR-10.  
- Understood dimensionality reduction using 1×1 convolutions.  
- Compared performance with ResNet — similar accuracy (~80%) but different architectural philosophy.  
- Achieved ~77–79% test accuracy after 25 epochs.  

## Day 14 - VGG16 / VGG19

- Learned how stacking small 3×3 convolutions builds deeper networks efficiently.  
- Implemented a VGG16-like model on CIFAR-10.  
- Observed increased depth → better generalization but higher computation.  
- Compared with Inception: VGG is simpler but slower to train.  
- Achieved ~89–91% accuracy after 30 epochs.  

## Day 15 - Vision Transformers (ViT)

- Learned how Transformers can process images by splitting them into patches.  
- Implemented a mini Vision Transformer from scratch for CIFAR-10.  
- Understood patch embedding, positional encoding, and self-attention.  
- Observed that ViT performs competitively even without convolution layers.  
- Achieved ~85–88% test accuracy after 30 epochs.  
