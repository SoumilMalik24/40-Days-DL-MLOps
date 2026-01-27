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

## Day 16 - RNN Fundamentals

- Learned how RNNs handle sequential data through hidden states.  
- Understood forward pass, recurrence, and vanishing gradient issue.  
- Implemented a Simple RNN for sentiment analysis on IMDB dataset.  
- Compared sequence learning with previous CNN-based architectures.  
- Achieved ~83–85% accuracy after 10 epochs.  

## Day 17 - LSTM & GRU

- Understood how LSTMs and GRUs overcome RNN’s vanishing gradient problem.  
- Implemented and compared LSTM and GRU models on IMDB sentiment analysis.  
- LSTM captured long-term dependencies better; GRU trained faster with similar accuracy.  
- Achieved ~88–89% accuracy with LSTM and ~87–88% with GRU.  

## Day 18 - NLP & Word Embeddings

- Understood how word embeddings represent semantic meaning in vector space.  
- Implemented a simple embedding-based sentiment model using Keras.  
- Compared static vs contextual embeddings (Word2Vec vs BERT).  
- Achieved ~86–88% accuracy on IMDB with learned embeddings.  
- Prepared ground for contextual embeddings (Day 22 - BERT).  

## Day 19 - Sentiment Analysis Project

- Built a complete end-to-end sentiment analysis pipeline using IMDB dataset.  
- Implemented an LSTM model with word embeddings and dropout regularization.  
- Visualized training, validation, and confusion matrix results.  
- Achieved ~90% accuracy on IMDB test set.  
- Saved the trained model for later deployment (Day 30+ in MLOps).  

## Day 20 - Attention Mechanisms

- Learned how attention allows neural networks to focus on important sequence parts.  
- Implemented a custom attention layer on top of a Bidirectional LSTM.  
- Observed significant improvement over plain RNN/LSTM models.  
- Achieved ~90–91% accuracy on the IMDB dataset.  

## Day 21 - Transformer From Scratch

- Learned the full architecture of a Transformer Encoder.  
- Implemented positional encoding, multi-head attention, and feed-forward networks manually.  
- Built a mini Transformer for IMDB sentiment analysis.  
- Achieved ~87–89% accuracy without pretraining.  
- Prepared foundation for Day 22 (BERT using Hugging Face).  

## Day 22 - BERT & Hugging Face

- Learned how BERT uses bidirectional Transformers for language understanding.  
- Understood pretraining tasks: MLM and NSP.  
- Fine-tuned bert-base-uncased on IMDB sentiment dataset.  
- Used Hugging Face Trainer API for efficient training.  
- Achieved ~92–94% accuracy, outperforming all previous NLP models.  

## Day 23 - Image Captioning

- Learned multimodal deep learning using CNN + LSTM architecture.  
- Understood encoder–decoder paradigm for vision-language tasks.  
- Studied image feature extraction using pretrained CNNs.  
- Implemented a simplified image captioning pipeline.  
- Built foundation for advanced multimodal models (CLIP, BLIP).  

## Day 24 - Deep Learning Capstone Project

- Built an end-to-end sentiment intelligence system using BERT.
- Fine-tuned a pretrained Transformer on IMDB dataset.
- Designed a production-style deep learning pipeline.
- Achieved ~93–94% accuracy on test data.
- Prepared the model for deployment and MLOps integration.

## Day 25 - Introduction to MLOps

- Learned what MLOps is and why it is critical for production ML systems.
- Studied the complete ML lifecycle and common failure points.
- Understood data versioning, experiment tracking, deployment, and monitoring.
- Designed a scalable project structure for future MLOps work.
- Prepared foundation for MLflow and DVC integration.

## Day 26 - Experiment Tracking with MLflow

- Learned why experiment tracking is essential in ML systems.
- Understood MLflow components and lifecycle integration.
- Logged parameters, metrics, and model artifacts using MLflow.
- Compared experiment runs using MLflow UI.
- Prepared foundation for model registry and deployment.

## Day 27 - Data Version Control (DVC)

- Learned why data versioning is critical in ML systems.
- Understood how DVC extends Git for large datasets.
- Versioned datasets using `.dvc` files.
- Integrated data tracking with Git workflows.
- Prepared foundation for reproducible ML pipelines.

## Day 28 - Model Registry & Model Versioning

- Learned why model versioning is essential in production ML systems.
- Understood MLflow Model Registry and lifecycle stages.
- Registered trained models with version control.
- Managed promotion to staging and production.
- Completed the model governance pillar of MLOps.

## Day 29 - Modular Training Pipelines

- Learned why modular ML pipelines are essential for production systems.
- Refactored monolithic scripts into reusable components.
- Implemented config-ready training and evaluation logic.
- Prepared training code for MLflow, DVC, and deployment.

## Day 30 - Model Serving with FastAPI

- Learned how ML models are served in production systems.
- Built a real-time inference API using FastAPI.
- Defined request and response schemas with Pydantic.
- Loaded trained model efficiently at application startup.
- Prepared the service for containerization and deployment.

## Day 31 - Dockerization of ML Systems

- Learned why Docker is essential for production ML systems.
- Containerized a FastAPI inference service.
- Wrote a Dockerfile and requirements.txt for ML deployment.
- Built and ran Docker containers locally.
- Prepared the system for cloud deployment.

## Day 32 - Cloud Deployment

- Learned how ML systems are deployed to cloud platforms.
- Deployed a Dockerized FastAPI inference service.
- Configured cloud build and start commands.
- Exposed a public prediction endpoint.
- Completed the deployment phase of MLOps.

## Day 33 - Model Monitoring & Drift Detection

- Learned why deployed ML models degrade over time.
- Studied data drift, concept drift, and prediction drift.
- Implemented data drift detection using Evidently AI.
- Generated monitoring reports for production models.
- Closed the feedback loop in the ML lifecycle.

## Day 34 - CI/CD Automation for ML Systems

- Learned CI/CD concepts specific to ML systems.
- Designed an automated pipeline using GitHub Actions.
- Validated ML inference during CI runs.
- Automated Docker image builds.
- Improved reliability and deployment safety.

## Day 35 - Streamlit Interface

- Learned why ML systems need a product layer.
- Built a Streamlit UI for ML inference.
- Connected UI to backend FastAPI service.
- Enabled non-technical interaction with ML models.
- Created a demo-ready ML product interface.

## Day 36 - Documentation & Model Cards

- Learned why documentation is essential for ML systems.
- Created a model card describing model behavior and limitations.
- Documented inference API endpoints.
- Improved transparency and maintainability of the ML system.
- Prepared the system for collaboration and audits.

## Day 37 - Git Workflow & Collaboration

- Learned Git workflows tailored for ML projects.
- Studied branching strategies and pull request practices.
- Integrated Git workflows with CI/CD pipelines.
- Understood collaboration challenges in ML systems.
- Prepared for team-based MLOps development.
