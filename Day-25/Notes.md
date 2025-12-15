# Day 25 — Introduction to MLOps (Foundations)

## Overview

This document serves as a knowledge-first introduction to MLOps, marking the transition from model-centric Deep Learning to production-grade Machine Learning systems.

While earlier days focused on building models, Day 25 focuses on operationalizing models — ensuring they are reproducible, deployable, monitorable, and maintainable over time.

---

## What is MLOps?

MLOps (Machine Learning Operations) is the practice of applying software engineering and DevOps principles to machine learning systems in order to:

- Deploy models reliably
- Reproduce experiments
- Manage data and model versions
- Monitor model performance in production
- Continuously improve models as data evolves

MLOps exists because machine learning systems behave differently from traditional software systems.

---

## Why MLOps is Necessary

Most machine learning projects fail after model training, not during it.

Common reasons include:
- Training results cannot be reproduced
- Training code differs from inference code
- Data changes silently over time
- No tracking of experiments or metrics
- Models degrade in production without detection
- No rollback or recovery strategy exists

MLOps addresses these problems systematically.

---

## Machine Learning Lifecycle vs Software Lifecycle

### Traditional Software Lifecycle
``` 
Code → Build → Test → Deploy → Monitor
```

### Machine Learning Lifecycle
```
Data → Train → Evaluate → Deploy → Monitor
↑______________________________|
```


Key distinction:
- Software changes mainly through code
- ML systems change through data

Because data evolves, ML systems require continuous validation and retraining.

---

## Core Pillars of MLOps

### 1. Data Versioning

Data is a first-class citizen in ML systems.

Goals:
- Track dataset versions
- Reproduce experiments exactly
- Detect data drift

Common tools:
- DVC
- Git-LFS

---

### 2. Experiment Tracking

Every experiment must be traceable.

Track:
- Hyperparameters
- Metrics
- Artifacts (models, plots)
- Code version

Common tools:
- MLflow
- Weights & Biases

---

### 3. Model Versioning and Registry

Models must be treated as versioned artifacts.

Capabilities:
- Store trained models
- Promote models across stages (Staging → Production)
- Rollback to previous versions

Common tools:
- MLflow Model Registry

---

### 4. Deployment

Models must be served reliably.

Common deployment patterns:
- REST APIs (FastAPI)
- Batch inference
- Streaming inference

Supporting technologies:
- Docker
- Cloud platforms (Render, AWS, GCP, Hugging Face Spaces)

---

### 5. Monitoring

Once deployed, models must be continuously monitored.

Monitor:
- Input data drift
- Prediction distribution drift
- Performance decay
- System health

Common tools:
- Evidently AI
- Custom dashboards

---

## MLOps Maturity Levels

| Level | Description |
|------|------------|
| Level 0 | Manual training and deployment |
| Level 1 | Automated training pipelines |
| Level 2 | CI/CD for ML systems |
| Level 3 | Continuous Training (CT) |

The goal of this journey is to reach Level 2 or higher MLOps maturity.

---

## DevOps vs MLOps

| DevOps | MLOps |
|------|------|
| Code-centric | Data-centric |
| Deterministic outputs | Probabilistic outputs |
| Static binaries | Evolving models |
| Unit tests | Data and model validation |

MLOps extends DevOps by introducing data and model management as core concerns.

---

## How This Connects to the Deep Learning Phase

The Day 24 Deep Learning Capstone becomes the central artifact for the MLOps phase.

That model will be:
- Tracked using MLflow
- Versioned using DVC
- Served via FastAPI
- Containerized with Docker
- Deployed to the cloud
- Monitored for drift and performance

Day 25 establishes the mental and architectural foundation for all of this.

---

## What Will Be Built in the MLOps Phase (Day 25–40)

By the end of this phase, the project will include:
- Reproducible experiments
- Versioned datasets
- A model registry
- A production inference API
- Dockerized deployment
- CI/CD automation
- Model monitoring dashboards

This represents a complete, industry-grade ML system.

---

## Key Takeaways

- Training a model is only a small part of ML engineering
- Data and models must be versioned and monitored
- ML systems require continuous maintenance
- MLOps bridges research and production
- This day marks the transition from ML practitioner to ML engineer

---
