# Day 30 — Model Serving with FastAPI

## Overview

This document introduces **model serving**, the process of exposing a trained machine
learning model so that it can be used by external systems in real time.

On this day, we transition from offline model training to **online inference**, which is
a core requirement for production ML systems.

---

## What is Model Serving?

Model serving is the practice of making a trained model available for inference via:

- APIs
- Batch jobs
- Streaming systems

In most modern ML systems, model serving is implemented using **HTTP-based APIs**.

---

## Why Model Serving is Critical

A trained model that cannot serve predictions is effectively useless.

Without proper model serving:
- Applications cannot consume model predictions
- Models cannot be deployed to production
- Monitoring and retraining are impossible
- ML systems cannot scale

Model serving is the bridge between ML and software systems.

---

## Common Model Serving Patterns

### 1. Online (Real-Time) Inference
- Low latency
- API-based
- Used for user-facing applications

### 2. Batch Inference
- Large-scale predictions
- Scheduled jobs
- Used for analytics and reporting

### 3. Streaming Inference
- Continuous data flow
- Event-driven systems

Day 30 focuses on **online inference**, the most common interview and industry use case.

---

## Why FastAPI?

FastAPI is a modern Python web framework designed for APIs.

Advantages:
- High performance
- Automatic validation using type hints
- Easy integration with ML models
- Automatic API documentation
- Widely used in production ML systems

FastAPI has become the standard for ML inference APIs.

---

## Core Components of an ML Inference API

1. **Model Loader**
   - Loads trained model once at startup
   - Avoids reloading on every request

2. **Request Schema**
   - Validates input data
   - Prevents malformed requests

3. **Prediction Logic**
   - Applies preprocessing
   - Generates predictions

4. **Response Schema**
   - Returns structured output

---

## Typical Model Serving Flow

```
Client Request
↓
FastAPI Endpoint
↓
Input Validation
↓
Model Inference
↓
Prediction Response
```


---

## Best Practices for Model Serving

- Load model at application startup
- Keep inference stateless
- Validate all inputs
- Log requests and errors
- Separate training and inference code
- Make APIs deterministic and fast

---

## How This Connects to Previous Days

- Day 29: Modular training produces a saved model artifact
- Day 30: That artifact is loaded and served via API
- Day 31: This API will be containerized using Docker
- Day 32: The container will be deployed to the cloud

Day 30 is the **entry point to production**.

---

## Learning Outcomes

After Day 30, you can:
- Explain model serving clearly
- Build inference APIs using FastAPI
- Load and serve trained ML models
- Design production-ready prediction endpoints

