# Day 33 — Model Monitoring & Drift Detection

## Overview

This document introduces **model monitoring**, the practice of continuously observing
a deployed machine learning system to ensure it remains accurate, reliable, and safe
over time.

Model monitoring is a core MLOps responsibility because machine learning models
can silently fail even when the code and infrastructure remain unchanged.

---

## Why Model Monitoring is Necessary

Unlike traditional software, ML models are:
- Data-dependent
- Probabilistic
- Sensitive to changes in input distribution

A model that performed well at deployment time can degrade without any code changes.

Common causes:
- Changes in user behavior
- Changes in data sources
- Seasonal trends
- Feedback loops

Monitoring exists to detect these failures early.

---

## What is Model Drift?

**Model drift** refers to changes that negatively impact model performance after deployment.

There are three main types of drift.

---

## 1. Data Drift (Covariate Drift)

Data drift occurs when the **input data distribution changes** over time.

Examples:
- Customer demographics change
- Sensor readings shift
- Text vocabulary evolves

Symptoms:
- Model receives unfamiliar inputs
- Performance degrades gradually

Detection:
- Statistical comparison of feature distributions
- Population Stability Index (PSI)
- Kolmogorov–Smirnov test

---

## 2. Concept Drift

Concept drift occurs when the **relationship between input and target changes**.

Examples:
- User sentiment changes meaning over time
- Fraud patterns evolve
- Market conditions shift

Symptoms:
- Model predictions become systematically wrong
- Accuracy drops even if inputs look similar

Detection:
- Monitoring prediction quality
- Comparing predictions to delayed ground truth

---

## 3. Prediction Drift

Prediction drift occurs when the **model output distribution changes**.

Examples:
- Model starts predicting mostly one class
- Confidence scores collapse

Detection:
- Monitoring prediction distributions
- Threshold-based alerts

---

## Why Drift is Dangerous

Drift is often:
- Silent
- Gradual
- Invisible without monitoring

Without monitoring:
- Models lose business value
- Decisions become unreliable
- Trust in ML systems erodes

---

## What is Model Monitoring?

Model monitoring involves continuously tracking:

- Input data distributions
- Prediction distributions
- Performance metrics
- System health metrics

Monitoring is a continuous process, not a one-time task.

---

## Monitoring Metrics

### Data Metrics
- Feature means
- Feature variance
- Missing values
- Category frequency

### Model Metrics
- Accuracy
- Precision / Recall
- F1 score
- Confidence distributions

### System Metrics
- Latency
- Error rates
- Throughput

---

## Introduction to Evidently AI

**Evidently AI** is an open-source library for:
- Data drift detection
- Model performance monitoring
- Interactive dashboards

It is widely used in MLOps workflows for post-deployment monitoring.

---

## How Monitoring Fits into the ML Lifecycle

```
Deploy Model
↓
Collect Inference Data
↓
Monitor Distributions
↓
Detect Drift
↓
Trigger Retraining or Alerts
```

Monitoring closes the MLOps feedback loop.

---

## How This Connects to Previous Days

- Day 30–32: Model is deployed and serving predictions
- Day 33: Model behavior is continuously evaluated
- Day 34: CI/CD will automate responses to failures

Day 33 ensures the deployed system remains trustworthy.

---

## Learning Outcomes

After Day 33, you can:
- Explain data and concept drift clearly
- Monitor ML models in production
- Use Evidently AI for drift detection
- Design feedback loops for retraining
