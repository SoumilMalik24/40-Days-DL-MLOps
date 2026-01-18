# Day 26 — Experiment Tracking with MLflow

## Overview

This document introduces **experiment tracking**, the first practical pillar of MLOps.
It explains why experiment tracking is essential, how MLflow solves this problem,
and how it fits into a production-grade ML system.

---

## What is Experiment Tracking?

Experiment tracking is the practice of systematically recording:

- Model parameters
- Hyperparameters
- Training metrics
- Artifacts (models, plots, files)
- Code version and run metadata

Its goal is to ensure **reproducibility, comparability, and auditability** of ML experiments.

Without tracking, ML development becomes guesswork.

---

## Why Experiment Tracking is Critical

In real ML projects, teams often ask:

- Which model performed best?
- What hyperparameters were used?
- Can we reproduce last month’s results?
- Why did performance suddenly degrade?

Without experiment tracking:
- Results cannot be reproduced
- Models cannot be trusted
- Collaboration breaks down

---

## What is MLflow?

**MLflow** is an open-source platform for managing the ML lifecycle.
It is framework-agnostic and widely used in industry.

MLflow provides four main components:

---

## MLflow Components

### 1. MLflow Tracking

Logs:
- Parameters
- Metrics
- Artifacts
- Runs

Allows comparison across experiments using a UI.

---

### 2. MLflow Projects
(Standardized project structure for reproducible runs)

(Not used today; introduced conceptually.)

---

### 3. MLflow Models
Standard format to package models for deployment.

---

### 4. MLflow Model Registry
Central place to manage model versions and stages
(Staging, Production, Archived).

---

## How MLflow Fits into the ML Lifecycle

```
Train Model
↓
Log Params, Metrics, Artifacts (MLflow)
↓
Compare Runs
↓
Register Best Model
↓
Deploy & Monitor
```


MLflow becomes the **single source of truth** for experiments.

---

## Key Concepts in MLflow

### Run
A single execution of training code.

### Experiment
A collection of related runs.

### Artifact
Any output file (model, plots, metrics).

### Parameter
Configuration value (learning rate, epochs).

### Metric
Numerical performance indicator (accuracy, loss).

---

## Benefits of MLflow

- Reproducibility
- Transparency
- Easy collaboration
- Model governance
- Deployment readiness

MLflow turns ML from experimentation into engineering.

---

## How This Connects to the Capstone Project

The BERT Sentiment Model from Day 24 will now be:
- Tracked using MLflow
- Compared across runs
- Prepared for registration and deployment

Day 26 marks the shift from **training models** to **managing models**.

---

## Learning Outcomes

- Explain experiment tracking clearly
- Use MLflow in real projects
- Compare multiple model runs
- Prepare models for registry and deployment.
