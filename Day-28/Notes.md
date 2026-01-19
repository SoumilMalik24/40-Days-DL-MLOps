# Day 28 — Model Registry & Model Versioning

## Overview

This document introduces **Model Registry and Model Versioning**, a critical MLOps capability
that enables teams to manage, govern, and deploy machine learning models reliably.

While experiment tracking helps decide *which model is best* and data versioning ensures
*what data was used*, a model registry answers the question:

Which model is currently trusted and running in production?

---

## Why Model Versioning is Necessary

In real-world ML systems, multiple models exist simultaneously:

- Multiple experiments
- Multiple data versions
- Multiple retrained models
- Multiple deployment environments

Without model versioning:
- Teams do not know which model is deployed
- Rollbacks are difficult or impossible
- Production failures are hard to debug
- Governance and audits fail

Model versioning introduces control and traceability.

---

## What is a Model Registry?

A **Model Registry** is a centralized system that:

- Stores trained models
- Assigns version numbers
- Tracks model metadata
- Manages lifecycle stages
- Controls promotion to production

It acts as the **single source of truth** for models.

---

## MLflow Model Registry

MLflow provides a built-in **Model Registry** that supports:

- Model versioning
- Model staging
- Model promotion
- Model annotations
- Deployment integration

Typical lifecycle stages:
- None
- Staging
- Production
- Archived

---

## Model Lifecycle Stages

### 1. None
- Newly logged model
- Not yet evaluated

### 2. Staging
- Passed validation
- Under testing
- Candidate for production

### 3. Production
- Actively serving predictions
- Trusted model

### 4. Archived
- Deprecated
- Kept for audit or rollback

This lifecycle enforces discipline in model deployment.

---

## Model Registry Workflow

```
Train Model
↓
Log Model (MLflow)
↓
Register Model
↓
Promote to Staging
↓
Validate
↓
Promote to Production
```

This workflow ensures **controlled deployment**.

---

## Model Registry vs Experiment Tracking

| Experiment Tracking | Model Registry |
|--------------------|----------------|
| Logs runs | Governs models |
| Many experiments | Few approved models |
| Research-focused | Production-focused |
| No lifecycle | Explicit lifecycle stages |

Both are required in mature MLOps systems.

---

## How Model Registry Fits into the ML Lifecycle

```
Data (DVC)
↓
Training (MLflow Tracking)
↓
Model Registry
↓
Deployment
↓
Monitoring
```


The registry acts as the **bridge between training and deployment**.

---

## How This Connects to the Capstone Project

Day 24 BERT sentiment model will:

- Be logged as an MLflow artifact
- Be registered as a versioned model
- Be promoted to Production
- Be served via FastAPI (Day 30)
- Be monitored in production (Day 33)

Day 28 ensures this process is structured and safe.

---

## Governance and Compliance Benefits

Model registries support:
- Audit trails
- Rollbacks
- Reproducibility
- Approval workflows
- Team collaboration

These are mandatory in regulated industries.

---

## Learning Outcomes

After Day 28, you can:
- Explain model versioning clearly
- Use MLflow Model Registry
- Manage model lifecycle stages
- Connect training outputs to deployment inputs



