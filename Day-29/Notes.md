# Day 29 — Modular Training Pipelines

## Overview

This document introduces **modular training pipelines**, a critical step in transforming
machine learning code from experimental scripts into **production-grade systems**.

The goal of Day 29 is to refactor monolithic notebooks and scripts into:
- Reusable components
- Clear data flow
- Config-driven pipelines
- Training logic that is reproducible, testable, and deployable

This is a foundational skill for all real-world MLOps systems.

---

## Why Modular Pipelines Are Necessary

Most beginner ML projects fail to scale because they rely on:
- Single large scripts
- Hardcoded parameters
- Implicit dependencies
- Notebook-only workflows

Problems caused by non-modular code:
- Difficult debugging
- No reusability
- Poor collaboration
- Inconsistent training and inference logic
- Impossible automation

Modular pipelines solve these problems.

---

## What is a Modular ML Pipeline?

A modular ML pipeline separates concerns into **independent, reusable stages**.

Typical stages:
1. Data ingestion
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Artifact saving

Each stage:
- Has a single responsibility
- Can be tested independently
- Can be reused or replaced

---

## Script-Based Pipelines vs Modular Pipelines

### Script-Based (Anti-Pattern)
- One file does everything
- Parameters hardcoded
- No clear interfaces

### Modular Pipeline (Best Practice)
- Multiple small modules
- Config-driven behavior
- Clear input/output contracts
- Easy automation

---

## Recommended Project Structure

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       └── config.py
├── tests/
│   ├── unit/
│   └── integration/
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```


This structure separates data, logic, configuration, and artifacts.

---

## Configuration-Driven Training

Hardcoding parameters is a major anti-pattern.

Instead, all configurable values should live in a config file:
- Learning rate
- Batch size
- Number of epochs
- Model parameters
- File paths

Benefits:
- Reproducibility
- Easier experimentation
- Cleaner code
- Better automation

---

## How Modular Pipelines Enable MLOps

Modular pipelines make it possible to:
- Plug in MLflow for tracking
- Integrate DVC for data versioning
- Automate training via CI/CD
- Deploy models consistently
- Monitor models reliably

Without modularization, none of these scale.

---

## How This Connects to Previous Days

- Day 26 (MLflow): tracking attaches to training module
- Day 27 (DVC): data loaders pull versioned data
- Day 28 (Model Registry): trained models are registered
- Day 29: glue that holds everything together

This is the **engineering backbone** of the MLOps phase.

---

## Learning Outcomes

After Day 29, you can:
- Design modular ML codebases
- Separate training, evaluation, and data logic
- Use config-driven ML pipelines
- Prepare ML systems for automation and deployment

