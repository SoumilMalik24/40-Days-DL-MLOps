# Day 27 — Data Version Control (DVC)

## Overview

This document introduces **Data Version Control (DVC)**, the second major pillar of MLOps.
It explains why data versioning is critical, how DVC works, and how it integrates with Git
to make machine learning experiments reproducible and auditable.

---

## Why Data Versioning is Necessary

In machine learning, **data is as important as code**.

Unlike traditional software:
- Data changes over time
- Small changes in data can significantly affect model performance
- Training data is often too large to store directly in Git

Without data versioning:
- Experiments cannot be reproduced
- Model performance cannot be trusted
- Debugging becomes impossible
- Collaboration breaks down

---

## What is DVC?

**DVC (Data Version Control)** is an open-source tool that enables:

- Versioning large datasets
- Tracking data pipelines
- Reproducing experiments
- Integrating data versioning with Git

DVC does **not** replace Git.
It extends Git to handle data and ML artifacts efficiently.

---

## How DVC Works (Conceptual)

DVC separates:
- **Metadata** (stored in Git)
- **Actual data files** (stored in external storage)
Workflow:
```
Data File → dvc add → .dvc file → Git commit
↓
Remote Storage (disk / cloud)
```

Git tracks `.dvc` files.
DVC manages the actual data behind the scenes.

---

## Core DVC Concepts

### 1. DVC Files (`.dvc`)
- Small metadata files
- Contain checksums and paths
- Tracked by Git

### 2. DVC Cache
- Local storage for data versions
- Avoids duplication

### 3. Remote Storage
- Location where actual data is stored
- Can be local disk, S3, GCS, Azure, etc.

### 4. Pipelines
- Define data and model workflows
- Enable reproducibility

---

## Git vs DVC Responsibilities

| Git | DVC |
|----|----|
| Source code | Large datasets |
| Config files | Model artifacts |
| `.dvc` metadata | Actual data files |
| Branching logic | Data lineage |

Git and DVC work together, not separately.

---

## Typical DVC Workflow

1. Initialize DVC in a repository
2. Add data using `dvc add`
3. Commit `.dvc` files to Git
4. Push data to remote storage
5. Pull exact data versions when needed

This ensures:
- Same code + same data = same results

---

## How DVC Fits into the ML Lifecycle
```
Raw Data
↓
Versioned with DVC
↓
Training (MLflow tracking)
↓
Model Artifacts
↓
Deployment
```

DVC ensures that **every model can be traced back to the exact data version** it was trained on.

---

## How This Connects to the Capstone Project

The Day 24 sentiment model depends heavily on training data.

With DVC:
- Training data versions are tracked
- Model performance changes can be explained
- Retraining becomes reliable

DVC + MLflow together provide:
- Data lineage
- Experiment lineage

---

## Common Mistakes Without DVC

- Overwriting datasets
- Losing old training data
- Inconsistent results across machines
- “It worked on my system” failures

DVC prevents all of these.

---

## Learning Outcomes

- Explain why data versioning is essential
- Use DVC with Git in real projects
- Track dataset changes safely
- Reproduce ML experiments reliably

