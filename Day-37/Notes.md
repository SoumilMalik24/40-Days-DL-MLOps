# Day 37 — Git Workflow & Collaboration for ML Systems

## Overview

This document introduces **Git workflows and collaboration practices** specifically
tailored for machine learning systems.

Unlike traditional software, ML projects involve code, data, models, and experiments.
This day focuses on how teams collaborate safely, reproducibly, and efficiently
using Git-based workflows in MLOps environments.

---

## Why Git Workflows Matter in ML

Machine learning systems are built by teams, not individuals.

Without a clear workflow:
- Changes conflict
- Experiments become irreproducible
- Deployments break unexpectedly
- Rollbacks are difficult
- Accountability is lost

A disciplined Git workflow is essential for reliable MLOps.

---

## Git Challenges Unique to ML Projects

ML projects introduce challenges not present in standard software:

- Large files (datasets, models)
- Frequent experiment changes
- Configuration-heavy code
- Non-deterministic results
- Multiple deployment environments

Git workflows must account for these realities.

---

## Core Git Principles for MLOps

1. Git tracks **code and metadata**, not large data
2. Data and models are versioned using DVC and registries
3. Every change must be traceable
4. Main branch must always be deployable

These principles guide all collaboration decisions.

---

## Recommended Branching Strategy

### Main Branch
- Always stable
- Always deployable
- Protected branch

### Feature Branches
- Used for new features, experiments, or refactors
- Named descriptively

Examples:
- `feature/add-monitoring`
- `feature/refactor-training-pipeline`
- `experiment/new-embedding`

---

## Typical Git Workflow for ML Teams

```
Create Feature Branch
↓
Implement Changes
↓
Commit Small, Clear Changes
↓
Push Branch
↓
Open Pull Request
↓
Run CI/CD Checks
↓
Review & Merge to Main
```

No direct commits to `main`.

---

## Commit Best Practices for ML Projects

Good commits:
- Are small and focused
- Describe *why*, not just *what*
- Do not mix unrelated changes

Bad commits:
- "fix stuff"
- "update"
- Large, monolithic commits

---

## Example Commit Messages

Good:
- "Add data drift monitoring with Evidently"
- "Refactor training pipeline into modular components"
- "Update FastAPI inference schema"

Bad:
- "changes"
- "final version"
- "working now"

---

## Handling Experiments with Git

Experiments should:
- Live in feature branches
- Use configuration files
- Log results via MLflow
- Never be committed as raw outputs

Git tracks:
- Code changes
- Config changes
- DVC metadata

MLflow tracks:
- Metrics
- Artifacts
- Runs

---

## Pull Requests in ML Projects

Pull Requests (PRs) should include:
- Description of the change
- Why the change is needed
- Impact on model performance
- Testing performed

PRs enable:
- Code review
- Knowledge sharing
- Error detection
- Team alignment

---

## CI/CD Integration with Git Workflow

CI/CD pipelines should run:
- On every pull request
- On every merge to main

CI checks may include:
- Code linting
- Import tests
- Inference tests
- Docker builds

Only passing PRs should be merged.

---

## How This Connects to Previous Days

- Day 26–28: Experiments, data, and models are versioned
- Day 29: Modular pipelines support clean diffs
- Day 34: CI/CD enforces quality gates
- Day 37: Git workflow ties everything together

This day formalizes **team-level MLOps practices**.

---

## Learning Outcomes

After Day 37, you can:
- Design Git workflows for ML projects
- Collaborate safely on ML systems
- Review and merge ML code responsibly
- Maintain deployable main branches