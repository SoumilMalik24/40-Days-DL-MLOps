# Day 36 — Documentation & Model Cards

## Overview

This document focuses on **documentation**, a critical but often overlooked component
of production machine learning systems.

On this day, the ML system built so far is formally documented using:
- Project-level documentation
- API documentation
- Model Cards

Good documentation ensures that ML systems are understandable, auditable,
maintainable, and trustworthy.

---

## Why Documentation is Critical in ML Systems

Machine learning systems are complex because they involve:
- Data pipelines
- Training logic
- Models
- Infrastructure
- Monitoring and automation

Without documentation:
- Knowledge is lost
- Onboarding becomes difficult
- Debugging is slow
- Trust in the system decreases
- Compliance becomes impossible

Documentation turns ML projects into sustainable systems.

---

## Types of Documentation in MLOps

### 1. Project Documentation
Describes:
- What the system does
- How components interact
- How to run the system locally and in production

Audience:
- Engineers
- Reviewers
- Contributors

---

### 2. API Documentation
Describes:
- Available endpoints
- Input/output formats
- Error handling
- Example requests

Audience:
- Frontend developers
- Integrators
- External users

---

### 3. Model Documentation (Model Cards)
Describes:
- Model purpose
- Training data
- Evaluation metrics
- Limitations
- Ethical considerations

Audience:
- ML engineers
- Product owners
- Auditors
- Stakeholders

---

## What is a Model Card?

A **Model Card** is a standardized document that summarizes essential information
about a machine learning model.

Model cards were introduced to promote:
- Transparency
- Responsible AI
- Accountability
- Fair use

They are increasingly required in regulated and enterprise environments.

---

## Typical Sections of a Model Card

1. Model Overview  
2. Intended Use  
3. Training Data  
4. Evaluation Metrics  
5. Performance  
6. Limitations  
7. Ethical Considerations  
8. Maintenance & Updates  

---

## Why Model Cards Matter in Production

Model cards help answer:
- What was this model trained for?
- When should it not be used?
- How reliable is it?
- How often should it be retrained?

They prevent misuse and misinterpretation of models.

---

## How Documentation Fits into the ML Lifecycle

Design → Build → Train → Deploy → Monitor
↓
Document & Review


Documentation is continuous, not a one-time task.

---

## How This Connects to Previous Days

- Day 30–32: API and deployment exist
- Day 33: Monitoring tracks behavior
- Day 34: CI/CD automates changes
- Day 36: Documentation explains the system holistically

Day 36 makes the system understandable beyond its original author.

---

## Learning Outcomes

After Day 36, you can:
- Write professional ML documentation
- Create model cards for ML systems
- Document APIs and architectures
- Improve trust and maintainability of ML systems