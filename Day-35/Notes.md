# Day 35 — Streamlit Interface (ML Product Layer)

## Overview

This document introduces the **ML product layer**, where a machine learning system
is exposed through a simple, interactive user interface.

On this day, a Streamlit application is built on top of the deployed inference API,
transforming the ML system into a usable product rather than just an engineering artifact.

---

## Why an ML Product Layer is Important

Most machine learning projects fail to create impact because:
- They are difficult to demo
- Only engineers can interact with them
- Stakeholders cannot visualize results

A product layer:
- Makes ML accessible to non-technical users
- Improves communication and adoption
- Increases portfolio and business value

---

## What is Streamlit?

Streamlit is a Python framework for building lightweight web applications.

Key properties:
- Minimal boilerplate
- Tight integration with Python and ML
- Fast prototyping
- Ideal for ML demos and internal tools

Streamlit is widely used for ML dashboards and product demos.

---

## Architecture with Streamlit

```
User Interface (Streamlit)
↓
Inference API (FastAPI)
↓
ML Model
```

The UI does not contain ML logic.
It only sends requests to the backend inference service.

---

## Why UI Should Be Separate from Inference

Separation of concerns is critical.

Benefits:
- UI can change without retraining models
- Backend can scale independently
- Cleaner architecture
- Easier maintenance

This mirrors real-world ML systems.

---

## Typical Streamlit Use Cases in MLOps

- Model demos
- Internal dashboards
- Prediction explorers
- Stakeholder reviews
- Debugging inference behavior

Streamlit is not a replacement for production frontends,
but it is perfect for ML-facing products.

---

## Best Practices for Streamlit ML Apps

- Keep UI logic simple
- Call backend APIs, do not load models
- Validate user inputs
- Handle errors gracefully
- Log predictions if needed

---

## How This Connects to Previous Days

- Day 30: FastAPI inference service
- Day 31: Dockerized backend
- Day 32: Cloud deployment
- Day 35: User-facing product layer

Day 35 completes the **ML product experience**.

---

## Learning Outcomes

After Day 35, you can:
- Build ML demo applications
- Connect UIs to inference APIs
- Present ML systems to stakeholders
- Create portfolio-ready ML products

---