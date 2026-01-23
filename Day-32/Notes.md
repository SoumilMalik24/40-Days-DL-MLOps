# Day 32 — Cloud Deployment of ML Systems

## Overview

This document introduces **cloud deployment**, the process of making a machine learning
system accessible over the internet using cloud infrastructure.

On this day, the Dockerized ML inference service built on Day 31 is deployed to a cloud
platform, transforming it from a local system into a publicly accessible application.

---

## What is Cloud Deployment?

Cloud deployment is the process of running an application on remote infrastructure
managed by a cloud provider rather than on a local machine.

For ML systems, cloud deployment enables:
- Public accessibility
- Scalability
- Reliability
- Resource management

---

## Why Cloud Deployment is Critical for ML Systems

A machine learning system that runs only on a local machine:
- Cannot be used by others
- Cannot scale
- Cannot be monitored realistically
- Has no real-world value

Cloud deployment turns an ML model into a **usable service**.

---

## Common Cloud Deployment Options for ML

### 1. Platform-as-a-Service (PaaS)
- Minimal infrastructure management
- Fast deployment
- Ideal for demos and MVPs

Examples:
- Render
- Hugging Face Spaces
- Railway

---

### 2. Infrastructure-as-a-Service (IaaS)
- Full control over servers
- More complex setup

Examples:
- AWS EC2
- Google Compute Engine
- Azure Virtual Machines

Day 32 focuses on **PaaS-style deployment**, which is ideal for portfolio projects.

---

## Deployment Architecture

```
Client (Browser / App)
↓
Cloud URL
↓
Docker Container
↓
FastAPI Inference Service
↓
ML Model
```

This mirrors real-world production deployments.

---

## Environment Configuration

In cloud environments:
- Configuration must not be hardcoded
- Secrets and paths should use environment variables
- Containers should remain stateless

This ensures portability and security.

---

## Deployment Workflow

1. Push Dockerized code to GitHub
2. Connect repository to cloud platform
3. Configure build and start commands
4. Deploy container
5. Verify public endpoint

Once deployed, the API is accessible via a public URL.

---

## Best Practices for Cloud ML Deployment

- Use Docker images for consistency
- Expose only required ports
- Log application output
- Handle startup failures gracefully
- Monitor resource usage
- Avoid hardcoded credentials

---

## How This Connects to Previous Days

- Day 30: FastAPI created the inference API
- Day 31: Docker packaged the service
- Day 32: Cloud runs the container publicly
- Day 33: Monitoring will be added to this deployment

Day 32 completes the **deployment milestone**.

---

## Learning Outcomes

After Day 32, you can:
- Explain cloud deployment for ML systems
- Deploy Dockerized ML services
- Make models accessible via public endpoints
- Understand cloud-based ML architecture
