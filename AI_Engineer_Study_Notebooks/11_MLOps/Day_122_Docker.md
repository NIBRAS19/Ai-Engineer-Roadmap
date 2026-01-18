# Day 122: Docker & Kubernetes

## 1. The "It works on my machine" Problem
Your laptop has Python 3.9, Library X version 1.2.
The server has Python 3.8, Library X version 1.0.
Your code crashes on the server.

**Docker** solves this by packaging your Code + OS + Libraries + Python into a single **Container**.
If it runs in Docker, it runs ANYWHERE.

---

## 2. Docker Basics

### 2.1 The Dockerfile (Recipe)
```dockerfile
# 1. Start from a base OS with Python installed
FROM python:3.9-slim

# 2. Copy your requirements
COPY requirements.txt .

# 3. Install dependencies
RUN pip install -r requirements.txt

# 4. Copy your code
COPY . .

# 5. Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

### 2.2 Commands
- `docker build -t my-ml-app .` (Bake the cake).
- `docker run -p 80:80 my-ml-app` (Eat the cake).

---

## 3. Scaling: Introduction to Kubernetes (K8s)
Docker runs **one** container. usage is simple.
But what if you need **50 containers** to handle million users? And if one crashes, you need to restart it instantly?

**Kubernetes** is an **Orchestrator**. It manages a fleet of containers.

### 3.1 Key Concepts
1.  **Pod**: The smallest unit. Usually contains 1 container (e.g., your ML API).
2.  **Node**: A physical or virtual machine (Server) that runs Pods.
3.  **Cluster**: A group of Nodes.
4.  **Service**: The "Receptionist". It gives a stable IP address to a set of Pods and load balances traffic between them.

### 3.2 Why K8s for AI?
- **Auto-Scaling**: Only 10 users? Run 1 Pod. 10,000 users? K8s automatically spins up 50 Pods.
- **Self-Healing**: If a Pod crashes (Out of Memory), K8s restarts it automatically.
- **GPU Sharing**: Assign GPUs to specific Pods.

---

## 4. Summary
- **Docker**: Packages the app (Standard Unit).
- **Kubernetes**: Manages the shipping traffic (Orchestration).

**Next Up:** **MLflow & DVC**—Tracking Experiments and Data.

