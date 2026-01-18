# Day 57: Introduction to PyTorch

## 1. What is PyTorch?
**PyTorch** is an open-source Deep Learning library developed by Meta (Facebook).
It is the standard for **Research** and increasingly for **Production**.
Why?
- **Pythonic**: It feels like writing normal Python code.
- **Dynamic**: You can change the graph on the fly (easier debugging).
- **GPU Acceleration**: It runs on NVIDIA GPUs using CUDA.

---

## 2. Installation
```bash
# Visit pytorch.org to get the exact command for your OS/GPU
pip install torch torchvision
```

---

## 3. The Core Philosophy
1.  **Tensor**: A NumPy array that lives on the GPU.
2.  **Autograd**: Automatic differentiation (Calculus engine).
3.  **Module**: Neural Network layers (`nn.Linear`, `nn.Conv2d`).

---

## 4. Hello World
```python
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a tensor
x = torch.rand(5, 3).to(device)
print(x)
```

---

## 5. Summary
- **PyTorch**: The tool used to build Neural Networks.
- **CUDA**: The software layer that lets PyTorch use your GPU.

**Next Up:** **Tensors**—The building blocks.
