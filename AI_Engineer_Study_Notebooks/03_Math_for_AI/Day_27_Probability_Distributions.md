# Day 27: Probability Distributions

## 1. Introduction
Data isn't random chaos; it follows patterns called **Distributions**.
Knowing the distribution helps us choose the right model and loss function.

---

## 2. Normal Distribution (Gaussian)
The "Bell Curve".
- Defined by Mean ($\mu$) and Std Dev ($\sigma$).
- 68% of data is within $1\sigma$.
- 95% of data is within $2\sigma$.

**AI Context:**
- We initialize weights using a Normal Distribution.
- We assume noise in regression is Gaussian.
- We normalize inputs to be Standard Normal ($\mu=0, \sigma=1$).

---

## 3. Bernoulli Distribution
Coin flip. Binary outcome (0 or 1).
- Parameter $p$: Probability of success (1).

**AI Context:**
- Binary Classification (Cat vs Dog).
- The output of the final neuron (Sigmoid) represents $p$.

---

## 4. Categorical Distribution (Multinoulli)
Dice roll. Multiple categories ($K > 2$).
- Integers $1 ... K$.

**AI Context:**
- Multi-class Classification (Digit recognition 0-9).
- The output layer (Softmax) produces a categorical distribution.

---

## 5. Practical Exercises

### Exercise 1: Sampling
Use `np.random` to:
1.  Generate 1000 samples from a Normal Distribution ($\mu=10, \sigma=2$).
2.  Calculate their actual mean and std. Do they match 10 and 2?

---

## 6. Summary
- **Normal**: Continuous values (Height, Price).
- **Bernoulli**: Binary classes (Yes/No).
- **Categorical**: Multi-class (A/B/C).

**Next Up:** **Loss Functions**—Mathematically measuring "How bad is my model?".
