# Day 26: Statistics Basics

## 1. Introduction
Statistics frames how we interpret data.
- Is this result significant?
- Is this image an outlier?
- How is the data distributed?

---

## 2. Descriptive Statistics (Review)
- **Mean ($\mu$)**: Average.
- **Median**: Middle value (Robust to outliers).
- **Mode**: Most frequent value.
- **Variance ($\sigma^2$)**: How spread out data is.
- **Standard Deviation ($\sigma$)**: Square root of variance.

---

## 3. Covariance and Correlation
How do two variables move together?
- **Covariance**: Direction of relationship (+/-). Unbounded measure.
- **Correlation ($r$)**: Normalized ($ -1 \le r \le 1 $).
    - +1: Perfect positive linear relationship.
    - -1: Perfect negative linear relationship.
    - 0: No linear relationship.

**AI Context:** Feature Selection.
If Feature A and Feature B have correlation 0.99, you can drop one of them (they carry the same info). This reduces model complexity.

---

## 4. Probability Basics
- **P(A)**: Probability of event A.
- **Conditional Probability P(A|B)**: Probability of A given B happened.
- **Bayes' Theorem**: Updating beliefs with new evidence.
$$ P(A|B) = \frac{P(B|A) P(A)}{P(B)} $$

**AI Context:** Naive Bayes Classifier (Spam filtering).

---

## 5. Practical Exercises

### Exercise 1: Correlation Matrix
Create a Pandas DataFrame with 3 columns:
- `A`: Random numbers.
- `B`: `2 * A` (Perfectly correlated).
- `C`: Random numbers (Unrelated).
Compute `df.corr()` and observe the values.

---

## 6. Summary
- **Correlation**: Measures linear relationships.
- **Bayes**: The foundation of probabilistic inference.

**Next Up:** **Probability Distributions**—The shape of data (Normal, Bernoulli).
