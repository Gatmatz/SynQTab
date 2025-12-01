# Reproducibility Modifications in TabPFN Library

This document summarizes the changes needed in the TabPFN library to ensure deterministic results by setting a fixed random seed.

---

## 1. `_sgld_step` Function

In the `_sgld_step` function, after the comment:

```python
# Update using gradients and noise
```

Add the following code:

```python
generator = torch.Generator(device=self.device)
generator.manual_seed(42)

rand_like = torch.randn(
    x_synth.size(),
    generator=generator,
    dtype=x_synth.dtype,
    layout=x_synth.layout,
    device=x_synth.device
)
noise = rand_like * np.sqrt(2 * self.sgld_step_size)
```

This ensures that the stochastic gradient Langevin dynamics step is reproducible.

---

## 2. `generate_classification` Function

In the classification task, inside the `else` branch, use a fixed random seed for generating synthetic data:

```python
generator = torch.Generator(device=self.device)
generator.manual_seed(42)

x_synth = torch.randn(
    n_samples, X_train.shape[1],
    device=self.device,
    generator=generator
) * 0.01

y_synth = torch.randint(
    0, len(np.unique(y_train)),
    (n_samples,),
    device=self.device,
    generator=generator
)
```

---

## 3. `TabPFNClassifier`

When instantiating the classifier, set a fixed random state:

```python
classifier = TabPFNClassifier(random_state=42)
```

---

## 4. Regression Tasks

For regression, apply the same logic:

* Use `generator.manual_seed(42)` wherever random numbers are generated in the `_sgld_step` and data generation functions.
* Set `random_state=42` for `TabPFNRegressor`:

```python
regressor = TabPFNRegressor(random_state=42)
```

This ensures deterministic outputs for regression tasks as well.

