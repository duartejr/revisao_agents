# Tutorial Authoring Guide

This reference defines how to write high-signal tutorials for Python libraries.

## Tutorial Objectives

A good tutorial should:
- Solve one concrete user problem.
- Be runnable end-to-end.
- Minimize prerequisites and hidden setup.
- End with expected output and next steps.

## Recommended Tutorial Structure

```text
1. Goal
2. Prerequisites
3. Install and setup
4. Step-by-step implementation
5. Validation (expected output)
6. Troubleshooting
7. Next steps
```

## Template

````md
# Build Your First Forecast

## Goal
Create a minimal forecast from a CSV file.

## Prerequisites
- Python 3.11+
- `pip install your-package`

## Step 1: Load data
```python
from your_package import load_data

data = load_data("sample.csv")
print(data.head())
```

## Step 2: Train model
```python
from your_package import train

model = train(data)
```

## Step 3: Predict
```python
pred = model.predict(14)
print(pred[:3])
```

## Expected output
- Model trains in under 30 seconds on sample data.
- `pred` returns a sequence of 14 values.

## Troubleshooting
- If import fails, verify virtual environment activation.
- If training is slow, use sample dataset from `examples/`.

## Next steps
- Evaluate with custom metrics.
- Replace sample data with your dataset.
````

## Writing Standards

- Keep each tutorial focused on one task.
- Use short sections and executable snippets.
- Show realistic inputs and outputs.
- Prefer copy-paste-ready code blocks.
- Avoid unexplained project-specific assumptions.

## Example Quality Checklist

```text
Tutorial Quality:
- [ ] Single clear objective
- [ ] Setup steps are complete
- [ ] Code runs in listed order
- [ ] Expected output is explicit
- [ ] Errors and fixes are documented
- [ ] Next steps are actionable
```

## Decision Rules

1. If users are beginners, prefer guided steps and more context.
2. If users are advanced, reduce narrative and emphasize API patterns.
3. If a tutorial exceeds 15 minutes, split into beginner and advanced tracks.
4. If snippets are reused in docs, centralize them in tested examples.

## Maintenance Practices

- Re-run tutorials each release cycle.
- Track tutorial execution time and dependencies.
- Add regression tests for canonical tutorial snippets where practical.
