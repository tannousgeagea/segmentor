# Installation Guide

## Standard Installation

### 1. Install Segmentor with PyTorch
```bash
pip install segmentor[torch]
```

### 2. Install SAM Models

SAM models are hosted on GitHub and must be installed separately:

**SAM v1 (Required):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**SAM v2 (Optional):**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 3. Verify Installation
```bash
python -m segmentor.quickstart
```

## Why Separate Installation?

PyPI doesn't allow packages to depend directly on Git repositories. This is a limitation of PyPI's infrastructure, not Segmentor.

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'segment_anything'"

**Solution:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Error: "No SAM models found"

**Solution:** Install at least SAM v1 (see above)

## Docker Installation

For Docker users, SAM models are included in the image:
```dockerfile
FROM python:3.10

RUN pip install segmentor[torch]
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Your app code here
```

## Requirements File

For reproducible environments, create `requirements.txt`:
```txt
# Core package
segmentor[torch]>=0.1.0

# SAM models (not on PyPI)
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
sam2 @ git+https://github.com/facebookresearch/segment-anything-2.git
```

Install with:
```bash
pip install -r requirements.txt
```