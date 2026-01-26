# Installation Guide

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekaagramupta/roadDamageDetection/blob/main/notebooks/detect.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells

## Local Installation

### Prerequisites
- Python 3.10 or higher
- CUDA 11.8+ (for GPU support)

### Steps
```bash
# Clone repository
git clone https://github.com/ekaagramupta/roadDamageDetection.git
cd roadDamageDetection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting

### GPU not detected
- Install CUDA Toolkit 11.8+
- Install cuDNN 8.6+
- Verify: `nvidia-smi`

### Import errors
- Ensure Python 3.10+
- Reinstall: `pip install -r requirements.txt --force-reinstall`