# AIHT
# Devanagari Character Recognition with EHCR-Net v2

This repository contains an efficient deep learning model for recognizing Devanagari handwritten characters using an enhanced architecture with attention mechanisms and multi-scale feature extraction.

## Features

- EfficientNet-based backbone with CBAM attention
- Atrous Spatial Pyramid Pooling (ASPP) for multi-scale features
- Advanced preprocessing for Devanagari characters
- Comprehensive visualization tools
- High accuracy on the Devanagari Handwritten Character Dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/devanagari-character-recognition.git
cd devanagari-character-recognition
pip install -r requirements.txt
```

##Dataset Setup
Download the Devanagari Handwritten Character Dataset and place it in the data/ directory with the following structure:

```bash
data/
└── DevanagariHandwrittenCharacterDataset/
    ├── Train/
    │   ├── character_01_ka/
    │   ├── character_02_kha/
    │   └── ...
    └── Test/
        ├── character_01_ka/
        ├── character_02_kha/
        └── ...
```

## Training 
```bash
python training/train.py
```

## Evaluation
```bash
python training/evaluate.py
```

##Results
Our EHCR-Net v2 achieves 99.59% accuracy on the test set. See the visualization notebooks for detailed performance analysis.
