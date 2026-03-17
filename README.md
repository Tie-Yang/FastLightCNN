# FastLightCNN
FastLightCNN: An ultra-lightweight attention-based convolutional neural network for edge-oriented MNIST digit recognition (8.2K params, 0.11M FLOPs, 98.49% accuracy)

## Overview
FastLightCNN is an ultra-lightweight convolutional neural network **specifically engineered for resource-deprived IoT endpoints and edge microcontrollers (MCUs)**. It addresses the critical SRAM and computational bottlenecks of low-power edge hardware by combining depthwise separable convolutions with a novel Lightweight Channel Attention (LCA) module.

Key performance on MNIST:
- **98.49% Top-1 Accuracy**
- Only **8,210 parameters**
- Only **0.11M FLOPs**
- **0.55 ms inference time** (legacy CPU, no GPU acceleration)
- INT8 quantized size < **10 KB**
- **81.5% smaller memory footprint** than LeNet-5
- **60.7% fewer computational operations** than LeNet-5

---

## Performance Comparison (MNIST)
| Model         | Accuracy | Params   | FLOPs   | Inference Time |
|---------------|----------|----------|---------|----------------|
| LeNet-5       | 99.14%   | 44,426   | 0.28M   | 0.63 ms        |
| FastLightCNN  | 98.49%   | 8,210    | 0.11M   | 0.55 ms        |

> **Note**: A standard tolerance of ±0.05% for accuracy and ±0.05 ms for inference latency may exist across independent experimental runs due to environmental factors and data batching. All inference results are tested on a legacy Intel Core i7-3632QM CPU **without any GPU acceleration**.

---

## Key Innovations
1. **Depthwise Separable Convolution Design**
   Drastically reduces time and space complexity, optimized for strict MCU memory budgets.
2. **Lightweight Channel Attention (LCA) Module**
   Recovers representational precision lost during extreme miniaturization with negligible parameter overhead (reduction ratio r=8).
3. **Numerical Stability & Quantization-Friendly Pipeline**
   Two-stage input normalization ensures stable execution on microcontroller ALUs and optimal low-precision INT8 quantization.
4. **MCU-Optimized Topology**
   Tailored for 28×28 MNIST input, total parameters capped under 10K for real-time edge perception.

---

## Model Architecture
FastLightCNN adopts a streamlined structure strictly optimized for edge deployment:
- Input: 28×28×1 normalized grayscale image
- Conv1 (3×3) → MaxPool1 (2×2) → DW-Conv1 → LCA Module → MaxPool2 (2×2) → Flatten → Classifier

Detailed layer topology (from paper):
| Layer Name |         Operation       | Output Shape |   Params  |
|------------|-------------------------|--------------|-----------|
| Input      |   Image Normalization   |     28×28×1  |    0      |
| Conv1      |   Standard Conv (3×3)   |     28×28×8  |    80     |
| Pool1      |     MaxPool (2×2)       |     14×14×8  |    0      |
| DW-Conv1   | Depthwise + Pointwise   |   14×14×16   |   224     |
| LCA        | Channel Attention (r=8) |   14×14×16   |    56     |
| Pool2      |       MaxPool (2×2)     |     7×7×16   |    0      |
| Flatten    |       Tensor Reshape    |     1×784    |    0      |
| Classifier |     Linear + Softmax    |      1×10    |  7,850    |
| **Total**  |          -              |        -     | **8,210** |

---

## Lightweight Channel Attention (LCA) Module
The LCA module dynamically recalibrates cross-channel feature responses with minimal overhead:
1. Global Average Pooling (GAP) squeezes spatial information
2. Two FC layers with reduction ratio r=8 model channel dependencies
3. Sigmoid gating generates channel-wise attention weights
4. Feature map scaling recovers accuracy lost from depthwise factorization

---

## Edge Deployment
FastLightCNN is fully optimized for ultra-low-power MCUs:
- FP32 model size: ~32.84 KB
- INT8 quantized size: **8.21 KB**
- SRAM usage on 64KB MCU: <13%
- Compatible with: ARM Cortex-M / RISC-V MCUs (STM32, ESP8266, etc.)
- Theoretical SRAM utilization (INT8):
  - STM32F103: 41.05%
  - ESP8266: 10.26%
  - STM32F401: 8.55%

---

## Experimental Setup
- Framework: PyTorch
- Hardware: Intel Core i7-3632QM (legacy CPU, no GPU)
- Dataset: MNIST (28×28 grayscale)
- Epochs: 10
- Batch size: 64
- Optimizer: Adam (initial lr=0.001, with scheduler)
- Loss: Cross-entropy
- Input preprocessing: Linear scaling + Z-score standardization

---

## Ablation Study
| DW-Conv | LCA | Params | FLOPs | Accuracy |
|---------|-----|--------|-------|----------|
| ✗       | ✗   | 25,320 | 0.21M | 98.75%   |
| ✓       | ✗   | 8,122  | 0.10M | 97.82%   |
| ✓       | ✓   | 8,210  | 0.11M | 98.49%   |

The LCA module effectively restores accuracy with only 56 additional parameters, validating its efficiency for lightweight networks.

---

## Getting Started
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train, Test and Inference
```bash
python mnist_model.py
```

## Paper Access
This repository contains the Author’s Submitted Manuscript (AAM) of the paper: 
- FastLightCNN: An Ultra-Lightweight Attention-Based Convolutional Network for Edge-Oriented Digit Recognition
- To be reviewed for: Proceedings of the 3rd International Conference on Machine Learning and Neural Networks (MLNN’26), ACM, 2026.
- © 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
- For personal and academic use only. No commercial use allowed.

## Citation (paper is currently under review)
If you use this work in your research, please cite the original paper:
```bibtex
@inproceedings{yang2026fastlightcnn,
  title={FastLightCNN: An Ultra-Lightweight Attention-Based Convolutional Network for Edge-Oriented Digit Recognition},
  author={Yang, Kevin and Wang, Zihan and Yang, Tiebao},
  booktitle={Proceedings of the 3rd International Conference on Machine Learning and Neural Networks (MLNN'26)},
  year={2026},
  organization={ACM}
}
```

##License
- Code: MIT License
- Paper: © ACM 2026. Author’s Accepted Manuscript for non-commercial academic use only.
