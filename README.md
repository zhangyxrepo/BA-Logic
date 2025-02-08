# Clean-Label Backdoor Attacks by Poisoning the Inner Prediction Logic of Graph Neural Networks

## Announcement
This anonymous repository is only used for submissions to ACM KDD 2025.

## Introduction
We propose a clean-label graph backdoor attack method, BA-Logic, to conduct effective backdoor attack under clean-label settings by poisoning the inner logic of target models.

<img src="Framework.png" alt="image_alt_text" width="100%"/>

## Independencies

- Python 3.9.17
- PyTorch 2.0.1
- PyTorch Geometric 2.5.2

## Usage

To run the BA-Logic model with a specific task, use the following command:

```bash
python run_clip.py
```

You can modify `run_clip.py` to change the input data or the task being performed.
