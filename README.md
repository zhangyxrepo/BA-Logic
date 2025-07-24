# A Clean-Label Backdoor Attack by Poisoning the Inner Prediction Logic of Graph Neural Networks

## Announcement
This repository is maintained anonymously for submission and review purposes by SIGKDD 2026 double-blind reviewing guidelines. All materials are intended solely for evaluation during the conference review process.

## Introduction
We propose a clean-label graph backdoor attack method, BA-Logic, to conduct effective backdoor attacks under clean-label settings by poisoning the inner logic of target models.

<img src="Framework.png" alt="image_alt_text" width="100%"/>

## Independencies

- Python 3.9.20
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0

## Usage

To run the BA-Logic model with a specific task, use the following command:

```bash
python run_clip.py
```

You can modify `run_clip.py` to change the input data or the task being performed.
