# Poisoning the Inner Prediction Logic of Graph Neural Networks for Clean-Label Backdoor Attacks

## Announcement
This repository is maintained anonymously for submission and review purposes by the double-blind reviewing guidelines of Transactions on Machine Learning Research. All materials are intended solely for evaluation during the conference review process.

## Introduction
We propose a clean-label graph backdoor attack method, BA-Logic, to conduct effective backdoor attacks by poisoning the inner logic of target models under the clean-label setting.

<img src="Framework.png" alt="image_alt_text" width="100%"/>

## Independencies

- Python 3.9.20
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0

## Usage

To run the BA-Logic model with a specific task, use the following command:

```bash
python run_logic.py
```

You can modify `run_logic.py` to change the input data or the task being performed.
