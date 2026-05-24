# ARC: Audit-Guided Risk Calibration for Black-Box Multimodal Jailbreak Attacks

## Overview

This repository provides a partial implementation of **ARC**, an audit-guided risk calibration framework for studying black-box multimodal jailbreak attacks against multimodal large language models (MLLMs).

ARC focuses on the trade-off between **visible risk suppression** and **semantic recoverability**. It calibrates textual and visual risk exposure through audit feedback, and organizes the remaining multimodal cues into a scenario-grounded reconstruction input.

The current release includes the core code for ARC input construction and related evaluation utilities. Full experiment scripts, prompt templates, and reproduction configurations will be released after publication.

## Installation

We recommend using a Python virtual environment.

```bash
conda create -n arc python=3.10
conda activate arc
````

Install common dependencies:

```bash
pip install torch transformers pillow numpy tqdm openai
```

Depending on your local environment and target models, additional packages may be required.

Please configure API keys through environment variables rather than hard-coding them in the source files:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Data

This repository does not include benchmark datasets or generated model outputs. Please download the datasets from their official sources and organize them locally.

Example directory structure:

```text
data/
├── SafeBench/
└── HADES/
```

For ethical and legal reasons, we do not release harmful model responses, private data, or directly reusable high-risk examples.

## Ethical Considerations

This repository is intended only for controlled safety evaluation and responsible red-teaming research. The goal is to better understand the limitations of current multimodal safety alignment and to support the development of stronger defenses.

Users should follow the terms of service of model providers and applicable laws and regulations. The released code should not be used to generate, distribute, or facilitate harmful content. Generated harmful outputs should not be redistributed.

