# Unsloth-Llama3-Finetune
# Fine-Tuning Llama 3.2 3B with Unsloth on Google Colab

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Datasets-yellow?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-purple?style=for-the-badge&logo=googlecolab)

This repository contains a Jupyter Notebook demonstrating how to efficiently fine-tune the `unsloth/Llama-3.2-3B-Instruct` model using the **Unsloth** library. The entire process is optimized to run on a free Google Colab T4 GPU, making advanced model customization accessible to everyone.

## Overview

The goal of this project is to perform Supervised Fine-Tuning (SFT) on a powerful, small-sized language model. We leverage several cutting-edge techniques to achieve this with minimal hardware requirements:

- **Unsloth:** A library that significantly speeds up fine-tuning (by up to 2x) and reduces memory usage by patching Hugging Face transformers.
- **QLoRA:** Quantization with Low-Rank Adapters allows us to fine-tune the model in 4-bit precision, drastically cutting down the VRAM footprint.
- **Llama 3.2 3B:** A state-of-the-art 3-billion parameter model that offers a great balance between performance and resource requirements.
- **FineTome-100k Dataset:** A high-quality, curated dataset of 100,000 instruction-response pairs for effective fine-tuning.

## Key Features

- **🚀 Fast & Efficient Training:** Fine-tunes a 3B parameter model in approximately **12 minutes** on a single T4 GPU.
- **🧠 Parameter-Efficient Fine-Tuning (PEFT):** Uses LoRA adapters to train only a small fraction (0.75%) of the model's parameters, preserving the original model's knowledge while adapting it to new data.
- **💬 Chat-Formatted Data:** The notebook properly formats the dataset into a chat template (`llama-3.1`) for instruction-following tasks.
- **📊 Training Monitoring:** Integrated with Weights & Biases (`wandb`) for real-time tracking of training loss and other metrics.
- 

## Workflow

The notebook follows a clear, step-by-step process:

1.  **Environment Setup:** Installs necessary libraries (`unsloth`, `transformers`, `trl`, `datasets`) and configures the environment to be compatible with the Colab T4 GPU by disabling Flash Attention.
2.  **Model Loading:** Loads the `unsloth/Llama-3.2-3B-Instruct` model and its tokenizer using Unsloth's `FastLanguageModel`, which automatically applies performance patches and loads the model in 4-bit precision.
3.  **PEFT Configuration:** Injects LoRA adapters into the model's attention and MLP layers, making it ready for efficient training.
4.  **Data Preparation:** Loads the `mlabonne/FineTome-100k` dataset, standardizes it to the ShareGPT format, and applies the Llama 3.1 chat template.
5.  **Training:** Sets up and runs the `SFTTrainer` from the TRL library with optimized training arguments (batch size, learning rate, etc.) for 60 steps.
6.  **Saving the Model:** Saves the trained LoRA adapters to a local directory (`finetuned_Llama_model`) for later use.

## How to Run

You can run this project directly in Google Colab.

1.  **Open in Google Colab:**
2. The notebook explicitly sets environment variables before importing transformers/accelerate to disable FlashAttention, for example:
   ```python
      import os
      os.environ["USE_FLASH_ATTENTION"] = "0"
      os.environ["DISABLE_FLASH_ATTN"] = "1"
   
   ```   
3.  **Provide WandB API Key:**
    When you run the training cell, you will be prompted to enter your Weights & Biases API key to log the training run. You can get your key from [wandb.ai/authorize](https://wandb.ai/authorize).

4.  **Execute All Cells:**
    Run the remaining cells in order to download the model, prepare the data, and start the fine-tuning process.

## Results

The model was trained for 60 steps on the `FineTome-100k` dataset.
- **Training Time:** ~12 minutes on a T4 GPU.
- **Final Training Loss:** The training loss successfully decreased, ending at approximately **0.8742**.

This indicates that the model was learning effectively from the dataset.



## Future Improvements

- **Train for More Steps:** The model was only trained for 60 steps as a demonstration. Training for more epochs or steps would likely yield better performance.
- **Merge and Save:** The LoRA adapters can be merged into the base model and saved as a single model for easier deployment.
- **Push to Hub:** The fine-tuned model can be pushed to the Hugging Face Hub for sharing and community use.
- **Evaluation:** Implement a formal evaluation pipeline using a separate test set to measure performance on metrics like ROUGE, BLEU, or accuracy on specific tasks.
