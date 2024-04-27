---
license: apache-2.0
library_name: peft
tags:
- generated_from_trainer
base_model: distilbert-base-uncased
metrics:
- accuracy
model-index:
- name: distilbert-base-uncased-lora-text-classification
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-lora-text-classification

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8437
- Accuracy: {'accuracy': 0.881}

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy            |
|:-------------:|:-----:|:----:|:---------------:|:-------------------:|
| No log        | 1.0   | 250  | 0.3423          | {'accuracy': 0.886} |
| 0.4235        | 2.0   | 500  | 0.3493          | {'accuracy': 0.892} |
| 0.4235        | 3.0   | 750  | 0.5340          | {'accuracy': 0.881} |
| 0.207         | 4.0   | 1000 | 0.6471          | {'accuracy': 0.868} |
| 0.207         | 5.0   | 1250 | 0.7612          | {'accuracy': 0.874} |
| 0.0831        | 6.0   | 1500 | 0.8176          | {'accuracy': 0.875} |
| 0.0831        | 7.0   | 1750 | 0.8788          | {'accuracy': 0.872} |
| 0.0284        | 8.0   | 2000 | 0.8236          | {'accuracy': 0.886} |
| 0.0284        | 9.0   | 2250 | 0.8466          | {'accuracy': 0.881} |
| 0.0128        | 10.0  | 2500 | 0.8437          | {'accuracy': 0.881} |


### Framework versions

- PEFT 0.10.1.dev0
- Transformers 4.41.0.dev0
- Pytorch 2.1.0+cpu
- Datasets 2.19.0
- Tokenizers 0.19.1