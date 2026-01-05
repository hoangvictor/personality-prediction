# Personality Prediction Script Usage

This document describes how to use the newly created multi-trait fine-tuning scripts. These scripts allow you to train a single model that predicts all personality traits simultaneously (Multi-Output).

## 1. Feature-based Fine-tuning (MLP only)
**Script**: `finetune_models/MLP_LM_multitrait.py`
**Description**: Trains a lightweight MLP on top of pre-extracted LM embeddings.

### Prerequisites
First, run the `LM_extractor.py` to generate the `.pkl` files containing the embeddings (if you haven't already).
```bash
python LM_extractor.py -dataset_type 'essays' -embed 'bert-base' -op_dir 'pkl_data'
```

### Training
To train the multi-trait MLP model:
```bash
python finetune_models/MLP_LM_multitrait.py \
    -dataset_type 'essays' \
    -embed 'bert-base' \
    -op_dir 'pkl_data/' \
    -save_model 'yes'
```
*   `op_dir`: Directory where `.pkl` embeddings are stored.
*   `save_model`: Set to 'yes' to save the best model to `finetune_mlp_lm_multitrait/`.

## 2. Full Fine-tuning (End-to-End)
**Script**: `finetune_models/full_finetune_multitrait.py`
**Description**: Trains the entire BERT model + MLP head end-to-end. This is computationally expensive but potentially more accurate.

### Training
```bash
python finetune_models/full_finetune_multitrait.py \
    -dataset_type 'essays' \
    -embed 'bert-base' \
    -batch_size 16 \
    -epochs 3 \
    -save_model 'yes'
```
*   The model will be saved to `finetune_lm_mlp_multitrait/LM_MLP_MultiTrait_Best.pt`.

## 3. Inference (Predicting on New Data)

To infer on new text using the trained **Full Fine-tuned Model** (`full_finetune_multitrait.py`), you can use the following python code snippet (you can save this as `inference_multitrait.py`):

```python
import torch
from transformers import BertTokenizer, BertModel
# Import the model class from the training script
from finetune_models.full_finetune_multitrait import LM_MLP_MultiTrait

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "finetune_lm_mlp_multitrait/LM_MLP_MultiTrait_Best.pt"
embed_name = "bert-base-uncased"

# Load Tokenizer & Base Model
tokenizer = BertTokenizer.from_pretrained(embed_name)
base_lm = BertModel.from_pretrained(embed_name)

# Initialize Model Structure (Must match training params)
# 5 traits for Essays, 4 for Kaggle
n_traits = 5 
model = LM_MLP_MultiTrait(lm=base_lm, hidden_dim=768, embed_mode='cls', n_traits=n_traits)

# Load State Dict
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Predict
texts = ["I really enjoy passing time with my friends and meeting new people."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    probs = torch.sigmoid(logits)
    predictions = probs.round()

print("Probabilities:", probs)
print("Predictions (Binary):", predictions)
```

Note: For the feature-based MLP (`MLP_LM_multitrait.py`), inference requires you to first extract embeddings using `LM_extractor` (or equivalent code) and then pass them to the saved Keras/TensorFlow model.
