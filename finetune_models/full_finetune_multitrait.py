import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())

from transformers import *
from utils import gen_utils, data_utils
from LM_extractor import get_model as get_lm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset, tokenizer, token_length, mode):
    author_ids, input_ids, targets = data_utils.load_raw_text_dataset(
        dataset, tokenizer, token_length, mode
    )
    return author_ids, input_ids, targets


# ---------------------------
# Model
# ---------------------------
class LM_MLP_MultiTrait(nn.Module):
    def __init__(self, lm, hidden_dim, embed_mode, n_traits):
        super().__init__()
        self.lm = lm
        self.embed_mode = embed_mode

        self.fc1 = nn.Linear(hidden_dim, 50)
        self.relu = nn.ReLU()
        # Output layer now has n_traits units (one for each trait)
        self.fc2 = nn.Linear(50, n_traits)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if self.embed_mode == "cls":
            x = outputs.last_hidden_state[:, 0, :]
        else:
            x = outputs.last_hidden_state.mean(dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


# ---------------------------
# Train / Eval helpers
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.float().to(DEVICE)  # BCEWithLogitsLoss needs float targets

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    # We will track accuracy per trait
    total_correct = None
    total_samples = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids)
        # Apply sigmoid to get probabilities, then round to get 0 or 1
        preds = torch.sigmoid(logits).round()

        if total_correct is None:
            total_correct = torch.zeros(labels.shape[1], device=DEVICE)

        # accurate predictions per trait
        total_correct += (preds == labels).sum(dim=0)
        total_samples += labels.size(0)

    # Average accuracy across all samples for each trait
    accuracies = (total_correct / total_samples).cpu().numpy()
    return accuracies


# ---------------------------
# Training pipeline
# ---------------------------
def training(
    dataset,
    input_ids,
    targets,
    token_length,
    lm,
    hidden_dim,
    embed,
    embed_mode,
    lr,
    batch_size,
    epochs,
    save_model,
):
    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    n_traits = len(trait_labels)
    n_splits = 10

    expdata = {"trait": [], "fold": [], "acc": []}

    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(DEVICE)
    targets = np.asarray(targets)

    # Use KFold instead of StratifiedKFold for multi-label data
    # StratifiedKFold doesn't support multi-label natively well without complex combination classes
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_model_state = None
    best_avg_acc = 0.0

    # We train ONE model for ALL traits per fold
    for fold, (tr, te) in tqdm(enumerate(kf.split(input_ids, targets), 1)):
        print(f"Fold {fold}/{n_splits}")

        y_train = torch.tensor(targets[tr], dtype=torch.float)
        y_test = torch.tensor(targets[te], dtype=torch.float)

        train_ds = TensorDataset(input_ids[tr], y_train)
        test_ds = TensorDataset(input_ids[te], y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Reload fresh model for each fold
        lm_fold, _, _, hidden_dim = get_lm(embed)
        model = LM_MLP_MultiTrait(
            lm=lm_fold,
            hidden_dim=hidden_dim,
            embed_mode=embed_mode,
            n_traits=n_traits,
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        fold_best_acc = 0.0

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            # Evaluate
            accuracies = evaluate(model, test_loader)
            avg_acc = np.mean(accuracies)
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Avg Acc: {avg_acc:.4f} - Per trait: {accuracies}"
            )

            if avg_acc > fold_best_acc:
                fold_best_acc = avg_acc
                # Logic to save global best model could go here,
                # but typically we just report cross-val results

        # Record results for this fold
        # Since we evaluate after training, we verify final performance on test set
        final_accuracies = evaluate(model, test_loader)

        for i, trait in enumerate(trait_labels):
            expdata["trait"].append(trait)
            expdata["fold"].append(fold)
            expdata["acc"].append(final_accuracies[i] * 100)

        # If saving model, we might want to save the one from the best fold or retrain on all data.
        # Here, let's just save the model from the last fold or if it beats global best
        if np.mean(final_accuracies) > best_avg_acc:
            best_avg_acc = np.mean(final_accuracies)
            best_model_state = model.state_dict()

    if str(save_model).lower() == "yes" and best_model_state is not None:
        out_dir = Path("finetune_lm_mlp_multitrait/")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, out_dir / "LM_MLP_MultiTrait_Best.pt")
        print(f"Saved best model with avg accuracy: {best_avg_acc:.4f}")

    return pd.DataFrame(expdata)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    (
        inp_dir,
        dataset,
        lr,
        batch_size,
        epochs,
        log_expdata,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
        save_model,
        token_length,
    ) = gen_utils.parse_args_full_finetune()

    # Reproducibility
    torch.manual_seed(jobid)
    np.random.seed(jobid)

    # Initial LM load just to get dimensions,
    # actual training reload for each fold to avoid leakage/state retention
    lm, tokenizer, n_hl, hidden_dim = get_lm(embed)

    author_ids, input_ids, targets = load_dataset(
        dataset,
        tokenizer,
        token_length,
        mode,
    )

    df = training(
        dataset=dataset,
        input_ids=input_ids,
        targets=targets,
        token_length=token_length,
        lm=lm,
        hidden_dim=hidden_dim,
        embed=embed,
        embed_mode=embed_mode,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        save_model=save_model,
    )

    # Save results
    df.to_csv("expdata_multitrait.csv")
    print(df.groupby("trait")["acc"].mean())
