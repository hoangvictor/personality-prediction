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
from sklearn.model_selection import StratifiedKFold

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
class LM_MLP(nn.Module):
    def __init__(self, lm, hidden_dim, embed_mode, n_classes):
        super().__init__()
        self.lm = lm
        self.embed_mode = embed_mode

        self.fc1 = nn.Linear(hidden_dim, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, n_classes)

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
        labels = labels.float().to(DEVICE).unsqueeze(1)

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
    correct = 0
    total = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(input_ids)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


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
    dropout,
):
    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    n_classes = 1
    n_splits = 10

    expdata = {"acc": [], "trait": [], "fold": []}
    best_models = {}

    from utils.log_utils import HistoryLogger

    # Create log directory
    log_dir = f"logs/full_finetune_{embed}_{jobid}"
    logger = HistoryLogger(log_dir)

    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(DEVICE)
    targets = np.asarray(targets)

    for trait_idx, trait in enumerate(trait_labels):
        y = targets[:, trait_idx]

        best_trait_acc = 0.0
        best_trait_model = None
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        for fold, (tr, te) in tqdm(enumerate(skf.split(input_ids, y), 1)):
            y_train = torch.tensor(y[tr], dtype=torch.float)
            y_test = torch.tensor(y[te], dtype=torch.float)

            train_ds = TensorDataset(input_ids[tr], y_train)
            test_ds = TensorDataset(input_ids[te], y_test)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            lm, _, _, hidden_dim = get_lm(embed)
            lm.dropout = dropout

            model = LM_MLP(
                lm=lm,
                hidden_dim=hidden_dim,
                embed_mode=embed_mode,
                n_classes=n_classes,
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            best_fold_acc = 0.0
            best_fold_model = None

            for epoch in range(epochs):
                loss = train_one_epoch(model, train_loader, optimizer, criterion)

                # Evaluate val loss and acc
                model.eval()
                val_loss_accum = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for v_input_ids, v_labels in test_loader:
                        v_input_ids = v_input_ids.to(DEVICE)
                        # BCE needs [Batch, 1], so unsqueeze
                        v_labels = v_labels.float().to(DEVICE).unsqueeze(1)
                        v_logits = model(v_input_ids)
                        v_loss = criterion(v_logits, v_labels)
                        val_loss_accum += v_loss.item()

                        preds = torch.sigmoid(v_logits).round()
                        correct += (preds == v_labels).sum().item()
                        total += v_labels.size(0)

                val_loss = val_loss_accum / len(test_loader)
                val_acc = correct / total

                print(
                    f"Epoch {epoch+1} - Trait {trait} - Fold {fold} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

                # Log with trait included in fold name to distinguish
                logger.log_epoch(
                    f"{trait}_fold{fold}", epoch + 1, loss, val_loss, val_acc
                )

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_fold_model = model

            print(f"Best validation accuracy: {best_fold_acc}")
            expdata["acc"].append(100 * best_fold_acc)
            expdata["trait"].append(trait)
            expdata["fold"].append(fold)

            if best_fold_acc > best_trait_acc:
                best_trait_acc = best_fold_acc
                best_trait_model = best_fold_model

        # Plot curves for this trait after all folds are done (or inside loop per fold)
        # Let's plot per fold inside loop or at end.
        logger.save_logs(f"logs_{trait}.json")
        logger.plot_curves(f"curves_{trait}")

        best_models[trait] = {
            "model_state": best_trait_model.state_dict(),
            "acc": best_trait_acc,
        }

    if str(save_model).lower() == "yes":
        out_dir = Path("finetune_lm_mlp/")
        out_dir.mkdir(parents=True, exist_ok=True)

        for trait, state in best_models.items():
            torch.save(
                state,
                out_dir / f"LM_MLP_{trait}.pt",
            )

    logger.save_logs("final_logs.json")
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
        dropout,
    ) = gen_utils.parse_args_full_finetune()

    torch.manual_seed(jobid)
    np.random.seed(jobid)

    lm, tokenizer, n_hl, hidden_dim = get_lm(embed)
    lm.to(DEVICE)

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
        dropout=dropout,
    )
    df.to_csv(f"{embed}_expdata.csv")
    print(df.head())
