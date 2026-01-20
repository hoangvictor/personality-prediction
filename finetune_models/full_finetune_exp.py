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
        labels = labels.to(DEVICE)

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
    save_path
):
    os.makedirs(save_path, exist_ok=True)
    csv_file_path = os.path.join(save_path, "results.csv")
    models_dir = os.path.join(save_path, "models")

    results_df = None
    if os.path.exists(csv_file_path):
        results_df = pd.read_csv(csv_file_path)

    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    n_classes = 2
    n_splits = 10

    expdata = {"acc": [], "trait": [], "fold": []}
    if results_df is not None:
        expdata["acc"] = results_df["acc"].tolist()
        expdata["trait"] = results_df["trait"].tolist()
        expdata["fold"] = results_df["fold"].tolist()

    best_models = {}
    if os.path.exists(models_dir):
        for trait in trait_labels:
            if os.path.exists(os.path.join(models_dir, f"best_{trait}.pt")):
                best_models[trait] = torch.load(os.path.join(models_dir, f"best_{trait}.pt")),

    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    ).to(DEVICE)
    targets = np.asarray(targets)

    for trait_idx, trait in enumerate(trait_labels):
        print("-"*50)
        print(f"Trait {trait}")
        y = targets[:, trait_idx]

        best_trait_acc = 0.0
        best_trait_model = None
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        for fold, (tr, te) in tqdm(enumerate(skf.split(input_ids, y), 1)):
            if results_df is not None and results_df[(results_df["fold"] == fold) & (results_df["trait"] == trait)].shape[0] > 0:
                continue

            print(f"Fold {fold}")
            y_train = torch.tensor(y[tr], dtype=torch.long)
            y_test = torch.tensor(y[te], dtype=torch.long)

            train_ds = TensorDataset(input_ids[tr], y_train)
            test_ds = TensorDataset(input_ids[te], y_test)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False
            )
            lm, _, _, hidden_dim = get_lm(embed)
            lm.dropout = dropout

            model = LM_MLP(
                lm=lm,
                hidden_dim=hidden_dim,
                embed_mode=embed_mode,
                n_classes=n_classes,
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            best_fold_acc = 0.0
            best_fold_model = None
            losses = []
            val_accs = []

            for _ in tqdm(range(epochs)):
                loss = train_one_epoch(model, train_loader, optimizer, criterion)
                print(f"Loss: {loss}")

                val_acc = evaluate(model, test_loader)

                losses.append(loss)
                val_accs.append(val_acc)

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_fold_model = model

            print(f"Losses: {losses}")
            print(f"Validation accuracies: {val_accs}")
            print(f"Best validation accuracy: {best_fold_acc}")
            expdata["acc"].append(100 * best_fold_acc)
            expdata["trait"].append(trait)
            expdata["fold"].append(fold)

            if best_fold_acc > best_trait_acc:
                best_trait_acc = best_fold_acc
                best_trait_model = best_fold_model

            results_df = pd.DataFrame(expdata)
            results_df.to_csv(csv_file_path)
        
        if best_trait_model:
            best_models[trait] = {
                "model_state": best_trait_model.state_dict(),
                "acc": best_trait_acc,
            }

        if str(save_model).lower() == "yes":
            out_dir = Path(models_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            for trait, state in best_models.items():
                torch.save(
                    state,
                    out_dir / f"best_{trait}.pt",
                )


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
        dropout
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

    for epochs in [10, 20, 40]:
        print("-"*50)
        print("-"*50)
        print("-"*50)
        print(f"Epochs {epochs}")
        training(
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
            save_path=f"{embed}_expdata_num_epochs_{epochs}"
        )
