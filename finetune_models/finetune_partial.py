import os
import sys
import argparse
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
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())

from transformers import *
from utils import gen_utils, data_utils
from LM_extractor import get_model as get_lm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Freezing Logic
# ------------------------------------------------------------------------------
def freeze_layers(model, n_freeze):
    """
    Freezes the embeddings and the first n_freeze layers of the model.
    Assumes standard HF BERT/RoBERTa structure.
    """
    print(f"Freezing embeddings and first {n_freeze} layers...")

    # Freeze embeddings (always freeze if we are freezing layers)
    if hasattr(model, "embeddings"):
        for param in model.embeddings.parameters():
            param.requires_grad = False

    # Freeze encoder layers
    # BERT/RoBERTa usually have 'encoder.layer'
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    elif hasattr(model, "bert") and hasattr(model.bert.encoder, "layer"):
        # DistilBERT or some wrappers
        layers = model.bert.encoder.layer
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        # Some other architectures
        layers = model.transformer.layer
    else:
        print(
            "Warning: Could not find encoder layers to freeze. structure might be different."
        )
        return

    # Freeze the first n_freeze layers
    for i in range(min(n_freeze, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False

    # Verify freezing
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Print only first few to check
            if "layer" in name:
                # print(f"  {name}")
                pass
            else:
                pass  # classifier head etc


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class LM_MLP_MultiTrait(nn.Module):
    def __init__(self, lm, hidden_dim, embed_mode, n_traits):
        super().__init__()
        self.lm = lm
        self.embed_mode = embed_mode
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_dim, 50)
        self.relu = nn.ReLU()
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

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


class LM_MLP_SingleTrait(nn.Module):
    def __init__(self, lm, hidden_dim, embed_mode, n_classes=1):
        super().__init__()
        self.lm = lm
        self.embed_mode = embed_mode
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_dim, 50)
        self.relu = nn.ReLU()
        # Binary classification (can use 1 output with BCE or 2 with CE. Let's use 1 with BCE for consistency)
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

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


# ------------------------------------------------------------------------------
# Logging & Visualization
# ------------------------------------------------------------------------------
def plot_history(history, save_path):
    """
    Plots training and validation loss/accuracy.
    history: dict containing lists 'train_loss', 'val_loss', 'val_acc'
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    # val_acc might be a list of scalars (avg) or list of lists (per trait)
    # For simplicity, if multi-trait, we plot average, or we plot individual lines

    val_accs = np.array(history["val_acc"])
    if len(val_accs.shape) > 1:  # Multi-trait
        avg_acc = np.mean(val_accs, axis=1)
        plt.plot(avg_acc, label="Average Val Accuracy", linewidth=2, color="black")
        for i in range(val_accs.shape[1]):
            plt.plot(val_accs[:, i], label=f"Trait {i} Acc", linestyle="--", alpha=0.7)
    else:
        plt.plot(val_accs, label="Validation Accuracy")

    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


# ------------------------------------------------------------------------------
# Training Helpers
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.float().to(DEVICE)

        # If single trait, labels might need shape adjustment [B, 1] usually
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    # For accuracy
    total_correct = None
    total_samples = 0
    total_correct_scalar = 0  # For single trait

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE)
        labels = labels.float().to(DEVICE)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.sigmoid(logits).round()

        if labels.shape[1] > 1:  # Multi-trait
            if total_correct is None:
                total_correct = torch.zeros(labels.shape[1], device=DEVICE)
            total_correct += (preds == labels).sum(dim=0)
        else:  # Single trait
            total_correct_scalar += (preds == labels).sum().item()

        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)

    if labels.shape[1] > 1:
        accuracies = (total_correct / total_samples).cpu().numpy()  # [n_traits]
        return avg_loss, accuracies
    else:
        accuracy = total_correct_scalar / total_samples
        return avg_loss, accuracy


# ------------------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------------------
def training(args, input_ids, targets, trait_labels):
    n_splits = 10
    n_traits = len(trait_labels)

    # Prepare data
    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(DEVICE)
    targets = np.asarray(targets)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []  # To store final results

    from utils.log_utils import HistoryLogger

    # Create log directory
    log_dir = f"{args.log_dir}/partial_freeze{args.n_freeze}_{args.head_type}_{args.embed}"
    logger = HistoryLogger(log_dir)

    if args.head_type == "multi":
        print(f"Starting Multi-Head Training (jointly predicting {n_traits} traits)...")
        # Just train one model for all traits

        for fold, (tr, te) in enumerate(kf.split(input_ids, targets), 1):
            print(f"=== Fold {fold}/{n_splits} ===")
            y_train = torch.tensor(targets[tr], dtype=torch.float)
            y_test = torch.tensor(targets[te], dtype=torch.float)

            train_ds = TensorDataset(input_ids[tr], y_train)
            test_ds = TensorDataset(input_ids[te], y_test)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            # Initialize Model
            lm, _, _, hidden_dim = get_lm(args.embed)
            freeze_layers(lm, args.n_freeze)  # FREEZE HERE

            model = LM_MLP_MultiTrait(lm, hidden_dim, args.embed_mode, n_traits).to(
                DEVICE
            )
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()

            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_accs = evaluate(model, test_loader, criterion)

                # Multi-trait val_accs is an array [n_traits]
                avg_val_acc = np.mean(val_accs)

                logger.log_epoch(
                    fold, epoch + 1, train_loss, val_loss, avg_val_acc, val_accs
                )

                print(
                    f"Ep {epoch+1}: T_Loss={train_loss:.4f}, V_Loss={val_loss:.4f}, V_Acc (Avg)={avg_val_acc:.4f}"
                )

            # Plot history for this fold
            logger.save_logs(f"logs_fold_{fold}.json")
            logger.plot_curves(f"curves_fold{fold}")

            # Record final result
            final_loss, final_accs = evaluate(model, test_loader, criterion)
            results.append({"fold": fold, "trait": "Avg", "acc": np.mean(final_accs)})
            for i, trait in enumerate(trait_labels):
                results.append({"fold": fold, "trait": trait, "acc": final_accs[i]})

        logger.save_logs("final_logs_multi.json")

    else:  # single head - separate models
        print(f"Starting Single-Head Training (Independent models per trait)...")

        for i, trait in enumerate(trait_labels):
            print(f"--- Training Trait: {trait} ---")
            y_trait = targets[:, i]

            for fold, (tr, te) in enumerate(kf.split(input_ids, y_trait), 1):
                print(f"Fold {fold}/{n_splits}")
                y_train = torch.tensor(y_trait[tr], dtype=torch.float)
                y_test = torch.tensor(y_trait[te], dtype=torch.float)

                train_ds = TensorDataset(input_ids[tr], y_train)
                test_ds = TensorDataset(input_ids[te], y_test)
                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_ds, batch_size=args.batch_size, shuffle=False
                )

                lm, _, _, hidden_dim = get_lm(args.embed)
                freeze_layers(lm, args.n_freeze)  # FREEZE HERE

                # Revert to n_classes=2 for CrossEntropyLoss
                model = LM_MLP_SingleTrait(
                    lm, hidden_dim, args.embed_mode, n_classes=2
                ).to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                criterion = nn.CrossEntropyLoss()

                fold_key = f"{trait}_fold{fold}"

                for epoch in range(args.epochs):
                    # Manual training loop step to handle different label types if needed
                    # But train_one_epoch assumes float labels for BCE.
                    # We need to adapt it.
                    # Let's write a specific loop here or update train_one_epoch to be generic.
                    # Since we are inside the 'else', let's just inline the loop to be safe and clear.

                    model.train()
                    train_loss = 0.0
                    for input_ids_batch, labels_batch in train_loader:
                        input_ids_batch = input_ids_batch.to(DEVICE)
                        labels_batch = labels_batch.long().to(DEVICE)  # CE needs long

                        optimizer.zero_grad()
                        logits = model(input_ids_batch)
                        loss = criterion(logits, labels_batch)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()

                    train_loss /= len(train_loader)

                    # Evaluate
                    model.eval()
                    val_loss_accum = 0.0
                    total_correct_scalar = 0
                    total_samples = 0

                    with torch.no_grad():
                        for input_ids_batch, labels_batch in test_loader:
                            input_ids_batch = input_ids_batch.to(DEVICE)
                            labels_batch = labels_batch.long().to(DEVICE)

                            logits = model(input_ids_batch)
                            loss = criterion(logits, labels_batch)
                            val_loss_accum += loss.item()

                            # Argmax for CE
                            preds = torch.argmax(logits, dim=1)
                            total_correct_scalar += (preds == labels_batch).sum().item()
                            total_samples += labels_batch.size(0)

                    val_loss = val_loss_accum / len(test_loader)
                    val_acc = total_correct_scalar / total_samples

                    logger.log_epoch(fold_key, epoch + 1, train_loss, val_loss, val_acc)
                    print(f"Ep {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

                results.append({"fold": fold, "trait": trait, "acc": val_acc})

                # Save logs and plot for this trait
                logger.save_logs(f"logs_{trait}_{fold}.json")
                logger.plot_curves(f"curves_{trait}_{fold}")
            
                # Reset logs for next trait to avoid cluttered plots? Or keep cumulatively?
                # Creating a new logger instance or clearing logs is cleaner for per-trait files.
                logger.logs = {}

    return pd.DataFrame(results)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default="essays")
    parser.add_argument("-embed", type=str, default="bert-base")
    parser.add_argument("-embed_mode", type=str, default="cls", help="cls or mean")
    parser.add_argument("-lr", type=float, default=2e-5)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-token_length", type=int, default=512)
    parser.add_argument(
        "-n_freeze", type=int, default=8, help="Number of layers to freeze"
    )
    parser.add_argument(
        "-head_type",
        type=str,
        default="multi",
        choices=["multi", "single"],
        help="multi: joint model, single: separate models",
    )
    parser.add_argument("-mode", type=str, default="512_head", help="legacy mode arg")
    parser.add_argument(
        "-log_dir", type=str, default="log", help="Log directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Running with: {args}")

    # Load Data
    lm_temp, tokenizer, n_hl, hidden_dim = get_lm(args.embed)
    author_ids, input_ids, targets = data_utils.load_raw_text_dataset(
        args.dataset, tokenizer, args.token_length, args.mode
    )

    if args.dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    # Run Training
    df_results = training(args, input_ids, targets, trait_labels)

    # Save Results
    out_file = f"results_freeze{args.n_freeze}_{args.head_type}.csv"
    df_results.to_csv(out_file, index=False)
    print("Results:")
    print(df_results.groupby("trait")["acc"].mean())
    print(f"Saved results to {out_file}")
