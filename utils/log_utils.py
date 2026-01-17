import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class HistoryLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logs = {}

    def log_fold(self, fold, history):
        """
        Log complete history for a fold.
        history: dict with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
                 values should be lists of floats.
        """
        self.logs[f"fold_{fold}"] = history

    def log_epoch(
        self, fold, epoch, train_loss, val_loss, val_acc, val_acc_per_trait=None
    ):
        """
        Log a single epoch's metrics.
        """
        fold_key = f"fold_{fold}"
        if fold_key not in self.logs:
            self.logs[fold_key] = {
                "loss": [],
                "val_loss": [],
                "val_acc": [],
            }
            if val_acc_per_trait is not None:
                self.logs[fold_key]["val_acc_per_trait"] = []

        self.logs[fold_key]["loss"].append(float(train_loss))
        self.logs[fold_key]["val_loss"].append(float(val_loss))
        self.logs[fold_key]["val_acc"].append(float(val_acc))

        if val_acc_per_trait is not None:
            self.logs[fold_key]["val_acc_per_trait"].append(
                [float(x) for x in val_acc_per_trait]
            )

    def save_logs(self, filename="training_logs.json"):
        """Save logs to a JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs saved to {filepath}")

    def plot_curves(self, filename_prefix="learning_curve"):
        """
        Generate plots for each fold.
        """
        for fold_name, history in self.logs.items():
            epochs = range(1, len(history["loss"]) + 1)

            plt.figure(figsize=(12, 5))

            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, history["loss"], "b-", label="Training Loss")
            if "val_loss" in history:
                plt.plot(epochs, history["val_loss"], "r--", label="Validation Loss")
            plt.title(f"{fold_name} - Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            # Plot Accuracy
            plt.subplot(1, 2, 2)
            # Check for different accuracy keys
            acc_key = None
            val_acc_key = None

            if "accuracy" in history:
                acc_key = "accuracy"
            elif "acc" in history:
                acc_key = "acc"

            if "val_accuracy" in history:
                val_acc_key = "val_accuracy"
            elif "val_acc" in history:
                val_acc_key = "val_acc"

            if acc_key:
                plt.plot(epochs, history[acc_key], "b-", label="Training Acc")

            if val_acc_key:
                plt.plot(epochs, history[val_acc_key], "r--", label="Validation Acc")

            if "val_acc_per_trait" in history:
                # Plot individual traits if available
                traits_acc = np.array(history["val_acc_per_trait"])
                for i in range(traits_acc.shape[1]):
                    plt.plot(
                        epochs, traits_acc[:, i], linestyle=":", label=f"Trait {i}"
                    )

            plt.title(f"{fold_name} - Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"{filename_prefix}_{fold_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Plot saved to {save_path}")
