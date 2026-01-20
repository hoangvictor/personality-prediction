import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pickle
import time
import pandas as pd
from pathlib import Path

# add parent directory to the path
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())

import utils.gen_utils as utils


def get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer):
    """Read data from pkl file and prepare for training."""
    # Logic copied from MLP_LM.py but adapted if needed
    file = open(
        inp_dir + dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl", "rb"
    )
    data = pickle.load(file)
    author_ids, data_x, data_y = list(zip(*data))
    file.close()

    if "base" in embed:
        n_hl = 12
    elif "large" in embed:
        n_hl = 24

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)
    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    inputs = []
    targets = []
    n_batches = len(data_y)
    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))
        targets.extend(data_y[ii])

    inputs = np.array(inputs)
    full_targets = np.array(targets)

    return inputs, full_targets


def training(
    dataset,
    inputs,
    full_targets,
    inp_dir,
    save_model,
    hidden_dim,
    lr,
    epochs,
    batch_size,
):
    """Train MLP model for ALL traits simultaneously."""
    if dataset == "kaggle":
        trait_labels = ["E", "N", "F", "J"]
    else:
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    n_traits = len(trait_labels)
    n_splits = 10

    expdata = {"trait": [], "fold": [], "acc": []}

    # KFold for multi-label split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    from utils.log_utils import HistoryLogger

    # Create log directory
    log_dir = f"logs/MLP_LM_multitrait_{dataset}_{embed}_{jobid if 'jobid' in locals() else 0}"
    logger = HistoryLogger(log_dir)

    best_model = None
    best_avg_acc = 0.0

    print(f"Training Multi-Trait MLP on {dataset} traits: {trait_labels}")

    for fold, (train_index, test_index) in enumerate(kf.split(inputs, full_targets), 1):
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = full_targets[train_index], full_targets[test_index]

        # Multi-label model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(50, input_dim=hidden_dim, activation="relu"))
        # Output layer: n_traits units with sigmoid activation for multi-label binary classification
        model.add(tf.keras.layers.Dense(n_traits, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
        )

        fold_key = f"fold{fold}"

        for epoch in range(epochs):
            # Train for one epoch
            history = model.fit(
                x_train,
                y_train,
                epochs=1,
                batch_size=batch_size,
                verbose=0,
            )
            train_loss = history.history["loss"][0]

            # Predict to get per-trait metrics
            preds_probs = model.predict(x_test, verbose=0)
            preds_binary = np.round(preds_probs)

            # Calculate validation loss manually or via evaluate
            # model.evaluate returns [loss, bin_acc]
            val_metrics = model.evaluate(x_test, y_test, verbose=0)
            val_loss = val_metrics[0]

            # Calculate accuracy per trait
            acc_per_trait = np.mean(preds_binary == y_test, axis=0)  # [n_traits]
            avg_acc = np.mean(acc_per_trait)

            # Log
            logger.log_epoch(
                fold, epoch + 1, train_loss, val_loss, avg_acc, acc_per_trait
            )

            print(
                f"Ep {epoch+1}: T_Loss={train_loss:.4f}, V_Loss={val_loss:.4f}, V_Acc (Avg)={avg_acc:.4f}"
            )

        # Final evaluation for this fold
        preds_probs = model.predict(x_test, verbose=0)
        preds_binary = np.round(preds_probs)
        acc_per_trait = np.mean(preds_binary == y_test, axis=0)
        avg_acc = np.mean(acc_per_trait)

        print(f"Fold {fold} - Avg Acc: {avg_acc:.4f} - Per trait: {acc_per_trait}")

        for i, trait in enumerate(trait_labels):
            expdata["trait"].append(trait)
            expdata["fold"].append(fold)
            expdata["acc"].append(acc_per_trait[i] * 100)

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_model = model

        # Save logs for this fold
        logger.save_logs(f"logs_fold_{fold}.json")
        logger.plot_curves(f"curves_fold{fold}")

        # Clean logs for next fold to avoid accumulation issues if plot_curves uses all logs
        # Actually plot_curves plots all items in self.logs.
        # If we want separate plots per fold (lines 1-10 epochs), we should clear.
        # But maybe we want one big plot with Fold 1, Fold 2...
        # The user asked 'similar to partial multi-trait'. Partial multi-trait clears or saves?
        # In partial fit, I did: logger.save_logs(..fold..); logger.plot_curves(..fold..).
        # And I did NOT clear logs there. So it accumulates. Fold 2 plot will have Fold 1 and Fold 2 curves.
        # That is acceptable or even desired.

    # Save best model
    if str(save_model).lower() == "yes" and best_model is not None:
        path = inp_dir + "finetune_mlp_lm_multitrait"
        Path(path).mkdir(parents=True, exist_ok=True)
        best_model.save(f"{path}/MLP_LM_MultiTrait_{dataset}.h5")
        print(f"Saved best model to {path}/MLP_LM_MultiTrait_{dataset}.h5")

    # Save aggregated logs
    logger.save_logs("final_logs.json")

    df = pd.DataFrame(expdata)
    return df


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
    ) = utils.parse_args()

    # Pre-determined hidden dims based on embed model name (simplified logic)
    if "base" in embed:
        hidden_dim = 768
    elif "large" in embed:
        hidden_dim = 1024
    else:
        hidden_dim = 768  # fallback

    print(f"Starting Multi-trait MLP training for {dataset} with {embed}...")

    inputs, full_targets = get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer)
    df = training(
        dataset,
        inputs,
        full_targets,
        inp_dir,
        save_model,
        hidden_dim,
        lr,
        epochs,
        batch_size,
    )

    df.to_csv("expdata_multitrait_mlp.csv")
    print("\nAverage Accuracy per Trait:")
    print(df.groupby("trait")["acc"].mean())
