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

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=0,
        )

        # Evaluate
        # model.evaluate returns [loss, binary_accuracy]
        # BUT binary_accuracy in Keras computes accuracy over all elements flattened
        # We want per-trait accuracy if possible, or we leverage the global metric.
        # Let's predict and calculate manually for clarity.

        preds_probs = model.predict(x_test, verbose=0)
        preds_binary = np.round(preds_probs)

        # Calculate accuracy per trait
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

    # Save best model
    if str(save_model).lower() == "yes" and best_model is not None:
        path = inp_dir + "finetune_mlp_lm_multitrait"
        Path(path).mkdir(parents=True, exist_ok=True)
        best_model.save(f"{path}/MLP_LM_MultiTrait_{dataset}.h5")
        print(f"Saved best model to {path}/MLP_LM_MultiTrait_{dataset}.h5")

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
