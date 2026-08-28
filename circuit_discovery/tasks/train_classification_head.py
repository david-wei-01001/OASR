"""
circuit_discovery/tasks/train_classification_head.py

Plain supervised training for a CircuitHubertClassifier's head (backbone
frozen by default). This is deliberately separate from circuit discovery:
DiscoGP/ACDC/EAP/edge_pruning never touch model weights, only their own
edge_logits/weight_logits gating parameters -- so the head has to already
be a trained, fixed thing *before* circuit discovery starts. This script
produces that fixed thing and saves it; circuit discovery loads it back via
CircuitHubertClassifier.load_head and never imports this file.

Nothing here touches configs.yaml, run.py, or algorithms/*.py. Run it
directly, e.g.:

    python -m circuit_discovery.tasks.train_classification_head \
        --task_type consonant_classification --n_epochs 20

or from a notebook:

    from circuit_discovery.tasks.train_classification_head import train_classification_head
    result = train_classification_head("consonant_classification", n_epochs=20)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..metrics import evaluate_classification_accuracy
from ..models import load_circuit_model
from ..utils import DATASET_FOLDER_PATH, pick_device
from .articulatory_index import TASK_SPECS, load_articulatory_dataset
from .hubert_classifier import CircuitHubertClassifier


DEVICE = pick_device()
print(DEVICE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HEAD_CHECKPOINT_DIR = DATASET_FOLDER_PATH.parent / "trained_heads"


def train_classification_head(
    task_type: str,
    *,
    model_name: str = "hubert-base-ls960",
    batch_size: int = 32,
    n_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    eval_every: int = 1,
    freeze_backbone: bool = True,
    device: str | None = None,
    save_dir: str | Path | None = None,
    backbone: Any = None,  # pass an already-loaded CircuitHubert to skip reloading it
) -> dict[str, Any]:
    """
    Train a linear classification head on top of a (by default, frozen)
    CircuitHubert backbone. Returns a dict with the trained classifier, the
    checkpoint path it was saved to, and the final train/test accuracy.
    """
    if task_type not in TASK_SPECS:
        raise ValueError(f"unknown task_type {task_type!r}; expected one of {sorted(TASK_SPECS)}.")
    _label_extractor, num_classes, class_names = TASK_SPECS[task_type]

    device = device or DEVICE
    backbone = backbone if backbone is not None else load_circuit_model(model_name, device=device)

    model = CircuitHubertClassifier(
        backbone, num_classes=num_classes, freeze_backbone=freeze_backbone, device=device,
    )

    data = load_articulatory_dataset(task_type, batch_size=batch_size)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "no trainable parameters -- freeze_backbone=True should still leave "
            "the head's parameters trainable; check CircuitHubertClassifier construction."
        )
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    best_test_acc = -1.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in data.train:
            input_values = batch["input_ids"].to(device=device)
            labels = batch["label"].to(device=device)

            # circuit=None here: training the head happens on the full,
            # unablated model -- there's no circuit being discovered or
            # tested yet, so skip the (equivalent-cost but conceptually
            # unnecessary) dense edge-mask machinery entirely.
            logits = model(input_values, circuit=None)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            eval_result = evaluate_classification_accuracy(
                model=model, dataloader=data.test, circuit=model.full_circuit,
            )
            test_acc = eval_result["acc"]
            logger.info(
                "[%s] epoch %d/%d  train_loss=%.4f  test_acc=%.4f",
                task_type, epoch + 1, n_epochs, mean_loss, test_acc,
            )
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.head.state_dict().items()}

    if best_state is not None:
        model.head.load_state_dict(best_state)

    save_dir = Path(save_dir) if save_dir is not None else DEFAULT_HEAD_CHECKPOINT_DIR
    save_path = save_dir / f"{task_type}_head.pt"
    model.save_head(
        save_path,
        task_type=task_type,
        class_names=class_names,
        extra={"test_acc": best_test_acc, "n_epochs": n_epochs, "lr": lr, "batch_size": batch_size},
    )
    logger.info("Saved %s head (test_acc=%.4f) -> %s", task_type, best_test_acc, save_path)

    return {
        "model": model,
        "checkpoint_path": save_path,
        "test_acc": best_test_acc,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_type", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--model_name", default="hubert-base-ls960")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fine_tune_backbone", action="store_true")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    train_classification_head(
        args.task_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone=not args.fine_tune_backbone,
        save_dir=args.save_dir,
        device=DEVICE,
    )


if __name__ == "__main__":
    _main()
