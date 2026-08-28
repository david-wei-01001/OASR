"""
circuit_discovery/tasks/hubert_classifier.py

A classification head for CircuitHubert, attached *outside* the circuit's
own node accounting -- the same way GPT2's ln_final+unembed sits outside
CircuitGPT's node graph. CircuitHubert.forward returns raw final hidden
states ([batch, seq, d_model]); nothing in circuit.py, algorithms/*.py, or
CircuitHubert itself needs to know a classification head exists.

CircuitHubertClassifier wraps a CircuitHubert backbone and exposes the exact
same forward(input_values, circuit=..., runtime_masks=..., edge_intervention=...)
signature, so every algorithm file's `self.model(batch["input_ids"], circuit=circuit)`
call works completely unmodified: it now returns [batch, num_classes] logits
instead of [batch, seq, d_model] hidden states, and every circuit-bookkeeping
method (full_circuit, finalize_circuit, lookup_weight, edge_logit_group_specs,
weight_logit_group_specs, sample_runtime_masks, boolean_runtime_weight_masks)
is a pure delegation to the wrapped backbone -- the wrapper adds no new nodes
or edges of its own.

Pooling: mean over the time axis, unmasked. Because the dataset pads every
clip to one fixed sample count (see articulatory_index.py) rather than
padding a batch to its own max length, there's no attention_mask to route
through the conv stack's frame-count reduction here -- padded silence at the
tail of longer clips gets averaged in along with real signal. That's an
accepted simplification for a first pass (isolated syllables are short and
similarly sized, so the dilution is small); if it turns out to matter, the
fix is to also carry each clip's true frame count through and mask before
pooling, not to change anything here about how CircuitHubert itself works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from ..circuit import Circuit, edge_key, node_key
from .. import utils as _utils
from ..utils import pick_device

DEVICE = pick_device()
print(DEVICE)


class HubertClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, device: str | None = None):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes, device=device or DEVICE)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled)


class CircuitHubertClassifier(nn.Module):
    def __init__(
        self,
        backbone: Any,  # CircuitHubert
        num_classes: int,
        *,
        freeze_backbone: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.head = HubertClassificationHead(
            backbone.cfg.d_model, num_classes, device=device or backbone.device_name
        )
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        Toggle for the fine-tune-vs-frozen ablation. Defaults to frozen
        (freeze_backbone=True at construction) -- call
        classifier.set_backbone_trainable(True) later to fine-tune, without
        rebuilding the wrapper or reloading weights.
        """
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def freeze_all(self) -> None:
        """
        Freeze backbone AND head. Use this after loading a trained head for
        circuit discovery: DiscoGP/ACDC/EAP/edge_pruning never take a
        gradient step on model weights themselves (only on their own
        separate edge_logits/weight_logits nn.Parameters that gate the
        forward pass), so both the backbone and the already-trained head
        should be fully frozen at that point -- there's nothing left in
        this wrapper that circuit discovery should be updating.
        """
        for p in self.parameters():
            p.requires_grad = False

    # ---- save / load: the head is small (d_model x num_classes), the
    # backbone is not (and is loaded independently via load_circuit_model
    # each time anyway) -- so only the head's state gets persisted here,
    # bundled with enough metadata to reconstruct a matching wrapper without
    # the caller needing to remember num_classes/task_type by hand. ----

    def save_head(
        self,
        path: str | Path,
        *,
        task_type: str | None = None,
        class_names: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = {
            "num_classes": self.num_classes,
            "d_model": self.backbone.cfg.d_model,
            "model_name": getattr(self.backbone.cfg, "arch_name", None),
            "task_type": task_type,
            "class_names": class_names,
            "head_state_dict": self.head.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load_head(
        cls,
        path: str | Path,
        backbone: Any,  # CircuitHubert
        *,
        freeze: bool = True,
        device: str | None = None,
        map_location: str | None = "cpu",
    ) -> "CircuitHubertClassifier":
        checkpoint = torch.load(path, map_location=map_location)

        if checkpoint["d_model"] != backbone.cfg.d_model:
            raise ValueError(
                f"checkpoint at {path} was trained with d_model={checkpoint['d_model']}, "
                f"but the given backbone has d_model={backbone.cfg.d_model} -- "
                "these are almost certainly different model_name checkpoints."
            )

        classifier = cls(
            backbone,
            num_classes=checkpoint["num_classes"],
            freeze_backbone=True,
            device=device,
        )
        classifier.head.load_state_dict(checkpoint["head_state_dict"])

        if freeze:
            classifier.freeze_all()

        classifier.loaded_task_type = checkpoint.get("task_type")
        classifier.loaded_class_names = checkpoint.get("class_names")

        return classifier

    # ---- delegation: every circuit-bookkeeping method passes straight
    # through to the backbone, unchanged, so algorithms/*.py needs no
    # awareness that a head exists. ----

    @property
    def full_circuit(self) -> Circuit:
        return self.backbone.full_circuit

    @property
    def cfg(self):
        return self.backbone.cfg

    @property
    def device_name(self) -> str:
        return self.backbone.device_name

    def finalize_circuit(self, circuit: Circuit) -> Circuit:
        return self.backbone.finalize_circuit(circuit)

    def lookup_weight(self, n_key: node_key, w_key: str) -> torch.Tensor:
        return self.backbone.lookup_weight(n_key, w_key)

    def edge_logit_group_specs(self, circuit: Circuit):
        return self.backbone.edge_logit_group_specs(circuit)

    def weight_logit_group_specs(self, circuit: Circuit):
        return self.backbone.weight_logit_group_specs(circuit)

    def sample_runtime_masks(self, **kwargs):
        return self.backbone.sample_runtime_masks(**kwargs)

    def boolean_runtime_weight_masks(self, **kwargs):
        return self.backbone.boolean_runtime_weight_masks(**kwargs)

    # ---- the one method that actually differs from the backbone ----

    def forward(
        self,
        input_values: torch.Tensor,
        circuit: Circuit | None = None,
        *,
        runtime_masks=None,
        edge_intervention=None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_values,
            circuit,
            runtime_masks=runtime_masks,
            edge_intervention=edge_intervention,
            return_residual=return_residual,
        )
        pooled = hidden.mean(dim=1)
        return self.head(pooled)
