# modeling_hubert.py
#
# circuit-oriented modeling of post-LN speech transformers: HuBERT (base,
# do_stable_layer_norm=False) and Wav2Vec2 (base), which share an identical
# encoder-layer architecture in HuggingFace `transformers`.
#
# ----------------------------------------------------------------------------
# why this file looks different from modeling_gpt.py
# ----------------------------------------------------------------------------
# GPT2 is pre-LN: LayerNorm sits on a read-only branch, the residual stream
# itself is a plain, never-renormalized sum, so every node's contribution can
# be kept as a separate, un-merged slice for the entire depth of the network
# (see `residual: [batch, pos, n_sources, d_model]` in modeling_gpt.py).
#
# HuBERT/Wav2Vec2 (base config) are post-LN:
#     hidden = LayerNorm(attn_residual + Attention(attn_residual))
#     hidden = LayerNorm(hidden + FeedForward(hidden))
# LayerNorm here *is* the thing that produces the tensor which gets carried
# forward, so the residual stream physically merges at every layer boundary.
# To keep every original node (an attention head's output, the MLP's output,
# ...) independently addressable by every later layer -- the same "residual
# highway to all downstream nodes" property GPT2 gets for free -- we have to
# decompose each LayerNorm's output back into per-source pieces immediately
# after computing it, and keep propagating that decomposed list forward
# instead of a single merged tensor.
#
# The decomposition is *exact*, not an approximation, because it is computed
# fresh from the real (possibly edge-masked) tensor on every forward call:
#
#     x_i   = mask_i * source_i                      (a source's gated value)
#     x     = sum_i x_i                               (what LayerNorm actually sees)
#     mu    = mean_last_dim(x)      = sum_i mean_last_dim(x_i)      (mean is linear)
#     var   = variance_last_dim(x)                    (the only nonlinear part)
#     y_i   = gamma * (x_i - mean_last_dim(x_i)) / sqrt(var + eps)
#     bias  = beta
#
#     ==>  sum_i y_i + bias  ==  LayerNorm(x)   (exact identity, for any x_i)
#
# `var` is the only place information from *every* source mixes together (a
# nonlinear coupling across sources -- see the write-up this file accompanies
# for the full derivation), but since we always compute it fresh from the
# actual gathered sum in this forward call, autograd differentiates through
# the real thing; nothing is "frozen" or hand-linearized.
#
# One structural consequence worth remembering: because `y_i` depends on
# whether source i's edge *into this LayerNorm* is on, turning that edge off
# genuinely severs source i's ability to reach anything later in the network
# via the residual highway -- there is no other cross-layer channel in the
# real model. Pre-LN circuits don't have this "staged reachability" property;
# post-LN circuits do. This file treats `ln1`/`ln2` gathers as ordinary,
# genuinely prunable destination nodes so that discovery can represent this
# fact rather than hide it.

from __future__ import annotations

from ..utils import DEVICE

import copy
import math
from dataclasses import dataclass
from typing import Callable, cast

import torch
import torch.nn as nn

from ..circuit import (
    Circuit,
    Edge,
    Node,
    edge_key,
    node_key,
    create_circuit_from_nodes_and_edges,
)

from .modeling_gpt import (
    EdgeIntervention,
    EdgeLogitGroupSpec,
    WeightLogitGroupSpec,
    apply_gate,
    apply_weight_mask,
    source_axis,
    attention_keys,
    gather_for_dst,
    gather_for_dsts_dense,
    gather_for_dst_dense,
    dense_edge_mask_for_dsts,
    apply_node_gates_many,
    apply_edge_intervention,
)

# --------------------------------------------------------------------------------------
# HF architecture registry
# --------------------------------------------------------------------------------------
# Hubert (do_stable_layer_norm=False) and Wav2Vec2 (do_stable_layer_norm=False)
# share an identical inner module layout in `transformers`:
#     <root>.feature_extractor
#     <root>.feature_projection
#     <root>.encoder.pos_conv_embed
#     <root>.encoder.layer_norm
#     <root>.encoder.layers[i].attention.{q_proj,k_proj,v_proj,out_proj}
#     <root>.encoder.layers[i].layer_norm            # LN1, post-attention
#     <root>.encoder.layers[i].feed_forward.{intermediate_dense,output_dense}
#     <root>.encoder.layers[i].final_layer_norm       # LN2, post-feed-forward
# so one implementation covers both; only the HF class name, the checkpoint
# id, and the top-level attribute name differ.

@dataclass(frozen=True)
class HFArchSpec:
    hf_checkpoint: str
    hf_module: str
    hf_class_name: str
    root_attr: str


HF_ARCH_SPECS: dict[str, HFArchSpec] = {
    "hubert-base-ls960": HFArchSpec(
        hf_checkpoint="facebook/hubert-base-ls960",
        hf_module="transformers",
        hf_class_name="HubertModel",
        root_attr="hubert",
    ),
    "wav2vec2-base-960h": HFArchSpec(
        hf_checkpoint="facebook/wav2vec2-base-960h",
        hf_module="transformers",
        hf_class_name="Wav2Vec2Model",
        root_attr="wav2vec2",
    ),
    "wav2vec2-base": HFArchSpec(
        hf_checkpoint="facebook/wav2vec2-base",
        hf_module="transformers",
        hf_class_name="Wav2Vec2Model",
        root_attr="wav2vec2",
    ),
}

# --------------------------------------------------------------------------------------
# minimal cfg object (mirrors the fields modeling_gpt.py pulls off TransformerLens's cfg)
# --------------------------------------------------------------------------------------

@dataclass
class HubertCircuitConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    layer_norm_eps: float
    hidden_act: str
    arch_name: str
    conv_kernel: list[int]
    conv_stride: list[int]


# --------------------------------------------------------------------------------------
# weight masks (mirrors GPTAttentionWeightMasks / GPTMLPWeightMasks)
# --------------------------------------------------------------------------------------

@dataclass
class HubertAttentionWeightMasks:
    W_Q: torch.Tensor | None = None
    b_Q: torch.Tensor | None = None
    W_K: torch.Tensor | None = None
    b_K: torch.Tensor | None = None
    W_V: torch.Tensor | None = None
    b_V: torch.Tensor | None = None
    W_O: torch.Tensor | None = None
    b_O: torch.Tensor | None = None


@dataclass
class HubertMLPWeightMasks:
    W_in: torch.Tensor | None = None
    b_in: torch.Tensor | None = None
    W_out: torch.Tensor | None = None
    b_out: torch.Tensor | None = None


@dataclass
class HubertWeightMasks:
    attention: tuple[HubertAttentionWeightMasks, ...]
    mlp: tuple[HubertMLPWeightMasks, ...]


@dataclass
class HubertEdgeMasks:
    """
    Dense edge masks for the Hubert execution layout.

    One mask vector per single-destination gather point: attention qkv (shared
    source set, per modeling_gpt.py's convention), ln1, mlp, ln2 per layer,
    plus the final output gather.
    """

    attention_qkv: tuple[torch.Tensor, ...]
    ln1: tuple[torch.Tensor, ...]
    mlp: tuple[torch.Tensor, ...]
    ln2: tuple[torch.Tensor, ...]
    output: torch.Tensor


@dataclass
class HubertRuntimeMasks:
    edge_masks: HubertEdgeMasks | None = None
    weight_masks: HubertWeightMasks | None = None


ATTENTION_NODE_WEIGHT_KEYS: dict[str, tuple[str, ...]] = {
    "attn_q": ("W_Q", "b_Q"),
    "attn_k": ("W_K", "b_K"),
    "attn_v": ("W_V", "b_V"),
    "attn_o": ("W_O", "b_O"),
}
MLP_WEIGHT_KEYS = ("W_in", "b_in", "W_out", "b_out")

# --------------------------------------------------------------------------------------
# circuit construction
# --------------------------------------------------------------------------------------
# Per layer l, the source list grows by:
#     attn_o (l, head, "attn_o")   for each head          -- attention output
#     (l, 0, "ln1_bias")                                   -- LN1's beta term
#     (l, 0, "mlp")                                         -- feed-forward output
#     (l, 0, "ln2_bias")                                   -- LN2's beta term
# and two new destination-only gather points are added:
#     (l, 0, "ln1")  gathers {pre_srcs(l)} u {attn_o(l, *)}
#     (l, 0, "ln2")  gathers {post_ln1_srcs(l)} u {(l,0,"mlp")}
# "mlp" doubles as a destination (its own gather, the feed-forward's input)
# and, once computed, a source -- exactly the dual role modeling_gpt.py's
# "mlp" node already plays for GPT2.

def hubert_src_keys_before_layer(cfg: HubertCircuitConfig, layer_id: int) -> list[node_key]:
    keys: list[node_key] = [(-1, 0, "emb")]

    for layer in range(layer_id):
        for head in range(cfg.n_heads):
            keys.append((layer, head, "attn_o"))
        keys.append((layer, 0, "ln1_bias"))
        keys.append((layer, 0, "mlp"))
        keys.append((layer, 0, "ln2_bias"))

    return keys


def get_feat_extract_output_lengths(cfg: "HubertCircuitConfig", input_lengths: torch.Tensor) -> torch.Tensor:
    lengths = input_lengths
    for kernel_size, stride in zip(cfg.conv_kernel, cfg.conv_stride):
        lengths = torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1
    return lengths


def build_full_hubert_circuit(cfg: HubertCircuitConfig) -> Circuit:
    nodes: list[Node] = []
    edges: list[Edge] = []

    emb = Node(layer=-1, index=0, kind="emb")
    nodes.append(emb)

    for layer in range(cfg.n_layers):
        pre_srcs = hubert_src_keys_before_layer(cfg, layer)

        for head in range(cfg.n_heads):
            for kind, weight_keys in ATTENTION_NODE_WEIGHT_KEYS.items():
                node = Node(
                    layer=layer,
                    index=head,
                    kind=kind,
                    weight_masks={key: None for key in weight_keys},
                )
                nodes.append(node)

                if kind != "attn_o":
                    for src in pre_srcs:
                        edges.append(Edge(dst=node.key, src=src))

        attn_full_srcs = pre_srcs + [
            (layer, head, "attn_o") for head in range(cfg.n_heads)
        ]

        ln1 = Node(layer=layer, index=0, kind="ln1")
        nodes.append(ln1)
        for src in attn_full_srcs:
            edges.append(Edge(dst=ln1.key, src=src))

        ln1_bias = Node(layer=layer, index=0, kind="ln1_bias")
        nodes.append(ln1_bias)

        post_ln1_srcs = attn_full_srcs + [(layer, 0, "ln1_bias")]

        mlp = Node(
            layer=layer,
            index=0,
            kind="mlp",
            weight_masks={key: None for key in MLP_WEIGHT_KEYS},
        )
        nodes.append(mlp)
        for src in post_ln1_srcs:
            edges.append(Edge(dst=mlp.key, src=src))

        ln2_srcs = post_ln1_srcs + [(layer, 0, "mlp")]

        ln2 = Node(layer=layer, index=0, kind="ln2")
        nodes.append(ln2)
        for src in ln2_srcs:
            edges.append(Edge(dst=ln2.key, src=src))

        ln2_bias = Node(layer=layer, index=0, kind="ln2_bias")
        nodes.append(ln2_bias)

    output = Node(layer=cfg.n_layers, index=0, kind="output")
    nodes.append(output)

    for src in hubert_src_keys_before_layer(cfg, cfg.n_layers):
        edges.append(Edge(dst=output.key, src=src))

    return create_circuit_from_nodes_and_edges(nodes, edges)


def finalize_hubert_circuit(circuit: Circuit) -> Circuit:
    """
    Drop masks for edges/nodes that cannot affect the output node, and clone.

    Mirrors modeling_gpt.py's finalize_gpt_circuit. source_dependencies encode
    the architecture-specific fact that a source is only computable if its
    "generating" destination is kept:
        attn_o(l,h) needs attn_q/k/v(l,h) kept
        ln1_bias(l) needs ln1(l) kept        (it's ln1's own bias term)
        mlp(l)      needs mlp(l) [dst] kept   (self-referential, as in GPT2)
        ln2_bias(l) needs ln2(l) kept
    """
    if len(circuit.nodes) == 0:
        return circuit.clone()

    output_nodes = [key for key in circuit.nodes if key[2] == "output"]
    if len(output_nodes) == 0:
        return circuit.clone()

    output_key = max(output_nodes, key=lambda x: x[0])
    n_layers = output_key[0]
    n_heads = (
        max((key[1] for key in circuit.nodes if key[2] == "attn_o"), default=-1) + 1
    )

    source_dependencies: dict[node_key, tuple[node_key, ...]] = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            o = (layer, head, "attn_o")
            if o in circuit.nodes and circuit.nodes[o].is_kept():
                source_dependencies[o] = tuple(
                    key
                    for key in (
                        (layer, head, "attn_q"),
                        (layer, head, "attn_k"),
                        (layer, head, "attn_v"),
                    )
                    if key in circuit.nodes and circuit.nodes[key].is_kept()
                )

        ln1_bias = (layer, 0, "ln1_bias")
        if ln1_bias in circuit.nodes and circuit.nodes[ln1_bias].is_kept():
            ln1 = (layer, 0, "ln1")
            source_dependencies[ln1_bias] = (
                (ln1,) if ln1 in circuit.nodes and circuit.nodes[ln1].is_kept() else ()
            )

        mlp = (layer, 0, "mlp")
        if mlp in circuit.nodes and circuit.nodes[mlp].is_kept():
            source_dependencies[mlp] = (mlp,)

        ln2_bias = (layer, 0, "ln2_bias")
        if ln2_bias in circuit.nodes and circuit.nodes[ln2_bias].is_kept():
            ln2 = (layer, 0, "ln2")
            source_dependencies[ln2_bias] = (
                (ln2,) if ln2 in circuit.nodes and circuit.nodes[ln2].is_kept() else ()
            )

    needed_dsts: set[node_key] = {output_key}
    needed_sources: set[node_key] = set()
    dst_worklist: list[node_key] = [output_key]
    src_worklist: list[node_key] = []
    expanded_dsts: set[node_key] = set()
    expanded_sources: set[node_key] = set()

    while dst_worklist or src_worklist:
        while dst_worklist:
            dst = dst_worklist.pop()
            if dst in expanded_dsts:
                continue
            expanded_dsts.add(dst)

            for edge in circuit.incoming_edges_of(dst):
                if edge.is_kept() and edge.src not in needed_sources:
                    needed_sources.add(edge.src)
                    src_worklist.append(edge.src)

        while src_worklist:
            src = src_worklist.pop()
            if src in expanded_sources:
                continue
            expanded_sources.add(src)

            for dst in source_dependencies.get(src, ()):
                if dst not in needed_dsts:
                    needed_dsts.add(dst)
                    dst_worklist.append(dst)

    false_mask = torch.tensor(False, dtype=torch.bool, device=DEVICE)
    out = circuit.clone()

    for edge in out.all_edges():
        keep = (
            edge.is_kept()
            and edge.dst in needed_dsts
            and edge.src in needed_sources
        )
        if not keep:
            edge.edge_mask = false_mask.clone()

    for key, node in out.nodes.items():
        live = key in needed_sources if node.is_src() else key in needed_dsts
        if not live:
            node.node_mask = false_mask.clone()

    return out

# --------------------------------------------------------------------------------------
# the exact post-LN decomposition
# --------------------------------------------------------------------------------------

def decompose_post_layer_norm(
    per_source: torch.Tensor,
    *,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    per_source must already be the resolved (edge-masked OR intervened)
    per-source tensor -- callers are responsible for that, since masking
    and intervention need different logic upstream of this function.
    """
    total = per_source.sum(dim=2)
    var = total.var(dim=-1, unbiased=False, keepdim=True)
    per_source_mean = per_source.mean(dim=-1, keepdim=True)
    denom = torch.sqrt(var + eps).unsqueeze(2)
    weight = ln_weight.to(device=per_source.device, dtype=per_source.dtype)

    decomposed = weight * (per_source - per_source_mean) / denom
    return decomposed

def per_source_for_ln(
    *,
    circuit: Circuit | None,
    residual: torch.Tensor,
    src_keys: list[node_key],
    dst: node_key,
    edge_intervention: EdgeIntervention | None,
) -> torch.Tensor:
    """
    Per-source contributions feeding an ln1/ln2 gather. Mirrors
    gather_for_dst's edge-driven traversal so edge_intervention's
    replacement values are actually honored here too -- but keeps every
    source separate (summing happens inside decompose_post_layer_norm)
    since each source needs its own mean subtracted individually.
    """
    if circuit is None:
        return residual

    if edge_intervention is None:
        mask = dense_edge_mask_for_dsts(
            circuit=circuit, src_keys=src_keys, dst_keys=[dst],
            device=residual.device, dtype=residual.dtype,
        )[:, 0]
        return residual * mask.view(1, 1, -1, 1)

    src_index = {src: i for i, src in enumerate(src_keys)}
    pieces: list[torch.Tensor | None] = [None] * len(src_keys)
    for edge in circuit.incoming_edges_of(dst):
        if edge.src not in src_index:
            continue
        i = src_index[edge.src]
        pieces[i] = apply_edge_intervention(
            edge=edge, current_src=residual[:, :, i, :], dst=dst, src=edge.src,
            edge_intervention=edge_intervention,
        )
    zero_like = residual[:, :, 0, :]
    return torch.stack(
        [p if p is not None else torch.zeros_like(zero_like) for p in pieces],
        dim=2,
    )


def bias_source(
    reference: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a (d_model,) bias into a [batch, pos, 1, d_model] source slice."""
    b, p, _, d = reference.shape
    return bias.to(device=reference.device, dtype=reference.dtype).view(1, 1, 1, d).expand(b, p, 1, d)

# --------------------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------------------

class HubertMultiHeadAttention(nn.Module):
    """
    Bidirectional (no causal mask) multi-head self-attention, weights kept in
    per-head layout (n_heads, d_model, d_head) so individual heads can be
    independently weight-masked, matching modeling_gpt.py's convention.
    """

    def __init__(self, cfg: HubertCircuitConfig, device: str):
        super().__init__()
        self.cfg = cfg

        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device))
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device))
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device))
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model), device=device))
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model,), device=device))

    def _masked_projection_weights(
        self,
        circuit: Circuit | None,
        layer_id: int,
        weight_masks: HubertAttentionWeightMasks | None = None,
    ):
        if weight_masks is not None:
            def rm(weight, mask):
                if mask is None:
                    return weight
                return weight * mask.to(device=weight.device, dtype=weight.dtype)

            b_O = (
                self.b_O
                if weight_masks.b_O is None
                else self.b_O.unsqueeze(0) * weight_masks.b_O.to(device=self.b_O.device, dtype=self.b_O.dtype)
            )
            return (
                rm(self.W_Q, weight_masks.W_Q), rm(self.b_Q, weight_masks.b_Q),
                rm(self.W_K, weight_masks.W_K), rm(self.b_K, weight_masks.b_K),
                rm(self.W_V, weight_masks.W_V), rm(self.b_V, weight_masks.b_V),
                rm(self.W_O, weight_masks.W_O), b_O,
            )

        default = (self.W_Q, self.b_Q, self.W_K, self.b_K, self.W_V, self.b_V, self.W_O, self.b_O)
        if circuit is None:
            return default

        has_explicit = False
        for head in range(self.cfg.n_heads):
            for kind in ("attn_q", "attn_k", "attn_v", "attn_o"):
                masks = circuit.nodes[(layer_id, head, kind)].weight_masks
                if any(m is not None for m in masks.values()):
                    has_explicit = True
                    break
            if has_explicit:
                break
        if not has_explicit:
            return default

        def mask_heads(weight, *, kind, w_key):
            masks = [
                circuit.nodes[(layer_id, head, kind)].weight_masks.get(w_key)
                for head in range(self.cfg.n_heads)
            ]
            if all(m is None for m in masks):
                return weight
            gates = torch.stack(
                [
                    torch.ones_like(weight[head]) if m is None else m.to(device=weight.device, dtype=weight.dtype)
                    for head, m in enumerate(masks)
                ],
                dim=0,
            )
            return weight * gates

        def mask_shared_bias(bias, *, kind, w_key):
            masks = [
                circuit.nodes[(layer_id, head, kind)].weight_masks.get(w_key)
                for head in range(self.cfg.n_heads)
            ]
            if all(m is None for m in masks):
                return bias
            gates = torch.stack(
                [
                    torch.ones_like(bias) if m is None else m.to(device=bias.device, dtype=bias.dtype)
                    for m in masks
                ],
                dim=0,
            )
            return bias.unsqueeze(0) * gates

        return (
            mask_heads(self.W_Q, kind="attn_q", w_key="W_Q"),
            mask_heads(self.b_Q, kind="attn_q", w_key="b_Q"),
            mask_heads(self.W_K, kind="attn_k", w_key="W_K"),
            mask_heads(self.b_K, kind="attn_k", w_key="b_K"),
            mask_heads(self.W_V, kind="attn_v", w_key="W_V"),
            mask_heads(self.b_V, kind="attn_v", w_key="b_V"),
            mask_heads(self.W_O, kind="attn_o", w_key="W_O"),
            mask_shared_bias(self.b_O, kind="attn_o", w_key="b_O"),
        )

    def forward(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        *,
        circuit: Circuit | None,
        layer_id: int,
        weight_masks: HubertAttentionWeightMasks | None = None,
    ) -> torch.Tensor:
        W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O = self._masked_projection_weights(
            circuit, layer_id, weight_masks
        )

        q = torch.einsum("bphd,hde->bphe", q_input, W_Q) + b_Q
        k = torch.einsum("bphd,hde->bphe", k_input, W_K) + b_K
        v = torch.einsum("bphd,hde->bphe", v_input, W_V) + b_V

        # no causal mask: HuBERT/Wav2Vec2 self-attention is bidirectional
        attn_scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(self.cfg.d_head)
        pattern = attn_scores.softmax(dim=-1)

        z = torch.einsum("bhqk,bkhe->bqhe", pattern, v)
        attn_out = torch.einsum("bqhe,hed->bqhd", z, W_O) + (b_O / self.cfg.n_heads)

        if circuit is not None:
            for head in range(self.cfg.n_heads):
                key = (layer_id, head, "attn_o")
                attn_out[:, :, head, :] = apply_gate(attn_out[:, :, head, :], circuit.nodes[key].node_mask)

        return attn_out

# --------------------------------------------------------------------------------------
# feed forward
# --------------------------------------------------------------------------------------

_ACT_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "gelu_new": lambda x: nn.functional.gelu(x, approximate="tanh"),
    "swish": nn.functional.silu,
    "silu": nn.functional.silu,
}


class HubertFeedForward(nn.Module):
    def __init__(self, cfg: HubertCircuitConfig, device: str):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp), device=device))
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp,), device=device))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model), device=device))
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model,), device=device))
        self.act = _ACT_FNS.get(cfg.hidden_act, nn.functional.gelu)

    def forward(
        self,
        x: torch.Tensor,
        weight_masks: HubertMLPWeightMasks | dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor:
        if weight_masks is None:
            W_in, b_in, W_out, b_out = self.W_in, self.b_in, self.W_out, self.b_out
        else:
            if isinstance(weight_masks, HubertMLPWeightMasks):
                masks = {
                    "W_in": weight_masks.W_in, "b_in": weight_masks.b_in,
                    "W_out": weight_masks.W_out, "b_out": weight_masks.b_out,
                }
            else:
                masks = weight_masks
            if all(m is None for m in masks.values()):
                W_in, b_in, W_out, b_out = self.W_in, self.b_in, self.W_out, self.b_out
            else:
                W_in = apply_weight_mask(self.W_in, masks.get("W_in"))
                b_in = apply_weight_mask(self.b_in, masks.get("b_in"))
                W_out = apply_weight_mask(self.W_out, masks.get("W_out"))
                b_out = apply_weight_mask(self.b_out, masks.get("b_out"))

        hidden = self.act(x @ W_in + b_in)
        return hidden @ W_out + b_out

# --------------------------------------------------------------------------------------
# block
# --------------------------------------------------------------------------------------

class CircuitHubertBlock(nn.Module):
    def __init__(self, cfg: HubertCircuitConfig, layer_id: int, device: str):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.attn = HubertMultiHeadAttention(cfg, device=device)
        self.mlp = HubertFeedForward(cfg, device=device)

        self.ln1_weight = nn.Parameter(torch.ones((cfg.d_model,), device=device))
        self.ln1_bias_param = nn.Parameter(torch.zeros((cfg.d_model,), device=device))
        self.ln2_weight = nn.Parameter(torch.ones((cfg.d_model,), device=device))
        self.ln2_bias_param = nn.Parameter(torch.zeros((cfg.d_model,), device=device))

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        residual: torch.Tensor,
        src_keys: list[node_key],
        circuit: Circuit | None,
        *,
        weight_masks: HubertWeightMasks | None = None,
        edge_intervention: EdgeIntervention | None = None,
    ) -> tuple[torch.Tensor, list[node_key]]:
        cfg = self.cfg
        layer_id = self.layer_id

        q_keys = attention_keys(layer_id, cfg.n_heads, "attn_q")
        k_keys = attention_keys(layer_id, cfg.n_heads, "attn_k")
        v_keys = attention_keys(layer_id, cfg.n_heads, "attn_v")

        # --- attention input: raw gather, no LN (post-LN model) ---
        if edge_intervention is None:
            if circuit is None:
                full_input = residual.sum(dim=2)
                q_input = k_input = v_input = full_input.unsqueeze(2).expand(-1, -1, cfg.n_heads, -1)
            else:
                q_input, k_input, v_input = [
                    apply_node_gates_many(
                        gather_for_dsts_dense(circuit=circuit, residual=residual, src_keys=src_keys, dst_keys=keys),
                        circuit=circuit,
                        keys=keys,
                    )
                    for keys in (q_keys, k_keys, v_keys)
                ]
        else:
            if circuit is None:
                raise ValueError("circuit is required for edge interventions.")
            q_input, k_input, v_input = [
                torch.stack(
                    [
                        apply_gate(
                            gather_for_dst(
                                circuit=circuit, residual=residual, src_keys=src_keys,
                                dst=key, edge_intervention=edge_intervention,
                            ),
                            circuit.nodes[key].node_mask,
                        )
                        for key in keys
                    ],
                    dim=2,
                )
                for keys in (q_keys, k_keys, v_keys)
            ]

        attn_out = self.attn.forward(
            q_input, k_input, v_input,
            circuit=circuit, layer_id=layer_id,
            weight_masks=None if weight_masks is None else weight_masks.attention[layer_id],
        )

        residual = torch.cat([residual, attn_out], dim=2)
        src_keys = src_keys + [(layer_id, head, "attn_o") for head in range(cfg.n_heads)]

        # --- ln1: real LayerNorm over the current gated sum, decomposed back ---
        ln1_key = (layer_id, 0, "ln1")
        per_source1 = per_source_for_ln(
            circuit=circuit, residual=residual, src_keys=src_keys,
            dst=ln1_key, edge_intervention=edge_intervention,
        )
        decomposed = decompose_post_layer_norm(
            per_source1,
            ln_weight=self.ln1_weight, ln_bias=self.ln1_bias_param,
            eps=cfg.layer_norm_eps,
        )
        ln1_bias_slice = bias_source(decomposed, self.ln1_bias_param)
        residual = torch.cat([decomposed, ln1_bias_slice], dim=2)
        src_keys = src_keys + [(layer_id, 0, "ln1_bias")]

        # --- mlp: its own gather over the post-ln1 sources, no extra LN ---
        mlp_key = (layer_id, 0, "mlp")
        if edge_intervention is None:
            if circuit is None:
                mlp_input = residual.sum(dim=2)
            else:
                mlp_input = gather_for_dst_dense(circuit=circuit, residual=residual, src_keys=src_keys, dst=mlp_key)
        else:
            mlp_input = gather_for_dst(
                circuit=circuit, residual=residual, src_keys=src_keys,
                dst=mlp_key, edge_intervention=edge_intervention,
            )

        mlp_out = self.mlp.forward(
            mlp_input,
            weight_masks.mlp[layer_id] if weight_masks is not None
            else circuit.nodes[mlp_key].weight_masks if circuit is not None
            else None,
        )
        if circuit is not None:
            mlp_out = apply_gate(mlp_out, circuit.nodes[mlp_key].node_mask)

        residual = torch.cat([residual, source_axis(mlp_out)], dim=2)
        src_keys = src_keys + [mlp_key]

        # --- ln2: real LayerNorm over post-ln1 sources + mlp, decomposed back ---
        ln2_key = (layer_id, 0, "ln2")

        per_source2 = per_source_for_ln(
            circuit=circuit, residual=residual, src_keys=src_keys,
            dst=ln2_key, edge_intervention=edge_intervention,
        )
        decomposed2 = decompose_post_layer_norm(
            per_source2,
            ln_weight=self.ln2_weight, ln_bias=self.ln2_bias_param,
            eps=cfg.layer_norm_eps,
        )
        
        ln2_bias_slice = bias_source(decomposed2, self.ln2_bias_param)
        residual = torch.cat([decomposed2, ln2_bias_slice], dim=2)
        src_keys = src_keys + [(layer_id, 0, "ln2_bias")]

        return residual, src_keys

# --------------------------------------------------------------------------------------
# frozen preprocessing ("emb" lump: conv feature extractor + projection + pos conv + LN)
# --------------------------------------------------------------------------------------

class FrozenSpeechPreprocessor(nn.Module):
    """
    Wraps everything before the transformer stack as one frozen function,
    matching the user's framing: the CNN feature extractor, the feature
    projection, and the positional conv embedding (plus the encoder's own
    first LayerNorm, which is applied once here and never revisited) are
    treated as fixed data preprocessing, exactly like GPT2's wte+wpe lump.
    """

    def __init__(self, hf_root: nn.Module):
        super().__init__()
        self.feature_extractor = hf_root.feature_extractor
        self.feature_projection = hf_root.feature_projection
        self.pos_conv_embed = hf_root.encoder.pos_conv_embed
        self.layer_norm = hf_root.encoder.layer_norm

        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.feature_projection(extract_features)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states + self.pos_conv_embed(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

class CircuitHubert(nn.Module):
    def __init__(self, cfg: HubertCircuitConfig, preprocessor: FrozenSpeechPreprocessor, device: str | None = None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.device_name = device if device is not None else DEVICE

        self.preprocessor = preprocessor
        self.blocks = nn.ModuleList(
            [CircuitHubertBlock(self.cfg, layer_id, device=self.device_name) for layer_id in range(self.cfg.n_layers)]
        )

        self.full_circuit = build_full_hubert_circuit(self.cfg)
        self._validated_circuit_ids: set[int] = {id(self.full_circuit)}

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.to(device=self.device_name)

    def _assert_compatible_circuit(self, circuit: Circuit) -> None:
        circuit_id = id(circuit)
        if circuit_id in self._validated_circuit_ids:
            return
        self.full_circuit.assert_same_structure(circuit)
        self._validated_circuit_ids.add(circuit_id)

    def finalize_circuit(self, circuit: Circuit) -> Circuit:
        self._assert_compatible_circuit(circuit)
        return finalize_hubert_circuit(circuit)

    def embed_as_residual(
        self,
        input_values: torch.Tensor,
        circuit: Circuit | None,
    ) -> tuple[torch.Tensor, list[node_key]]:
        residual = self.preprocessor(input_values)
        emb_key = (-1, 0, "emb")
        if circuit is not None:
            residual = apply_gate(residual, circuit.nodes[emb_key].node_mask)
        return source_axis(residual), [emb_key]

    def forward(
        self,
        input_values: torch.Tensor,
        circuit: Circuit | None = None,
        *,
        runtime_masks: HubertRuntimeMasks | None = None,
        edge_intervention: EdgeIntervention | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        if runtime_masks is not None and edge_intervention is not None:
            raise ValueError("runtime_masks cannot be combined with edge_intervention.")
        if circuit is not None and runtime_masks is not None:
            raise ValueError("runtime_masks and circuit are mutually exclusive.")

        if runtime_masks is not None:
            return self.forward_runtime(input_values, runtime_masks=runtime_masks, return_residual=return_residual)

        if circuit is not None:
            self._assert_compatible_circuit(circuit)

        residual, src_keys = self.embed_as_residual(input_values, circuit)

        for block in self.blocks:
            block = cast(CircuitHubertBlock, block)
            residual, src_keys = block(residual, src_keys, circuit, edge_intervention=edge_intervention)

        output_key = (self.cfg.n_layers, 0, "output")

        if edge_intervention is None:
            if circuit is None:
                final_residual = residual.sum(dim=2)
            else:
                final_residual = gather_for_dst_dense(
                    circuit=circuit, residual=residual, src_keys=src_keys, dst=output_key,
                )
        else:
            if circuit is None:
                raise ValueError("circuit is required for edge interventions.")
            final_residual = gather_for_dst(
                circuit=circuit, residual=residual, src_keys=src_keys,
                dst=output_key, edge_intervention=edge_intervention,
            )

        if circuit is not None:
            final_residual = apply_gate(final_residual, circuit.nodes[output_key].node_mask)

        # No task head attached yet -- this returns the raw final hidden
        # state. Whatever downstream probe/loss is chosen later reads from
        # here, analogous to how GPT2's "output" node sits before ln_final
        # + unembed.
        return final_residual

    # ------------------------------------------------------------------
    # DiscoGP support: dense runtime masks
    # ------------------------------------------------------------------
    # Note: unlike GPT2, this fast path is NOT asymptotically cheaper than
    # the circuit= path above -- post-LN forces us to keep every source as
    # a separate slice through the whole depth regardless of how the masks
    # are represented, so both paths do the same underlying work. It exists
    # for interface parity with modeling_gpt.py / algorithms/discogp.py.

    def forward_runtime(
        self,
        input_values: torch.Tensor,
        *,
        runtime_masks: HubertRuntimeMasks,
        return_residual: bool = False,
    ) -> torch.Tensor:
        cfg = self.cfg
        residual, src_keys = self.embed_as_residual(input_values, None)
        edge_masks = runtime_masks.edge_masks
        weight_masks = runtime_masks.weight_masks

        for layer_id, block_module in enumerate(self.blocks):
            block = cast(CircuitHubertBlock, block_module)

            if edge_masks is not None:
                qkv_mask = edge_masks.attention_qkv[layer_id]  # [3, n_src, n_heads]
                qkv_input = torch.einsum("bpsd,qsh->bqphd", residual, qkv_mask)
                q_input, k_input, v_input = qkv_input[:, 0], qkv_input[:, 1], qkv_input[:, 2]
            else:
                full_input = residual.sum(dim=2)
                q_input = k_input = v_input = full_input.unsqueeze(2).expand(-1, -1, cfg.n_heads, -1)

            attn_out = block.attn.forward(
                q_input, k_input, v_input, circuit=None, layer_id=layer_id,
                weight_masks=None if weight_masks is None else weight_masks.attention[layer_id],
            )
            residual = torch.cat([residual, attn_out], dim=2)
            src_keys = src_keys + [(layer_id, head, "attn_o") for head in range(cfg.n_heads)]

            per_source1 = (
                residual * edge_masks.ln1[layer_id].view(1, 1, -1, 1)
                if edge_masks is not None else residual
            )
            decomposed = decompose_post_layer_norm(
                per_source1, ln_weight=block.ln1_weight, ln_bias=block.ln1_bias_param, eps=cfg.layer_norm_eps,
            )
            residual = torch.cat([decomposed, bias_source(decomposed, block.ln1_bias_param)], dim=2)
            src_keys = src_keys + [(layer_id, 0, "ln1_bias")]

            if edge_masks is not None:
                mlp_input = torch.einsum("bpsd,s->bpd", residual, edge_masks.mlp[layer_id])
            else:
                mlp_input = residual.sum(dim=2)

            mlp_out = block.mlp.forward(
                mlp_input, None if weight_masks is None else weight_masks.mlp[layer_id],
            )
            residual = torch.cat([residual, source_axis(mlp_out)], dim=2)
            src_keys = src_keys + [(layer_id, 0, "mlp")]

            per_source2 = (
                residual * edge_masks.ln2[layer_id].view(1, 1, -1, 1)
                if edge_masks is not None else residual
            )
            decomposed2 = decompose_post_layer_norm(
                per_source2, ln_weight=block.ln2_weight, ln_bias=block.ln2_bias_param, eps=cfg.layer_norm_eps,
            )
            
            residual = torch.cat([decomposed2, bias_source(decomposed2, block.ln2_bias_param)], dim=2)
            src_keys = src_keys + [(layer_id, 0, "ln2_bias")]

        if edge_masks is not None:
            final_residual = torch.einsum("bpsd,s->bpd", residual, edge_masks.output)
        else:
            final_residual = residual.sum(dim=2)

        return final_residual

    # ------------------------------------------------------------------
    # weight lookup (architecture-specific bridge used by DiscoGP)
    # ------------------------------------------------------------------

    def lookup_weight(self, n_key: node_key, w_key: str) -> torch.Tensor:
        layer, head, kind = n_key

        if kind in ATTENTION_NODE_WEIGHT_KEYS and w_key in ATTENTION_NODE_WEIGHT_KEYS[kind]:
            block = cast(CircuitHubertBlock, self.blocks[layer])
            parameter = getattr(block.attn, w_key)
            return parameter if w_key == "b_O" else parameter[head]

        if kind == "mlp":
            block = cast(CircuitHubertBlock, self.blocks[layer])
            return getattr(block.mlp, w_key)

        raise KeyError(f"cannot map weight key {(n_key, w_key)} to model parameter.")

    # ------------------------------------------------------------------
    # DiscoGP logit-group packing
    # ------------------------------------------------------------------

    def edge_logit_group_specs(self, circuit: Circuit) -> list[EdgeLogitGroupSpec]:
        self._assert_compatible_circuit(circuit)
        specs: list[EdgeLogitGroupSpec] = []
        seen: set[edge_key] = set()

        for layer in range(self.cfg.n_layers):
            heads = list(range(self.cfg.n_heads))
            qkv_dsts = {kind: [(layer, h, kind) for h in heads] for kind in ("attn_q", "attn_k", "attn_v")}
            srcs = circuit.incoming_srcs(qkv_dsts["attn_q"][0])
            for kind, dsts in qkv_dsts.items():
                for dst in dsts:
                    if circuit.incoming_srcs(dst) != srcs:
                        raise ValueError(f"cannot pack Hubert QKV edges for layer {layer}: sources differ for {dst}.")
            qkv_keys = tuple(
                (dst, src) for kind in ("attn_q", "attn_k", "attn_v") for src in srcs for dst in qkv_dsts[kind]
            )
            specs.append(EdgeLogitGroupSpec(keys=qkv_keys, shape=(3, len(srcs), self.cfg.n_heads), name=(layer, "attention_qkv")))
            seen.update(qkv_keys)

            for name, kind in (("ln1", "ln1"), ("mlp", "mlp"), ("ln2", "ln2")):
                dst_key = (layer, 0, kind)
                dst_srcs = circuit.incoming_srcs(dst_key)
                keys = tuple((dst_key, src) for src in dst_srcs)
                specs.append(EdgeLogitGroupSpec(keys=keys, shape=(len(dst_srcs),), name=(layer, name)))
                seen.update(keys)

        output_key = (self.cfg.n_layers, 0, "output")
        output_srcs = circuit.incoming_srcs(output_key)
        output_keys = tuple((output_key, src) for src in output_srcs)
        specs.append(EdgeLogitGroupSpec(keys=output_keys, shape=(len(output_srcs),), name=("output",)))
        seen.update(output_keys)

        if seen != circuit.all_edge_keys():
            missing = circuit.all_edge_keys() - seen
            extra = seen - circuit.all_edge_keys()
            raise ValueError(f"Hubert edge packing did not cover circuit exactly. missing={len(missing)}, extra={len(extra)}.")

        return specs

    def weight_logit_group_specs(self, circuit: Circuit) -> list[WeightLogitGroupSpec]:
        self._assert_compatible_circuit(circuit)
        specs: list[WeightLogitGroupSpec] = []
        seen: set[tuple[node_key, str]] = set()

        for layer in range(self.cfg.n_layers):
            for kind, weight_keys in ATTENTION_NODE_WEIGHT_KEYS.items():
                for w_key in weight_keys:
                    items = tuple(((layer, h, kind), w_key) for h in range(self.cfg.n_heads))
                    first = self.lookup_weight(*items[0])
                    specs.append(WeightLogitGroupSpec(items=items, shape=(self.cfg.n_heads, *tuple(first.shape)), name=(layer, "attention", w_key)))
                    seen.update(items)

            mlp_key = (layer, 0, "mlp")
            for w_key in MLP_WEIGHT_KEYS:
                item = (mlp_key, w_key)
                specs.append(WeightLogitGroupSpec(items=(item,), shape=tuple(self.lookup_weight(*item).shape), name=(layer, "mlp", w_key)))
                seen.add(item)

        all_items = {(n_key, w_key) for n_key, node in circuit.nodes.items() for w_key in node.weight_masks.keys()}
        if seen != all_items:
            missing = all_items - seen
            extra = seen - all_items
            raise ValueError(f"Hubert weight packing did not cover circuit exactly. missing={len(missing)}, extra={len(extra)}.")

        return specs

    def _runtime_edge_masks_from_logits(
        self, *, edge_logits, edge_group_specs, sample_mask_fn, reverse_edges: bool, random_mode, gs_temp_edge: float,
    ) -> HubertEdgeMasks:
        by_name = {
            spec.name: sample_mask_fn(logits, random_mode=random_mode, reverse=reverse_edges, gs_temp=gs_temp_edge)
            for spec, logits in zip(edge_group_specs, edge_logits)
        }
        return HubertEdgeMasks(
            attention_qkv=tuple(by_name[(l, "attention_qkv")] for l in range(self.cfg.n_layers)),
            ln1=tuple(by_name[(l, "ln1")] for l in range(self.cfg.n_layers)),
            mlp=tuple(by_name[(l, "mlp")] for l in range(self.cfg.n_layers)),
            ln2=tuple(by_name[(l, "ln2")] for l in range(self.cfg.n_layers)),
            output=by_name[("output",)],
        )

    def _runtime_weight_masks_from_logits(
        self, *, weight_logits, weight_group_specs, sample_mask_fn, reverse_weights: bool, random_mode, gs_temp_weight: float,
    ) -> HubertWeightMasks:
        by_name = {
            spec.name: sample_mask_fn(logits, random_mode=random_mode, reverse=reverse_weights, gs_temp=gs_temp_weight)
            for spec, logits in zip(weight_group_specs, weight_logits)
        }
        attention, mlp = [], []
        for layer in range(self.cfg.n_layers):
            attention.append(
                HubertAttentionWeightMasks(
                    W_Q=by_name[(layer, "attention", "W_Q")], b_Q=by_name[(layer, "attention", "b_Q")],
                    W_K=by_name[(layer, "attention", "W_K")], b_K=by_name[(layer, "attention", "b_K")],
                    W_V=by_name[(layer, "attention", "W_V")], b_V=by_name[(layer, "attention", "b_V")],
                    W_O=by_name[(layer, "attention", "W_O")], b_O=by_name[(layer, "attention", "b_O")],
                )
            )
            mlp.append(
                HubertMLPWeightMasks(
                    W_in=by_name[(layer, "mlp", "W_in")], b_in=by_name[(layer, "mlp", "b_in")],
                    W_out=by_name[(layer, "mlp", "W_out")], b_out=by_name[(layer, "mlp", "b_out")],
                )
            )
        return HubertWeightMasks(attention=tuple(attention), mlp=tuple(mlp))

    @torch.no_grad()
    def boolean_runtime_weight_masks(self, *, weight_logits, weight_group_specs, boolean_mask_fn) -> HubertWeightMasks:
        def detached(logits, **_):
            return boolean_mask_fn(logits).to(dtype=logits.dtype).detach()
        return self._runtime_weight_masks_from_logits(
            weight_logits=weight_logits, weight_group_specs=weight_group_specs,
            sample_mask_fn=detached, reverse_weights=False, random_mode=None, gs_temp_weight=1.0,
        )

    def sample_runtime_masks(
        self, *, edge_logits=None, edge_group_specs=None, weight_logits=None, weight_group_specs=None,
        frozen_weight_runtime=None, sample_mask_fn, boolean_mask_fn=None, mode: str,
        reverse_edges: bool = False, reverse_weights: bool = False,
        gs_temp_edge: float = 1.0, gs_temp_weight: float = 1.0, random_mode=None,
    ) -> HubertRuntimeMasks:
        edge_masks = None
        weight_masks = frozen_weight_runtime

        if mode == "edge":
            edge_masks = self._runtime_edge_masks_from_logits(
                edge_logits=edge_logits, edge_group_specs=edge_group_specs, sample_mask_fn=sample_mask_fn,
                reverse_edges=reverse_edges, random_mode=random_mode, gs_temp_edge=gs_temp_edge,
            )
        elif mode == "weight":
            weight_masks = self._runtime_weight_masks_from_logits(
                weight_logits=weight_logits, weight_group_specs=weight_group_specs, sample_mask_fn=sample_mask_fn,
                reverse_weights=reverse_weights, random_mode=random_mode, gs_temp_weight=gs_temp_weight,
            )
        else:
            raise ValueError(f"unknown runtime mask mode: {mode!r}.")

        return HubertRuntimeMasks(edge_masks=edge_masks, weight_masks=weight_masks)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load_model(cls, model_name: str = "hubert-base-ls960", *, device: str | None = None) -> "CircuitHubert":
        if model_name not in HF_ARCH_SPECS:
            raise ValueError(
                f"this modeling_hubert.py currently supports {sorted(HF_ARCH_SPECS)}; got model_name={model_name!r}."
            )
        spec = HF_ARCH_SPECS[model_name]
        device = device if device is not None else DEVICE

        import transformers

        hf_class = getattr(transformers, spec.hf_class_name)
        hf_model = hf_class.from_pretrained(spec.hf_checkpoint)
        hf_model.eval()
        hf_root = hf_model
        hf_cfg = hf_model.config

        cfg = HubertCircuitConfig(
            n_layers=hf_cfg.num_hidden_layers,
            n_heads=hf_cfg.num_attention_heads,
            d_model=hf_cfg.hidden_size,
            d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            d_mlp=hf_cfg.intermediate_size,
            layer_norm_eps=hf_cfg.layer_norm_eps,
            hidden_act=hf_cfg.hidden_act if isinstance(hf_cfg.hidden_act, str) else "gelu",
            arch_name=model_name,
            conv_kernel=list(hf_cfg.conv_kernel),
            conv_stride=list(hf_cfg.conv_stride),
        )

        preprocessor = FrozenSpeechPreprocessor(hf_root).to(device=device)
        model = cls(cfg, preprocessor, device=device)
        model._load_hf_weights(hf_root)

        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.to(device=device)
        model.eval()
        return model

    @torch.no_grad()
    def _load_hf_weights(self, hf_root: nn.Module) -> None:
        """
        Split HF's flat (embed_dim, embed_dim) attention projections into our
        per-head (n_heads, d_model, d_head) layout. HF splits the *output*
        dimension of q/k/v_proj into contiguous per-head chunks (see
        `.view(*input_shape, -1, head_dim)` in HubertAttention/Wav2Vec2Attention),
        and reads out_proj's *input* dimension the same way, so both directions
        use a plain contiguous reshape -- no permutation needed.
        """
        cfg = self.cfg
        n_heads, d_head = cfg.n_heads, cfg.d_head

        for layer_id, hf_layer in enumerate(hf_root.encoder.layers):
            block = cast(CircuitHubertBlock, self.blocks[layer_id])
            attn = hf_layer.attention

            def split_in(weight: torch.Tensor) -> torch.Tensor:
                # weight: (embed_dim, embed_dim) = (out, in); y = x @ weight.T
                # -> per-head (n_heads, d_model, d_head)
                return weight.t().reshape(cfg.d_model, n_heads, d_head).permute(1, 0, 2).contiguous()

            def split_bias(bias: torch.Tensor) -> torch.Tensor:
                return bias.reshape(n_heads, d_head).contiguous()

            block.attn.W_Q.copy_(split_in(attn.q_proj.weight.data))
            block.attn.b_Q.copy_(split_bias(attn.q_proj.bias.data))
            block.attn.W_K.copy_(split_in(attn.k_proj.weight.data))
            block.attn.b_K.copy_(split_bias(attn.k_proj.bias.data))
            block.attn.W_V.copy_(split_in(attn.v_proj.weight.data))
            block.attn.b_V.copy_(split_bias(attn.v_proj.bias.data))

            # out_proj: (embed_dim, embed_dim) = (out=d_model, in=embed_dim);
            # y = concat_h(z_h) @ out_proj.weight.T -> per-head (n_heads, d_head, d_model)
            w_o = attn.out_proj.weight.data.t().reshape(n_heads, d_head, cfg.d_model).contiguous()
            block.attn.W_O.copy_(w_o)
            block.attn.b_O.copy_(attn.out_proj.bias.data)

            block.ln1_weight.copy_(hf_layer.layer_norm.weight.data)
            block.ln1_bias_param.copy_(hf_layer.layer_norm.bias.data)
            block.ln2_weight.copy_(hf_layer.final_layer_norm.weight.data)
            block.ln2_bias_param.copy_(hf_layer.final_layer_norm.bias.data)

            block.mlp.W_in.copy_(hf_layer.feed_forward.intermediate_dense.weight.data.t())
            block.mlp.b_in.copy_(hf_layer.feed_forward.intermediate_dense.bias.data)
            block.mlp.W_out.copy_(hf_layer.feed_forward.output_dense.weight.data.t())
            block.mlp.b_out.copy_(hf_layer.feed_forward.output_dense.bias.data)
            


__all__ = [
    "HF_ARCH_SPECS",
    "HubertCircuitConfig",
    "CircuitHubert",
    "build_full_hubert_circuit",
    "finalize_hubert_circuit",
    "decompose_post_layer_norm",
    "get_feat_extract_output_lengths",
]
