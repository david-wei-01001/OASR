"""
circuit_discovery/tasks/articulatory_index.py

Builds an on-disk dataset for the Articulatory Index corpus (10329 isolated
CV / VC syllables, 20 speakers) and a matching dataloader, targeting
CircuitHubert. Two label sets are wired up first: consonant_classification
(24-way) and vowel_classification (15-way); the other five extractors below
(voicing, gender, speaker, syllable_order) are kept for later since they use
the same pipeline end to end -- just call prepare_and_save_articulatory_dataset
with a different task_type.

Why this is a *prep script*, not an on-the-fly per-batch loader: the existing
IOI pipeline (utils.load_task_dataset / run.load_task_dataset_from_config)
loads a pre-saved HF dataset from disk, calls `.with_format("torch")`, and
wraps it in a plain DataLoader with no custom collate_fn -- every row has to
already be a fixed-shape tensor. Articulatory Index clips are short isolated
sounds, so padding every clip to one fixed sample count up front is cheap and
lets the resulting dataset slot into that same shape (an `input_ids` column
holding the padded waveform, a `label` column) rather than inventing a new
loading convention.

Two HF "feature extractor" concepts that share a name but aren't the same
thing: `datasets.Audio` here does resampling/decoding at the *data* level
(this file); `CircuitHubert`'s internal `feature_extractor` (the CNN conv
stack) does representation learning at the *model* level, inside
FrozenSpeechPreprocessor, and is untouched by anything in this file. This
file only ever hands the model a raw waveform via `batch["input_ids"]`,
consistent with the "keep using the key input_ids" convention documented in
CircuitHubert's own integration guide.

Split strategy: plain stratified random split (datasets' own
`stratify_by_column`), not speaker-disjoint. Because that's a different
split contract than utils.load_task_dataset's own (non-stratified)
train_test_split, this file saves an already-split DatasetDict and loads it
directly with `load_articulatory_dataset` below, rather than routing through
utils.load_task_dataset -- utils.py, run.py, circuit.py, and algorithms/*.py
are all left untouched.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
from datasets import Audio, ClassLabel, Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader

from ..utils import DATASET_FOLDER_PATH, pick_device

DEVICE = pick_device()

gender_map = {"f": 1, "m": 0}

# --- phoneme code lists from mapping ---
# (sorted longest-first to ensure greedy matching of multi-char codes like 'xq', 'xz', etc.)
VOWEL_SET = {
    "xq", "xa", "xw", "xy", "xr", "xe", "xi", "xo", "xu",
    "a", "c", "e", "i", "o", "u",
}

CONSONANT_SET = {
    "xc", "xd", "xg", "xj", "xs", "xt", "xz",
    "b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "z",
}

CONSONANT_DICT = {
    "xc": 3, "xd": 0, "xg": 23, "xj": 8, "xs": 14, "xt": 16, "xz": 7,
    "b": 15, "d": 9, "f": 20, "g": 17, "h": 6, "k": 10, "l": 18, "m": 22, "n": 19, "p": 2,
    "r": 4, "s": 12, "t": 11, "v": 5, "w": 1, "y": 13, "z": 21,
}

VOWEL_DICT = {
    "xq": 0, "xa": 1, "xw": 2, "xy": 3, "xr": 4, "xe": 5, "xi": 6, "xo": 7, "xu": 8,
    "a": 9, "c": 10, "e": 11, "i": 12, "o": 13, "u": 14,
}

SINGLE_VOWELS = {v for v in VOWEL_SET if len(v) == 1}
SINGLE_CONSONANTS = {c for c in CONSONANT_SET if len(c) == 1}

SPEAKER_RE = re.compile(r"^[mf]\d{3}$", re.IGNORECASE)
TYPE_SET = {"s", "p"}

DATA_DIR = "/w/435/cse/noise_wer_correlations/disco_data/articulatory_index/AI_LSCP/isolated_sounds/wav"

voicing_list = {
    "voiced": {"xc", "f", "h", "k", "p", "s", "xs", "t", "xt"},
    "unvoiced": {"b", "d", "xd", "g", "xj", "l", "m", "n", "xg", "r", "v", "w", "y", "z", "xz"},
}

speaker_dict = {
    "f101": 0, "f103": 1, "f105": 2, "f106": 3, "f108": 4, "f109": 5, "f113": 6, "f119": 7,
    "m102": 8, "m104": 9, "m107": 10, "m110": 11, "m111": 12, "m112": 13, "m114": 14,
    "m115": 15, "m116": 16, "m117": 17, "m118": 18, "m120": 19,
}

label_mapping = {"voiced": 0, "unvoiced": 1}
syllable_order_map = {"CV": 1, "VC": 0}

_voiced_codes = sorted(voicing_list["voiced"], key=len, reverse=True)
_unvoiced_codes = sorted(voicing_list["unvoiced"], key=len, reverse=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# task_type -> (label_extractor, num_classes, ordered class names by index)
TASK_SPECS: dict[str, tuple[Callable, int, list[str]]] = {}


# --------------------------------------------------------------------------------------
# filename / syllable parsing (unchanged from the original loader)
# --------------------------------------------------------------------------------------

def _split_filename_parts(filename: str) -> Tuple[str, str, str]:
    """
    From a filename (without extension) return (recording_type, speaker_id, syllable_str).
    Supports either order:
      - s_f103_dxu
      - m112_s_xuxz
    Raises ValueError if required pieces cannot be found.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = [p for p in base.split("_") if p]

    rec_type = None
    speaker = None
    syllable = None

    for p in parts:
        if p in TYPE_SET and rec_type is None:
            rec_type = p
        elif SPEAKER_RE.match(p) and speaker is None:
            speaker = p
        else:
            if syllable is None:
                syllable = p
            else:
                syllable = syllable + "_" + p

    if rec_type is None or speaker is None or syllable is None:
        raise ValueError(f"Could not parse filename into (type, speaker, syllable): '{filename}' (parts={parts})")
    return rec_type, speaker, syllable


def _split_syllable_into_vowel_consonant(syllable: str) -> Tuple[str, str, str]:
    """
    Return (vowel_code, consonant_code, order) where order is "CV" or "VC".
    Raises ValueError on malformed syllables or if a clear vowel/consonant
    can't be determined.
    """
    s = syllable.lower()

    if "x" not in s:
        if len(s) != 2:
            raise ValueError(f"Expected 2-letter syllable when no 'x' present, got '{s}'")
        a, b = s[0], s[1]
        a_is_v, b_is_v = a in SINGLE_VOWELS, b in SINGLE_VOWELS
        a_is_c, b_is_c = a in SINGLE_CONSONANTS, b in SINGLE_CONSONANTS

        if a_is_v and b_is_c:
            return (a, b, "VC")
        if b_is_v and a_is_c:
            return (b, a, "CV")
        raise ValueError(f"Cannot determine vowel/consonant in '{s}' (no 'x').")

    x_count = s.count("x")

    if x_count == 1:
        if len(s) != 3:
            raise ValueError(f"Expected 3-letter syllable when single 'x' present, got '{s}'")
        i = s.index("x")
        if i == len(s) - 1:
            raise ValueError(f"Found 'x' at end in '{s}' -- cannot group with next char.")
        x_code = s[i:i + 2]
        other = s[:i] + s[i + 2:]
        other_idx = 0 if i != 0 else 2

        if x_code in VOWEL_SET and other in CONSONANT_SET:
            order = "VC" if i < other_idx else "CV"
            return (x_code, other, order)
        if x_code in CONSONANT_SET and other in VOWEL_SET:
            order = "CV" if i < other_idx else "VC"
            return (other, x_code, order)
        raise ValueError(f"Unable to classify pieces from '{s}': x_code='{x_code}', other='{other}'")

    if x_count == 2:
        if len(s) != 4:
            raise ValueError(f"Expected 4-letter syllable when two 'x' present, got '{s}'")
        i1 = s.index("x")
        i2 = s.index("x", i1 + 1)
        code1 = s[i1:i1 + 2]
        code2 = s[i2:i2 + 2]
        code1_is_v, code2_is_v = code1 in VOWEL_SET, code2 in VOWEL_SET
        code1_is_c, code2_is_c = code1 in CONSONANT_SET, code2 in CONSONANT_SET

        if code1_is_v and code2_is_c:
            return (code1, code2, "VC")
        if code2_is_v and code1_is_c:
            return (code2, code1, "CV")
        raise ValueError(f"Two 'x' codes in '{s}' do not resolve to vowel+consonant: '{code1}', '{code2}'")

    raise ValueError(f"Unexpected syllable format (more than 2 'x') for '{s}'")


# --------------------------------------------------------------------------------------
# label extractors (unchanged signatures; consonant/vowel are the two we're wiring up now)
# --------------------------------------------------------------------------------------

def consonant_label_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    if consonant is None:
        if unknown_label is not None:
            return int(unknown_label)
        raise ValueError(f"No consonant provided for speaker={speaker_id}, file type={recording_type}.")
    return int(CONSONANT_DICT[consonant.lower()])


def vowel_label_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    if vowel is None:
        if unknown_label is not None:
            return int(unknown_label)
        raise ValueError(f"No vowel provided for speaker={speaker_id}, file type={recording_type}.")
    return int(VOWEL_DICT[vowel.lower()])


def voicing_label_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    if consonant is None:
        if unknown_label is not None:
            return int(unknown_label)
        raise ValueError(f"No consonant provided for speaker={speaker_id}, file type={recording_type}.")
    c = consonant.lower()
    for code in _voiced_codes:
        if code in c:
            return int(label_mapping["voiced"])
    for code in _unvoiced_codes:
        if code in c:
            return int(label_mapping["unvoiced"])
    if unknown_label is not None:
        return int(unknown_label)
    raise ValueError(f"Consonant '{consonant}' for speaker={speaker_id} not found in voicing map.")


def gender_label_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    if "f" in speaker_id:
        return int(gender_map["f"])
    if "m" in speaker_id:
        return int(gender_map["m"])
    raise ValueError(f"Non-existing gender {speaker_id}")


def speaker_label_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    return speaker_dict[speaker_id]


def syllable_order_extractor(
    recording_type: str, speaker_id: str, vowel: Optional[str], consonant: Optional[str],
    order: Optional[str] = None, unknown_label: Optional[int] = None,
) -> int:
    if order not in syllable_order_map:
        if unknown_label is not None:
            return int(unknown_label)
        raise ValueError(
            f"Could not determine CV/VC order for speaker={speaker_id}, vowel={vowel}, "
            f"consonant={consonant}, order={order}."
        )
    return syllable_order_map[order]


def _class_names_by_index(label_dict: dict[str, int]) -> list[str]:
    by_index = sorted(label_dict.items(), key=lambda kv: kv[1])
    return [name for name, _ in by_index]


TASK_SPECS = {
    "consonant_classification": (consonant_label_extractor, len(CONSONANT_DICT), _class_names_by_index(CONSONANT_DICT)),
    "vowel_classification": (vowel_label_extractor, len(VOWEL_DICT), _class_names_by_index(VOWEL_DICT)),
    "voicing": (voicing_label_extractor, 2, _class_names_by_index(label_mapping)),
    "gender": (gender_label_extractor, 2, _class_names_by_index(gender_map)),
    "speaker": (speaker_label_extractor, len(speaker_dict), _class_names_by_index(speaker_dict)),
    "syllable_order_classification": (syllable_order_extractor, 2, _class_names_by_index(syllable_order_map)),
}


# --------------------------------------------------------------------------------------
# raw (audio_path, label) dataset
# --------------------------------------------------------------------------------------

def load_raw_articulatory_dataset(
    task_type: str,
    *,
    data_dir: str = DATA_DIR,
    require_any: bool = True,
    extensions: Optional[List[str]] = None,
) -> Dataset:
    """
    Return a HuggingFace Dataset with columns: audio_path (str), label (int),
    built by parsing filenames in `data_dir` per `task_type`'s label extractor.
    """
    if task_type not in TASK_SPECS:
        raise ValueError(f"unknown task_type {task_type!r}; expected one of {sorted(TASK_SPECS)}.")
    label_extractor, _num_classes, _names = TASK_SPECS[task_type]

    if extensions is None:
        extensions = [".wav"]

    audio_paths: list[str] = []
    labels: list[int] = []

    for fname in sorted(os.listdir(data_dir)):
        if not any(fname.lower().endswith(ext) for ext in extensions):
            continue
        full = os.path.join(data_dir, fname)
        rec_type, speaker_id, syllable = _split_filename_parts(fname)
        vowel_code, consonant_code, order = _split_syllable_into_vowel_consonant(syllable)

        try:
            label = label_extractor(rec_type, speaker_id, vowel_code, consonant_code, order=order)
        except Exception:
            logger.error("Error with file: %s, %s, %s, %s, %s", fname, rec_type, speaker_id, vowel_code, consonant_code)
            raise

        audio_paths.append(full)
        labels.append(int(label))

    if not audio_paths and require_any:
        raise ValueError(f"No audio files found/parsed in {data_dir}")

    logger.info("Parsed %d files for task_type=%s", len(audio_paths), task_type)
    return Dataset.from_dict({"audio_path": audio_paths, "label": labels})


# --------------------------------------------------------------------------------------
# audio decoding, fixed-length padding, save to disk
# --------------------------------------------------------------------------------------

def prepare_and_save_articulatory_dataset(
    task_type: str,
    *,
    data_dir: str = DATA_DIR,
    target_sr: int = 16000,
    max_seconds: Optional[float] = None,
    test_size: float = 0.1,
    seed: int = 42,
    normalize: bool = False,
    out_dir: Optional[Path] = None,
) -> DatasetDict:
    """
    Build and save an already-split (train/test) HF DatasetDict for one
    Articulatory Index task, with columns:
        input_ids: float32 array, fixed length (padded/truncated waveform, target_sr Hz)
        label:     ClassLabel int

    `normalize`: whether to zero-mean/unit-variance normalize each clip
    before padding. Check the checkpoint's own preprocessor config before
    turning this on --
        AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960").do_normalize
    facebook/hubert-base-ls960 and facebook/wav2vec2-base ship with
    do_normalize=False, so the default here is False; flip it only if your
    checkpoint's config says otherwise.

    `max_seconds`: if given, clips longer than this are truncated and the
    padding length is capped here instead of being set by the single longest
    clip in the corpus (useful if a few outlier files are much longer than
    the rest of the isolated-sound corpus).
    """
    if task_type not in TASK_SPECS:
        raise ValueError(f"unknown task_type {task_type!r}; expected one of {sorted(TASK_SPECS)}.")
    _label_extractor, num_classes, class_names = TASK_SPECS[task_type]

    ds = load_raw_articulatory_dataset(task_type, data_dir=data_dir)
    ds = ds.cast_column("audio_path", Audio(sampling_rate=target_sr))

    # first pass: find the padding length. datasets.Audio decodes lazily on
    # access, so this does incur one decode per file -- fine at ~10k short
    # clips, but this is the expensive step if the corpus grows a lot.
    lengths = [len(row["array"]) for row in ds["audio_path"]]
    max_len = max(lengths)
    if max_seconds is not None:
        max_len = min(max_len, int(max_seconds * target_sr))
    logger.info(
        "Articulatory Index (%s): %d clips, lengths %d..%d samples (%.3f..%.3fs), padding to %d samples (%.3fs)",
        task_type, len(lengths), min(lengths), max(lengths),
        min(lengths) / target_sr, max(lengths) / target_sr, max_len, max_len / target_sr,
    )

    def _to_fixed_length(example):
        arr = np.asarray(example["audio_path"]["array"], dtype=np.float32)
        if arr.shape[0] > max_len:
            arr = arr[:max_len]
        if normalize:
            std = arr.std()
            if std > 0:
                arr = (arr - arr.mean()) / std
        if arr.shape[0] < max_len:
            arr = np.pad(arr, (0, max_len - arr.shape[0]))
        return {"input_ids": arr}

    ds = ds.map(_to_fixed_length, remove_columns=["audio_path"], desc=f"pad/normalize ({task_type})")
    ds = ds.cast_column("label", ClassLabel(num_classes=num_classes, names=class_names))

    split = ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")

    out_dir = Path(out_dir) if out_dir is not None else DATASET_FOLDER_PATH / f"articulatory_{task_type}_dataset"
    split.save_to_disk(str(out_dir))
    logger.info("Saved %s -> %s (train=%d, test=%d)", task_type, out_dir, len(split["train"]), len(split["test"]))

    return split


# --------------------------------------------------------------------------------------
# loading (parallel to utils.load_task_dataset, kept separate because of the
# stratified-split contract -- see module docstring)
# --------------------------------------------------------------------------------------

def load_articulatory_dataset(
    task_type: str,
    *,
    batch_size: int,
    dataset_dir: Optional[Path] = None,
) -> SimpleNamespace:
    """
    Load a dataset previously written by prepare_and_save_articulatory_dataset
    and wrap it in train/test DataLoaders. Same return shape as
    utils.load_task_dataset (SimpleNamespace(train=..., test=...)), so it
    drops into notebooks/run.py-style code the same way.
    """
    path = Path(dataset_dir) if dataset_dir is not None else DATASET_FOLDER_PATH / f"articulatory_{task_type}_dataset"
    ds = load_from_disk(str(path))
    ds = ds.with_format("torch", output_all_columns=True, device=DEVICE)

    return SimpleNamespace(
        train=DataLoader(ds["train"], batch_size=batch_size, shuffle=True),
        test=DataLoader(ds["test"], batch_size=batch_size, shuffle=False),
    )
