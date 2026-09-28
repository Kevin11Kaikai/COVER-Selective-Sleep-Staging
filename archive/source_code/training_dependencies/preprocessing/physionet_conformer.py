import argparse
import os
from typing import Dict, Iterable, Tuple

import numpy as np
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
)
from sklearn.preprocessing import scale as standard_scale


"""
Preprocessing pipeline for using Sleep Physionet (Sleep-EDF) with EEG-Conformer.

Steps:
1) Load SleepPhysionet recordings (PSG + hypnogram) via braindecode.
2) Raw-level preprocessing:
   - Convert from Volts to microvolts (µV).
   - Band-pass / low-pass filter (defaults: 0.3–40 Hz).
3) Extract 30-second windows based on sleep stage annotations, with
   stage-3 and stage-4 merged into a single N3 class:
       W -> 0, N1 -> 1, N2 -> 2, N3/N4 -> 3, REM -> 4.
4) Window-level preprocessing:
   - Channel-wise z-score (per window) using sklearn's `scale`.
5) Export:
   - X: numpy array of shape (N, 1, C, T)
   - y: numpy array of shape (N,)

Example usage:
--------------
python preprocessing/physionet_conformer.py ^
    --subject-ids 0 1 ^
    --recording-ids 2 ^
    --crop-wake-mins 30 ^
    --low-cut-hz 0.3 ^
    --high-cut-hz 40.0 ^
    --window-size-sec 30 ^
    --out-dir datasets/PhysioNet/processed_conformer
"""


def build_raw_preprocessors(low_cut_hz: float, high_cut_hz: float):
    """Build preprocessors applied on raw recordings."""
    preprocessors = [
        # Convert from Volts to microvolts for numerical stability.
        Preprocessor(lambda data: data * 1e6, apply_on_array=True),
    ]
    # Band-pass / low-pass filter on raw data (sleep-related band).
    if low_cut_hz is not None or high_cut_hz is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz)
        )
    return preprocessors


def default_sleep_stage_mapping() -> Dict[str, int]:
    """Return default mapping from annotation labels to integer classes."""
    return {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,  # merge into N3
        "Sleep stage R": 4,
    }


def create_sleep_windows(
    dataset: SleepPhysionet,
    window_size_sec: float,
    sfreq: float = None,
    mapping: Dict[str, int] = None,
):
    """Create fixed-length windows from SleepPhysionet dataset."""
    if sfreq is None:
        sfreq = dataset.datasets[0].raw.info["sfreq"]
    if mapping is None:
        mapping = default_sleep_stage_mapping()

    window_size_samples = int(window_size_sec * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping,
    )
    return windows_dataset


def windows_to_numpy(
    windows_dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a braindecode windows dataset into numpy arrays for EEG-Conformer.

    Returns
    -------
    X : np.ndarray
        Shape (N, 1, C, T)
    y : np.ndarray
        Shape (N,)
    """
    Xs: Iterable[np.ndarray] = []
    ys: Iterable[int] = []

    for X, y, _ in windows_dataset:
        # X has shape (C, T); add a singleton "conv channel" dimension.
        Xs.append(X[None, ...])
        ys.append(int(y))

    X_arr = np.stack(Xs, axis=0)  # (N, 1, C, T)
    y_arr = np.asarray(ys, dtype=np.int64)
    return X_arr, y_arr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Sleep Physionet (Sleep-EDF) for EEG-Conformer: "
            "raw preprocessing, 30s window extraction and numpy export."
        )
    )
    parser.add_argument(
        "--subject-ids",
        type=int,
        nargs="+",
        required=True,
        help="List of subject indices to include (e.g. 0 1 2).",
    )
    parser.add_argument(
        "--recording-ids",
        type=int,
        nargs="+",
        default=[2],
        help="Recording indices per subject (default: 2, i.e., second night).",
    )
    parser.add_argument(
        "--crop-wake-mins",
        type=float,
        default=30.0,
        help="Minutes of wake time to remove at start and end (SleepPhysionet arg).",
    )
    parser.add_argument(
        "--low-cut-hz",
        type=float,
        default=0.3,
        help="Lower cutoff frequency for band-pass filter (Hz). Use None to disable.",
    )
    parser.add_argument(
        "--high-cut-hz",
        type=float,
        default=40.0,
        help="Upper cutoff frequency for band-pass / low-pass filter (Hz). "
        "Use None to disable.",
    )
    parser.add_argument(
        "--window-size-sec",
        type=float,
        default=30.0,
        help="Window size in seconds (default: 30s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets/PhysioNet/processed_conformer",
        help="Output directory for X.npy and y.npy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"Loading Sleep Physionet subjects={args.subject_ids}, "
        f"recordings={args.recording_ids}, crop_wake_mins={args.crop_wake_mins}"
    )
    dataset = SleepPhysionet(
        subject_ids=args.subject_ids,
        recording_ids=args.recording_ids,
        crop_wake_mins=args.crop_wake_mins,
        preload=True,
        load_eeg_only=True,
    )

    # Raw-level preprocessing
    print(
        f"Applying raw preprocessors: V→µV, "
        f"filter {args.low_cut_hz}-{args.high_cut_hz} Hz"
    )
    raw_preprocessors = build_raw_preprocessors(args.low_cut_hz, args.high_cut_hz)
    preprocess(dataset, raw_preprocessors)

    # Window extraction
    print(f"Creating {args.window_size_sec:.1f}s windows from annotations...")
    windows_dataset = create_sleep_windows(
        dataset, window_size_sec=args.window_size_sec
    )
    print(f"Total windows: {len(windows_dataset)}")

    # Window-level preprocessing (per-window, per-channel z-score)
    print("Applying window-level channel-wise z-score normalization...")
    preprocess(
        windows_dataset, [Preprocessor(standard_scale, channel_wise=True)]
    )

    # Convert to numpy arrays for EEG-Conformer
    print("Converting windows dataset to numpy arrays...")
    X, y = windows_to_numpy(windows_dataset)

    # Sanity checks
    print(f"X shape: {X.shape} (N, 1, C, T)")
    print(f"y shape: {y.shape} (N,)")
    if X.ndim != 4:
        raise RuntimeError("Expected X to have shape (N, 1, C, T)")

    # Save
    x_path = os.path.join(args.out_dir, "X.npy")
    y_path = os.path.join(args.out_dir, "y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"Saved X to {x_path}")
    print(f"Saved y to {y_path}")


if __name__ == "__main__":
    main()


