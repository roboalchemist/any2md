#!/usr/bin/env python3
"""Regenerate the segment vectors used by any2md-models/diarize.rs tests.

    .venv/bin/python rust/parity/diarize_segments.py
"""

import mlx.core as mx
import numpy as np

import mlx_audio.vad.models.sortformer.sortformer as sf

PREDS = np.array(
    [
        [0.9, 0.1], [0.8, 0.2], [0.2, 0.7], [0.1, 0.9], [0.1, 0.85], [0.6, 0.4],
        [0.4, 0.6], [0.3, 0.3], [0.95, 0.05], [0.9, 0.1], [0.2, 0.2], [0.7, 0.3],
    ],
    dtype=np.float32,
)


def main() -> None:
    model_cls = getattr(sf, "SortformerModel", None) or getattr(sf, "Model")
    for name, kwargs in (
        ("base", dict(threshold=0.5, min_duration=0.0, merge_gap=0.0)),
        ("merge_gap=0.3", dict(threshold=0.5, min_duration=0.0, merge_gap=0.3)),
        ("min_duration=0.1", dict(threshold=0.5, min_duration=0.1, merge_gap=0.0)),
    ):
        segs = model_cls._preds_to_segments(mx.array(PREDS), frame_duration=0.08, **kwargs)
        print(name, [(round(s.start * 1000), round(s.end * 1000), s.speaker) for s in segs])


if __name__ == "__main__":
    main()
