from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import torch

from .config import ExperimentConfig
from .data import default_dataset_paths
from .run_experiments import run_one


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh2")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--horizons", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    exp = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = default_dataset_paths()
    if args.dataset not in paths:
        raise ValueError(f"unknown dataset: {args.dataset}. valid={list(paths.keys())}")

    models = (
        _parse_csv_list(args.models)
        if args.models is not None
        else [
            "lstm",
            "lstm_star",
            "gru",
            "gru_star",
            "transformer",
            "transformer_star",
            "cnn1d",
            "cnn1d_star",
            "da_rnn",
            "da_rnn_star",
            "tpa_lstm",
            "tpa_lstm_star",
            "da_conv_lstm",
            "da_conv_lstm_star",
            "ts_conv_lstm",
            "ts_conv_lstm_star",
            "tcln_plain", 
            "tcln_starnet",
            "tcln_strict",
            "tcln_strict_star",
        ]
    )

    horizons = [int(x) for x in _parse_csv_list(args.horizons)] if args.horizons is not None else list(exp.horizons)

    os.makedirs(exp.results_dir, exist_ok=True)
    out_path = os.path.join(exp.results_dir, f"summary_all__{args.dataset}.csv")

    all_rows = []
    for h in horizons:
        for m in models:
            csv_expected = None
            base = m[: -len("_star")] if m.endswith("_star") else m
            run_name = f"{base}__{args.dataset}__h{h}"
            if m.endswith("_star"):
                run_name = f"{run_name}_star"
            csv_expected = os.path.join(exp.results_dir, f"{run_name}.csv")

            if args.skip_existing and os.path.exists(csv_expected):
                continue

            _, row = run_one(
                exp=exp,
                dataset_csv=paths[args.dataset],
                dataset_name=args.dataset,
                model_name=m,
                horizon=h,
                device=device,
            )
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(out_path, index=False)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print("summary saved:", out_path)


if __name__ == "__main__":
    main()
