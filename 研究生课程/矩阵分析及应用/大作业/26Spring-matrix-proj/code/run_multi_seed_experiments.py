import argparse
import csv
import math
from pathlib import Path

import torch

from lenet_cp_mnist import (
    MODEL_REGISTRY,
    build_cp_model_from_baseline,
    build_layerwise_finetuned_cp_model,
    count_parameters,
    evaluate,
    get_data_loaders,
    train,
    set_seed,
)


CONFIGS = {
    "lenet5": {
        "rank": 8,
    },
    "smallcnn": {
        "rank": 16,
    },
}

METHODS = [
    "baseline",
    "cp_no_ft",
    "global_ft",
    "layerwise_ft",
    "data_aware_layerwise_ft",
]


def read_existing(path):
    if not path.exists():
        return [], set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {(row["dataset"], row["model"], int(row["seed"]), row["method"]) for row in rows}
    return rows, keys


def append_row(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(rows, output_path):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["method"])
        grouped.setdefault(key, []).append(float(row["accuracy"]))

    fieldnames = ["dataset", "model", "method", "n", "mean_accuracy", "std", "standard_error"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped):
            values = grouped[key]
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                std = math.sqrt(variance)
                stderr = std / math.sqrt(len(values))
            else:
                std = 0.0
                stderr = 0.0
            writer.writerow(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "method": key[2],
                    "n": len(values),
                    "mean_accuracy": f"{mean:.4f}",
                    "std": f"{std:.4f}",
                    "standard_error": f"{stderr:.4f}",
                }
            )


def model_stats(model, conv_layers):
    total_params, conv_params = count_parameters(model, conv_layers)
    return total_params, conv_params


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), path)


def run_one(args, dataset, model_name, seed, rows_path, summary_path, fieldnames, done):
    key_prefix = (dataset, model_name, seed)
    config = CONFIGS[model_name]
    rank = config["rank"]
    dataset_tag = dataset.replace("-", "_")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    train_loader, test_loader = get_data_loaders(
        args.data_dir,
        args.batch_size,
        dataset=dataset,
        download=True,
    )

    model_config = MODEL_REGISTRY[model_name]
    model_factory = model_config["factory"]
    conv_layers = model_config["conv_layers"]
    baseline_checkpoint = Path(args.checkpoint_dir) / f"{model_name}_{dataset_tag}_seed{seed}.pt"

    baseline = model_factory()
    if baseline_checkpoint.exists() and not args.force_baseline:
        baseline.load_state_dict(torch.load(baseline_checkpoint, map_location="cpu"))
        print(f"Loaded baseline: {baseline_checkpoint}")
    else:
        print(f"Training baseline: dataset={dataset} model={model_name} seed={seed}")
        train(baseline, train_loader, epochs=args.baseline_epochs, lr=args.baseline_lr, device=device)
        save_checkpoint(baseline, baseline_checkpoint)

    baseline_acc = evaluate(baseline, test_loader, device)
    base_total_params, base_conv_params = model_stats(baseline, conv_layers)

    def record(method, accuracy, model, rank_desc=None):
        key = (*key_prefix, method)
        if key in done:
            return
        total_params, conv_params = model_stats(model, conv_layers)
        row = {
            "dataset": dataset,
            "model": model_name,
            "seed": seed,
            "method": method,
            "rank": rank_desc or str(rank),
            "accuracy": f"{accuracy:.4f}",
            "baseline_accuracy": f"{baseline_acc:.4f}",
            "baseline_gap_pp": f"{accuracy - baseline_acc:.4f}",
            "total_params": total_params,
            "conv_params": conv_params,
            "baseline_total_params": base_total_params,
            "baseline_conv_params": base_conv_params,
        }
        append_row(rows_path, fieldnames, row)
        done.add(key)
        print(
            f"RESULT dataset={dataset} model={model_name} seed={seed} "
            f"method={method} acc={accuracy:.2f}"
        )

    record("baseline", baseline_acc, baseline)

    if (*key_prefix, "cp_no_ft") not in done:
        set_seed(seed)
        cp_model = build_cp_model_from_baseline(model_factory, baseline.cpu(), conv_layers, rank, args.als_iters)
        cp_acc = evaluate(cp_model, test_loader, device)
        record("cp_no_ft", cp_acc, cp_model)

    if (*key_prefix, "global_ft") not in done:
        set_seed(seed)
        model = build_cp_model_from_baseline(model_factory, baseline.cpu(), conv_layers, rank, args.als_iters)
        train(model, train_loader, epochs=5, lr=1e-4, device=device)
        acc = evaluate(model, test_loader, device)
        save_checkpoint(model, Path(args.checkpoint_dir) / f"{model_name}_{dataset_tag}_cp_global_ft_seed{seed}.pt")
        record("global_ft", acc, model)

    if (*key_prefix, "layerwise_ft") not in done:
        set_seed(seed)
        model = build_layerwise_finetuned_cp_model(
            model_factory,
            baseline.cpu(),
            train_loader,
            conv_layers,
            rank,
            args.als_iters,
            layer_epochs=2,
            layer_lr=1e-3,
            final_epochs=5,
            final_lr=1e-4,
            device=device,
        )
        acc = evaluate(model, test_loader, device)
        save_checkpoint(model, Path(args.checkpoint_dir) / f"{model_name}_{dataset_tag}_cp_layerwise_ft_seed{seed}.pt")
        record("layerwise_ft", acc, model)

    if (*key_prefix, "data_aware_layerwise_ft") not in done:
        set_seed(seed)
        model = build_layerwise_finetuned_cp_model(
            model_factory,
            baseline.cpu(),
            train_loader,
            conv_layers,
            rank,
            args.als_iters,
            layer_epochs=2,
            layer_lr=1e-3,
            final_epochs=5,
            final_lr=1e-4,
            device=device,
            reconstruction_epochs=1,
            reconstruction_lr=1e-3,
        )
        acc = evaluate(model, test_loader, device)
        save_checkpoint(model, Path(args.checkpoint_dir) / f"{model_name}_{dataset_tag}_cp_layerwise_reconstruct_seed{seed}.pt")
        record("data_aware_layerwise_ft", acc, model)

    rows, _ = read_existing(rows_path)
    summarize(rows, summary_path)
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run CP compression experiments over multiple seeds.")
    parser.add_argument("--datasets", default="mnist,fashion-mnist")
    parser.add_argument("--models", default="lenet5,smallcnn")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="code/checkpoints")
    parser.add_argument("--output", default="code/results/multi_seed_results.csv")
    parser.add_argument("--summary-output", default="code/results/multi_seed_summary.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--als-iters", type=int, default=50)
    parser.add_argument("--force-baseline", action="store_true")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows_path = Path(args.output)
    summary_path = Path(args.summary_output)

    fieldnames = [
        "dataset",
        "model",
        "seed",
        "method",
        "rank",
        "accuracy",
        "baseline_accuracy",
        "baseline_gap_pp",
        "total_params",
        "conv_params",
        "baseline_total_params",
        "baseline_conv_params",
    ]
    rows, done = read_existing(rows_path)

    for dataset in datasets:
        for model_name in models:
            for seed in seeds:
                print()
                print(f"===== dataset={dataset} model={model_name} seed={seed} =====")
                run_one(args, dataset, model_name, seed, rows_path, summary_path, fieldnames, done)

    rows, _ = read_existing(rows_path)
    summarize(rows, summary_path)
    print(f"Wrote per-seed results: {rows_path}")
    print(f"Wrote summary results: {summary_path}")


if __name__ == "__main__":
    main()
