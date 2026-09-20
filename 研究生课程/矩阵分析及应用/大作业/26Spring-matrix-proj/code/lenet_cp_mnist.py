import argparse
import gzip
import math
import os
import random
import struct
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DATASETS = {
    "mnist": {
        "directory": "MNIST",
        "mean": 0.1307,
        "std": 0.3081,
        "urls": {
            "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        },
        "files": {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    },
    "fashion-mnist": {
        "directory": "FashionMNIST",
        "mean": 0.2860,
        "std": 0.3530,
        "urls": {
            "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        },
        "files": {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    },
}


class MNISTIdxDataset(Dataset):
    def __init__(self, root, dataset="mnist", train=True, download=True):
        self.root = Path(root)
        self.dataset_config = DATASETS[dataset]
        self.raw_dir = self.root / self.dataset_config["directory"] / "raw"
        if download:
            self._download()

        image_key = "train_images" if train else "test_images"
        label_key = "train_labels" if train else "test_labels"
        self.images = self._read_images(self.raw_dir / self.dataset_config["files"][image_key])
        self.labels = self._read_labels(self.raw_dir / self.dataset_config["files"][label_key])

        # Match torchvision.transforms.ToTensor() + dataset-specific Normalize((mean,), (std,)).
        self.images = (self.images.float() / 255.0 - self.dataset_config["mean"]) / self.dataset_config["std"]
        self.images = self.images.unsqueeze(1)

    def _download(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for key, url in self.dataset_config["urls"].items():
            dst = self.raw_dir / self.dataset_config["files"][key]
            if dst.exists():
                continue
            print(f"Downloading {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)

    @staticmethod
    def _read_images(path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid MNIST image file: {path}")
            data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).clone()
        return data.view(num, rows, cols)

    @staticmethod
    def _read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid MNIST label file: {path}")
            data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).clone()
        return data.long()

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu4(self.fc1(x))
        return self.fc2(x)


MODEL_REGISTRY = {
    "lenet5": {
        "factory": LeNet5,
        "conv_layers": ("conv1", "conv2"),
    },
    "smallcnn": {
        "factory": SmallCNN,
        "conv_layers": ("conv1", "conv2", "conv3"),
    },
}


def khatri_rao(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = (result.unsqueeze(1) * matrix.unsqueeze(0)).reshape(-1, result.shape[1])
    return result


def unfold(tensor, mode):
    dims = list(range(tensor.ndim))
    dims.pop(mode)
    return tensor.permute(mode, *dims).contiguous().view(tensor.shape[mode], -1)


def cp_als_decomposition(tensor, rank, n_iter_max=50):
    shape = tensor.shape
    factors = [torch.rand(dim, rank, device=tensor.device, dtype=tensor.dtype) for dim in shape]
    weights = torch.ones(rank, device=tensor.device, dtype=tensor.dtype)

    for _ in range(n_iter_max):
        for mode in range(len(shape)):
            other_modes = list(range(len(shape)))
            other_modes.pop(mode)
            other_modes.reverse()
            v = khatri_rao([factors[i] for i in other_modes])
            x_mode = unfold(tensor, mode)
            updated = x_mode @ v @ torch.linalg.pinv(v.T @ v)
            norms = torch.norm(updated, p=2, dim=0)
            nonzero = norms > 1e-12
            weights[nonzero] = norms[nonzero]
            updated[:, nonzero] /= norms[nonzero]
            factors[mode] = updated

    return weights, factors


def cp_decompose_conv_layer(layer, rank, n_iter_max=50):
    weights, factors = cp_als_decomposition(layer.weight.detach(), rank, n_iter_max=n_iter_max)
    out_factor, in_factor, h_factor, w_factor = factors
    out_factor = out_factor * weights

    pointwise_in = nn.Conv2d(in_factor.shape[0], rank, kernel_size=1, bias=False)
    depthwise_vertical = nn.Conv2d(
        rank,
        rank,
        kernel_size=(h_factor.shape[0], 1),
        padding=(layer.padding[0], 0),
        groups=rank,
        bias=False,
    )
    depthwise_horizontal = nn.Conv2d(
        rank,
        rank,
        kernel_size=(1, w_factor.shape[0]),
        stride=layer.stride,
        padding=(0, layer.padding[1]),
        groups=rank,
        bias=False,
    )
    pointwise_out = nn.Conv2d(rank, out_factor.shape[0], kernel_size=1, bias=True)

    pointwise_in.weight.data.copy_(in_factor.T.unsqueeze(-1).unsqueeze(-1))
    depthwise_vertical.weight.data.copy_(h_factor.T.unsqueeze(1).unsqueeze(-1))
    depthwise_horizontal.weight.data.copy_(w_factor.T.unsqueeze(1).unsqueeze(1))
    pointwise_out.weight.data.copy_(out_factor.unsqueeze(-1).unsqueeze(-1))
    if layer.bias is not None:
        pointwise_out.bias.data.copy_(layer.bias.data)
    else:
        pointwise_out.bias.data.zero_()

    return nn.Sequential(pointwise_in, depthwise_vertical, depthwise_horizontal, pointwise_out)


def materialize_cp_conv_layer(layer):
    if isinstance(layer, nn.Conv2d):
        dense = nn.Conv2d(
            layer.in_channels,
            layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
        )
        dense.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            dense.bias.data.copy_(layer.bias.data)
        return dense

    if not isinstance(layer, nn.Sequential) or len(layer) != 4:
        raise TypeError(f"Expected Conv2d or 4-layer CP Sequential, got {layer}")

    pointwise_in, depthwise_vertical, depthwise_horizontal, pointwise_out = layer
    in_factor = pointwise_in.weight.data.squeeze(-1).squeeze(-1)
    h_factor = depthwise_vertical.weight.data.squeeze(1).squeeze(-1)
    w_factor = depthwise_horizontal.weight.data.squeeze(1).squeeze(1)
    out_factor = pointwise_out.weight.data.squeeze(-1).squeeze(-1)

    dense_weight = torch.einsum("or,ri,rh,rw->oihw", out_factor, in_factor, h_factor, w_factor)
    dense = nn.Conv2d(
        pointwise_in.in_channels,
        pointwise_out.out_channels,
        kernel_size=(depthwise_vertical.kernel_size[0], depthwise_horizontal.kernel_size[1]),
        stride=depthwise_horizontal.stride,
        padding=(depthwise_vertical.padding[0], depthwise_horizontal.padding[1]),
        bias=pointwise_out.bias is not None,
    )
    dense.weight.data.copy_(dense_weight)
    if pointwise_out.bias is not None:
        dense.bias.data.copy_(pointwise_out.bias.data)
    return dense


def get_data_loaders(data_dir, batch_size, dataset="mnist", download=True):
    train_set = MNISTIdxDataset(data_dir, dataset=dataset, train=True, download=download)
    test_set = MNISTIdxDataset(data_dir, dataset=dataset, train=False, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
    model,
    train_loader,
    epochs,
    lr,
    device,
    weight_decay=0.0,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {running_loss / len(train_loader):.4f}, "
            f"Time: {time.time() - start:.1f}s"
        )


def reconstruct_cp_layer_activations(
    model,
    layer_name,
    dense_layer,
    train_loader,
    epochs,
    lr,
    device,
    max_batches=None,
):
    if epochs <= 0:
        return

    model.to(device)
    dense_layer.to(device)
    dense_layer.eval()
    for param in dense_layer.parameters():
        param.requires_grad_(False)

    cp_layer = getattr(model, layer_name)
    cp_layer.to(device)
    optimizer = optim.Adam(cp_layer.parameters(), lr=lr)

    print(f"Activation reconstruction for {layer_name}: epochs={epochs}, lr={lr}, max_batches={max_batches}")
    for epoch in range(epochs):
        model.eval()
        cp_layer.train()
        captured = {}

        def capture_input(_module, inputs):
            captured["input"] = inputs[0].detach()

        handle = cp_layer.register_forward_pre_hook(capture_input)
        running_loss = 0.0
        batches = 0
        start = time.time()

        for batch_idx, (inputs, _labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            captured.clear()
            with torch.no_grad():
                model(inputs)
            layer_inputs = captured["input"]
            with torch.no_grad():
                targets = dense_layer(layer_inputs)

            optimizer.zero_grad(set_to_none=True)
            outputs = cp_layer(layer_inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        handle.remove()
        print(
            f"Reconstruction epoch {epoch + 1}/{epochs}, "
            f"MSE: {running_loss / max(batches, 1):.6f}, "
            f"Time: {time.time() - start:.1f}s"
        )


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs).argmax(dim=1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def count_parameters(model, conv_layers):
    total = sum(p.numel() for p in model.parameters())
    conv = sum(p.numel() for name, p in model.named_parameters() if name.split(".", 1)[0] in conv_layers)
    return total, conv


def conv2d_output_hw(h, w, kernel_size, stride=1, padding=0, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    out_h = math.floor((h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return out_h, out_w


def module_flops(module, shape):
    if isinstance(module, nn.Conv2d):
        batch, channels, height, width = shape
        out_h, out_w = conv2d_output_hw(height, width, module.kernel_size, module.stride, module.padding, module.dilation)
        macs_per_output = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        outputs = module.out_channels * out_h * out_w
        flops = 2 * outputs * macs_per_output
        if module.bias is not None:
            flops += outputs
        return flops, (batch, module.out_channels, out_h, out_w)
    if isinstance(module, nn.ReLU):
        return math.prod(shape[1:]), shape
    if isinstance(module, nn.MaxPool2d):
        batch, channels, height, width = shape
        out_h, out_w = conv2d_output_hw(height, width, module.kernel_size, module.stride, module.padding, module.dilation)
        return channels * height * width, (batch, channels, out_h, out_w)
    if isinstance(module, nn.Linear):
        batch, features = shape
        return 2 * module.in_features * module.out_features, (batch, module.out_features)
    raise TypeError(f"Unsupported module for FLOPs: {module}")


def calculate_model_flops(model):
    shape = (1, 1, 28, 28)
    total = 0
    for module in model.children():
        if isinstance(module, nn.Sequential):
            for child in module.children():
                flops, shape = module_flops(child, shape)
                total += flops
        else:
            if isinstance(module, nn.Linear) and len(shape) == 4:
                shape = (shape[0], shape[1] * shape[2] * shape[3])
            flops, shape = module_flops(module, shape)
            total += flops
    return total / 1000.0


def print_summary(model, name, conv_layers):
    total_params, conv_params = count_parameters(model, conv_layers)
    kflops = calculate_model_flops(model)
    print(f"--- {name} Summary ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Conv Parameters: {conv_params:,}")
    print(f"FLOPs: {kflops:.4f} KFLOPs")
    return total_params, conv_params, kflops


def build_cp_model_from_baseline(model_factory, baseline, conv_layers, rank, als_iters):
    model = model_factory()
    model.load_state_dict(baseline.state_dict())
    for layer_name in conv_layers:
        layer = getattr(model, layer_name)
        setattr(model, layer_name, cp_decompose_conv_layer(layer, rank=rank, n_iter_max=als_iters))
    return model


def build_layerwise_finetuned_cp_model(
    model_factory,
    baseline,
    train_loader,
    conv_layers,
    rank,
    als_iters,
    layer_epochs,
    layer_lr,
    final_epochs,
    final_lr,
    device,
    reconstruction_epochs=0,
    reconstruction_lr=1e-3,
):
    model = model_factory()
    model.load_state_dict(baseline.state_dict())

    print()
    print("===== ITERATIVE LAYER-WISE FINETUNING =====")
    print(f"Rank: {rank}")
    print(f"Layer finetune: epochs={layer_epochs}, lr={layer_lr}")
    print(f"Final global finetune: epochs={final_epochs}, lr={final_lr}")
    if reconstruction_epochs > 0:
        print(
            "Activation reconstruction: "
            f"epochs={reconstruction_epochs}, lr={reconstruction_lr}"
        )

    for step, layer_name in enumerate(reversed(conv_layers), start=1):
        print(f"Layer step {step}: decomposing {layer_name}")
        layer = getattr(model, layer_name)
        dense_layer = materialize_cp_conv_layer(layer)
        setattr(model, layer_name, cp_decompose_conv_layer(layer, rank=rank, n_iter_max=als_iters))
        reconstruct_cp_layer_activations(
            model,
            layer_name,
            dense_layer,
            train_loader,
            epochs=reconstruction_epochs,
            lr=reconstruction_lr,
            device=device,
        )
        if layer_epochs > 0:
            train(
                model,
                train_loader,
                epochs=layer_epochs,
                lr=layer_lr,
                device=device,
            )

    if final_epochs > 0:
        print(f"Layer step {len(conv_layers) + 1}: final global finetune")
        train(
            model,
            train_loader,
            epochs=final_epochs,
            lr=final_lr,
            device=device,
        )

    return model


def main():
    parser = argparse.ArgumentParser(description="MNIST/Fashion-MNIST CNN baseline and CP decomposition check.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="lenet5")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--data-dir", default="data", help="Directory containing or receiving dataset raw files.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--als-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-finetune-epochs", type=int, default=0)
    parser.add_argument("--global-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--cp-checkpoint", default="code/checkpoints/lenet5_cp_global_ft.pt")
    parser.add_argument("--layerwise-finetune", action="store_true")
    parser.add_argument("--layerwise-layer-epochs", type=int, default=2)
    parser.add_argument("--layerwise-layer-lr", type=float, default=1e-3)
    parser.add_argument("--layerwise-final-epochs", type=int, default=5)
    parser.add_argument("--layerwise-final-lr", type=float, default=1e-4)
    parser.add_argument("--layerwise-checkpoint", default="code/checkpoints/lenet5_cp_layerwise_ft.pt")
    parser.add_argument("--layerwise-reconstruction-epochs", type=int, default=0)
    parser.add_argument("--layerwise-reconstruction-lr", type=float, default=1e-3)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-train", action="store_true", help="Load checkpoint and skip baseline training.")
    parser.add_argument("--smoke", action="store_true", help="Run model/FLOPs/CP checks without MNIST.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {args.dataset}")
    model_config = MODEL_REGISTRY[args.model]
    model_factory = model_config["factory"]
    conv_layers = model_config["conv_layers"]
    dataset_tag = args.dataset.replace("-", "_")
    checkpoint = Path(args.checkpoint or f"code/checkpoints/{args.model}_{dataset_tag}_seed{args.seed}.pt")

    if args.smoke:
        baseline = model_factory()
        print_summary(baseline, f"Baseline {args.model}", conv_layers)
        cp_model = build_cp_model_from_baseline(model_factory, baseline, conv_layers, args.rank, min(args.als_iters, 2))
        print_summary(cp_model, f"CP-Decomposed {args.model} (Rank {args.rank})", conv_layers)
        return

    train_loader, test_loader = get_data_loaders(
        args.data_dir,
        args.batch_size,
        dataset=args.dataset,
        download=args.download,
    )
    baseline = model_factory()

    if args.skip_train:
        baseline.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded baseline checkpoint: {checkpoint}")
    else:
        train(baseline, train_loader, epochs=args.epochs, lr=args.lr, device=device)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(baseline.state_dict(), checkpoint)
        print(f"Saved baseline checkpoint: {checkpoint}")

    base_acc = evaluate(baseline, test_loader, device)
    _, base_conv_params, base_kflops = print_summary(baseline, f"Baseline {args.model}", conv_layers)
    print(f"Baseline Test Accuracy: {base_acc:.2f}%")

    cp_model = build_cp_model_from_baseline(model_factory, baseline.cpu(), conv_layers, args.rank, args.als_iters)
    cp_acc = evaluate(cp_model, test_loader, device)
    _, cp_conv_params, cp_kflops = print_summary(cp_model, f"CP-Decomposed {args.model} (Rank {args.rank}, No Finetune)", conv_layers)
    print(f"CP-Decomposed Test Accuracy before finetuning: {cp_acc:.2f}%")

    if args.global_finetune_epochs > 0:
        print()
        print("===== GLOBAL FINETUNING CP-DECOMPOSED MODEL =====")
        print(f"Epochs: {args.global_finetune_epochs}, LR: {args.global_finetune_lr}")
        train(cp_model, train_loader, epochs=args.global_finetune_epochs, lr=args.global_finetune_lr, device=device)
        cp_ft_checkpoint = Path(args.cp_checkpoint)
        cp_ft_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cp_model.cpu().state_dict(), cp_ft_checkpoint)
        print(f"Saved CP global-finetuned checkpoint: {cp_ft_checkpoint}")
        cp_ft_acc = evaluate(cp_model, test_loader, device)
        print_summary(cp_model, f"CP-Decomposed {args.model} (Rank {args.rank}, Global Finetune)", conv_layers)
        print(f"CP-Decomposed Test Accuracy after global finetuning: {cp_ft_acc:.2f}%")

    if args.layerwise_finetune:
        layerwise_model = build_layerwise_finetuned_cp_model(
            model_factory,
            baseline.cpu(),
            train_loader,
            conv_layers,
            rank=args.rank,
            als_iters=args.als_iters,
            layer_epochs=args.layerwise_layer_epochs,
            layer_lr=args.layerwise_layer_lr,
            final_epochs=args.layerwise_final_epochs,
            final_lr=args.layerwise_final_lr,
            device=device,
            reconstruction_epochs=args.layerwise_reconstruction_epochs,
            reconstruction_lr=args.layerwise_reconstruction_lr,
        )
        layerwise_checkpoint = Path(args.layerwise_checkpoint)
        layerwise_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(layerwise_model.cpu().state_dict(), layerwise_checkpoint)
        print(f"Saved CP layer-wise finetuned checkpoint: {layerwise_checkpoint}")
        layerwise_acc = evaluate(layerwise_model, test_loader, device)
        print_summary(layerwise_model, f"CP-Decomposed {args.model} (Rank {args.rank}, Layer-wise Finetune)", conv_layers)
        print(f"CP-Decomposed Test Accuracy after layer-wise finetuning: {layerwise_acc:.2f}%")

    print()
    print("===== UNIFORM-RANK COMPRESSION CHECK =====")
    print(f"Conv parameter ratio: {base_conv_params / cp_conv_params:.2f}x")
    print(f"FLOPs reduced: {(1.0 - cp_kflops / base_kflops) * 100:.2f}%")


if __name__ == "__main__":
    main()
