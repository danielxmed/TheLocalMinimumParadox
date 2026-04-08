"""
utils_v1.py -- Shared experimental utilities for The Local Minimum Paradox project.

Provides reproducibility helpers, architecture factories, dataset generators,
training loops, Hessian analysis, loss surface visualization, mode connectivity,
NTK computation, CKA, and I/O utilities.

Target environment: Intel Mac, 16 GB RAM, CPU only.
    Python 3.12.7, PyTorch 2.2.2, torchvision 0.17.2
"""

import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# 1. Seed & Reproducibility
# ---------------------------------------------------------------------------

SEEDS: List[int] = [42, 137, 256, 512, 1024]
DEVICE: str = "cpu"


def set_seed(seed: int) -> None:
    """Set random seed across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. Architecture Factory
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Residual block: Conv -> BN -> ReLU -> Conv -> BN + skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class _ResNetTiny(nn.Module):
    """Tiny ResNet with 2 residual blocks, ~20-50K parameters."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.block1 = _ResidualBlock(16)
        self.block2 = _ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class _SimpleCNN(nn.Module):
    """Simple CNN for 28x28 single-channel inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden: int,
    depth: int,
    bias: bool,
) -> nn.Sequential:
    """Build a variable-depth MLP with ReLU activations."""
    layers: List[nn.Module] = []
    in_features = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_features, hidden, bias=bias))
        layers.append(nn.ReLU())
        in_features = hidden
    layers.append(nn.Linear(in_features, output_dim, bias=bias))
    return nn.Sequential(*layers)


def make_model(
    arch: str,
    input_dim: int,
    output_dim: int,
    **kwargs: Any,
) -> nn.Module:
    """
    Architecture factory.

    Args:
        arch: One of 'tiny_mlp', 'mlp_2layer', 'mlp_deep', 'mlp_2layer_nobias',
              'mlp_deep_nobias', 'simple_cnn', 'resnet_tiny'.
        input_dim: Input feature dimension (ignored for CNN architectures).
        output_dim: Number of output classes / regression targets.
        **kwargs: Architecture-specific keyword arguments (hidden, depth, in_channels).

    Returns:
        An nn.Module on DEVICE.
    """
    if arch == "tiny_mlp":
        hidden = kwargs.get("hidden", 10)
        model = _build_mlp(input_dim, output_dim, hidden=hidden, depth=1, bias=True)

    elif arch == "mlp_2layer":
        hidden = kwargs.get("hidden", 256)
        model = _build_mlp(input_dim, output_dim, hidden=hidden, depth=1, bias=True)

    elif arch == "mlp_deep":
        hidden = kwargs.get("hidden", 128)
        depth = kwargs.get("depth", 4)
        model = _build_mlp(input_dim, output_dim, hidden=hidden, depth=depth, bias=True)

    elif arch == "mlp_2layer_nobias":
        hidden = kwargs.get("hidden", 256)
        model = _build_mlp(input_dim, output_dim, hidden=hidden, depth=1, bias=False)

    elif arch == "mlp_deep_nobias":
        hidden = kwargs.get("hidden", 128)
        depth = kwargs.get("depth", 4)
        model = _build_mlp(input_dim, output_dim, hidden=hidden, depth=depth, bias=False)

    elif arch == "simple_cnn":
        model = _SimpleCNN(num_classes=output_dim)

    elif arch == "resnet_tiny":
        in_channels = kwargs.get("in_channels", 1)
        model = _ResNetTiny(num_classes=output_dim, in_channels=in_channels)

    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Supported: tiny_mlp, mlp_2layer, "
            f"mlp_deep, mlp_2layer_nobias, mlp_deep_nobias, simple_cnn, resnet_tiny."
        )

    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. Dataset Factory
# ---------------------------------------------------------------------------

def _generate_gaussian_mixture(
    n: int,
    d: int = 20,
    K: int = 5,
    separation: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate K-class Gaussian mixture in R^d."""
    # Place class centers on a sphere of radius separation * sqrt(d)
    radius = separation * np.sqrt(d)
    centers = torch.randn(K, d)
    centers = centers / centers.norm(dim=1, keepdim=True) * radius

    samples_per_class = n // K
    remainder = n - samples_per_class * K

    X_parts: List[torch.Tensor] = []
    Y_parts: List[torch.Tensor] = []
    for k in range(K):
        count = samples_per_class + (1 if k < remainder else 0)
        X_k = centers[k].unsqueeze(0) + torch.randn(count, d)
        Y_k = torch.full((count,), k, dtype=torch.long)
        X_parts.append(X_k)
        Y_parts.append(Y_k)

    X = torch.cat(X_parts, dim=0)
    Y = torch.cat(Y_parts, dim=0)

    # Shuffle
    perm = torch.randperm(X.size(0))
    return X[perm], Y[perm]


def _generate_xor_2d(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 2D XOR dataset with n points."""
    X = torch.randn(n, 2)
    Y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).long()
    return X, Y


def _generate_concentric_spheres(
    n: int,
    d: int = 10,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two concentric spheres in R^d."""
    n_inner = n // 2
    n_outer = n - n_inner

    # Inner sphere
    X_inner = torch.randn(n_inner, d)
    X_inner = X_inner / X_inner.norm(dim=1, keepdim=True) * r_inner
    X_inner = X_inner + torch.randn_like(X_inner) * 0.1

    # Outer sphere
    X_outer = torch.randn(n_outer, d)
    X_outer = X_outer / X_outer.norm(dim=1, keepdim=True) * r_outer
    X_outer = X_outer + torch.randn_like(X_outer) * 0.1

    X = torch.cat([X_inner, X_outer], dim=0)
    Y = torch.cat([torch.zeros(n_inner), torch.ones(n_outer)]).long()

    perm = torch.randperm(X.size(0))
    return X[perm], Y[perm]


def _load_mnist(
    n_train: int,
    n_test: Optional[int],
    flatten: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST via torchvision, normalize to [0,1]."""
    import torchvision
    import torchvision.transforms as transforms

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.ToTensor()  # already scales to [0,1]
    train_ds = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform,
    )

    X_train = train_ds.data[:n_train].float() / 255.0
    Y_train = train_ds.targets[:n_train]

    n_test_actual = n_test if n_test is not None else len(test_ds)
    X_test = test_ds.data[:n_test_actual].float() / 255.0
    Y_test = test_ds.targets[:n_test_actual]

    if flatten:
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
    else:
        # Add channel dimension for CNNs: (N, 1, 28, 28)
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    return X_train, Y_train, X_test, Y_test


def _load_cifar10(
    n_train: int,
    n_test: Optional[int],
    flatten: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 via torchvision, normalize to [0,1]."""
    import torchvision
    import torchvision.transforms as transforms

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.ToTensor()
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform,
    )

    # torchvision CIFAR10 stores data as numpy array (N, 32, 32, 3)
    X_train = torch.tensor(train_ds.data[:n_train], dtype=torch.float32) / 255.0
    Y_train = torch.tensor(train_ds.targets[:n_train], dtype=torch.long)

    n_test_actual = n_test if n_test is not None else len(test_ds)
    X_test = torch.tensor(test_ds.data[:n_test_actual], dtype=torch.float32) / 255.0
    Y_test = torch.tensor(test_ds.targets[:n_test_actual], dtype=torch.long)

    if flatten:
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
    else:
        # Convert to (N, C, H, W) for CNNs
        X_train = X_train.permute(0, 3, 1, 2)
        X_test = X_test.permute(0, 3, 1, 2)

    return X_train, Y_train, X_test, Y_test


def make_dataset(
    name: str,
    n_train: int,
    n_test: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dataset factory.

    Args:
        name: One of 'gaussian_mixture', 'mnist', 'cifar10', 'xor_2d',
              'concentric_spheres'.
        n_train: Number of training samples.
        n_test: Number of test samples (defaults vary by dataset).
        **kwargs: Dataset-specific keyword arguments.

    Returns:
        (X_train, Y_train, X_test, Y_test) as tensors on DEVICE.
    """
    if name == "gaussian_mixture":
        d = kwargs.get("d", 20)
        K = kwargs.get("K", 5)
        separation = kwargs.get("separation", 2.0)
        X_train, Y_train = _generate_gaussian_mixture(n_train, d=d, K=K, separation=separation)
        n_test_actual = n_test if n_test is not None else max(n_train // 5, 100)
        X_test, Y_test = _generate_gaussian_mixture(n_test_actual, d=d, K=K, separation=separation)

    elif name == "mnist":
        flatten = kwargs.get("flatten", True)
        return tuple(t.to(DEVICE) for t in _load_mnist(n_train, n_test, flatten=flatten))

    elif name == "cifar10":
        flatten = kwargs.get("flatten", True)
        return tuple(t.to(DEVICE) for t in _load_cifar10(n_train, n_test, flatten=flatten))

    elif name == "xor_2d":
        X_train, Y_train = _generate_xor_2d(n_train)
        n_test_actual = n_test if n_test is not None else max(n_train // 5, 100)
        X_test, Y_test = _generate_xor_2d(n_test_actual)

    elif name == "concentric_spheres":
        d = kwargs.get("d", 10)
        r_inner = kwargs.get("r_inner", 1.0)
        r_outer = kwargs.get("r_outer", 2.0)
        X_train, Y_train = _generate_concentric_spheres(n_train, d=d, r_inner=r_inner, r_outer=r_outer)
        n_test_actual = n_test if n_test is not None else max(n_train // 5, 100)
        X_test, Y_test = _generate_concentric_spheres(n_test_actual, d=d, r_inner=r_inner, r_outer=r_outer)

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: gaussian_mixture, mnist, "
            f"cifar10, xor_2d, concentric_spheres."
        )

    return (
        X_train.to(DEVICE),
        Y_train.to(DEVICE),
        X_test.to(DEVICE),
        Y_test.to(DEVICE),
    )


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    lr: float,
    n_steps: int,
    loss_fn: Optional[nn.Module] = None,
    optimizer_type: str = "sgd",
    batch_size: Optional[int] = None,
    callback: Optional[Callable[[nn.Module, int, float], Optional[Dict[str, Any]]]] = None,
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    """
    Training loop supporting full-batch GD and mini-batch SGD.

    Args:
        model: Neural network to train.
        X: Input tensor.
        Y: Target tensor.
        lr: Learning rate.
        n_steps: Number of optimization steps.
        loss_fn: Loss function (defaults to CrossEntropyLoss).
        optimizer_type: 'sgd' or 'adam'.
        batch_size: Mini-batch size. None means full-batch GD.
        callback: Optional function (model, step, loss) -> dict of extra metrics.
        log_every: How often to log and print progress.

    Returns:
        List of log dicts with at least 'step' and 'loss' keys.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type '{optimizer_type}'. Use 'sgd' or 'adam'.")

    n_samples = X.size(0)
    use_minibatch = batch_size is not None and batch_size < n_samples
    log: List[Dict[str, Any]] = []

    model.train()
    for step in range(1, n_steps + 1):
        if use_minibatch:
            indices = torch.randint(0, n_samples, (batch_size,))
            X_batch = X[indices]
            Y_batch = Y[indices]
        else:
            X_batch = X
            Y_batch = Y

        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, Y_batch)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == n_steps:
            loss_val = loss.item()
            entry: Dict[str, Any] = {"step": step, "loss": loss_val}

            if callback is not None:
                extra = callback(model, step, loss_val)
                if extra is not None:
                    entry.update(extra)

            log.append(entry)
            print(f"  step {step:5d}/{n_steps} | loss = {loss_val:.6f}")

    return log


# ---------------------------------------------------------------------------
# 5. Hessian Module
# ---------------------------------------------------------------------------

def hessian_vector_product(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    v: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute the Hessian-vector product Hv without forming the full Hessian.

    Args:
        model: Neural network.
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.
        v: List of tensors matching model.parameters() shapes.

    Returns:
        List of tensors representing Hv.
    """
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]

    # Forward pass and compute gradients
    output = model(X)
    loss = loss_fn(output, Y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Compute gradient-vector dot product
    gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))

    # Second backward pass for Hv
    hvp = torch.autograd.grad(gv, params, create_graph=False)
    hvp_detached = [h.detach().clone() for h in hvp]

    model.zero_grad()
    return hvp_detached


def _params_to_vec(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of parameter tensors into a single vector."""
    return torch.cat([p.detach().reshape(-1) for p in params])


def _vec_to_params(vec: torch.Tensor, params: List[torch.Tensor]) -> List[torch.Tensor]:
    """Split a flat vector into a list of tensors matching parameter shapes."""
    result: List[torch.Tensor] = []
    offset = 0
    for p in params:
        numel = p.numel()
        result.append(vec[offset : offset + numel].reshape(p.shape))
        offset += numel
    return result


def full_hessian(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """
    Compute the full NxN Hessian matrix via N Hessian-vector products.

    Only feasible for models with < 5000 parameters.

    Args:
        model: Neural network.
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.

    Returns:
        Hessian matrix of shape (N, N).

    Raises:
        ValueError: If the model has >= 5000 parameters.
    """
    n_params = count_parameters(model)
    if n_params >= 5000:
        raise ValueError(
            f"full_hessian requires < 5000 parameters, model has {n_params}. "
            f"Use top_hessian_eigenvalue or hessian_eigenvalues_lanczos instead."
        )

    params = [p for p in model.parameters() if p.requires_grad]
    H = torch.zeros(n_params, n_params)

    for i in range(n_params):
        # Standard basis vector e_i
        e_i = torch.zeros(n_params)
        e_i[i] = 1.0
        v = _vec_to_params(e_i, params)

        hvp = hessian_vector_product(model, X, Y, loss_fn, v)
        H[:, i] = torch.cat([h.reshape(-1) for h in hvp])

    return H


def top_hessian_eigenvalue(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    n_iter: int = 50,
) -> float:
    """
    Estimate the top eigenvalue of the Hessian via power iteration.

    Args:
        model: Neural network.
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.
        n_iter: Number of power iteration steps.

    Returns:
        Estimated top eigenvalue (float).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    # Initialize random vector
    v_flat = torch.randn(n_params)
    v_flat = v_flat / v_flat.norm()

    for _ in range(n_iter):
        v_list = _vec_to_params(v_flat, params)
        hvp = hessian_vector_product(model, X, Y, loss_fn, v_list)
        hv_flat = torch.cat([h.reshape(-1) for h in hvp])

        norm = hv_flat.norm()
        if norm < 1e-12:
            return 0.0
        v_flat = hv_flat / norm

    # Rayleigh quotient: v^T H v
    v_list = _vec_to_params(v_flat, params)
    hvp = hessian_vector_product(model, X, Y, loss_fn, v_list)
    hv_flat = torch.cat([h.reshape(-1) for h in hvp])
    eigenvalue = torch.dot(v_flat, hv_flat).item()

    return eigenvalue


def hessian_eigenvalues_lanczos(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Estimate Hessian eigenvalues via the Lanczos algorithm.

    Builds a tridiagonal matrix T via n_iter Lanczos steps with
    reorthogonalization, then returns eigenvalues of T.

    Args:
        model: Neural network.
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.
        n_iter: Number of Lanczos iterations.

    Returns:
        Sorted numpy array of estimated eigenvalues.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    # Clamp iterations to number of parameters
    m = min(n_iter, n_params)

    # Lanczos vectors stored as columns for reorthogonalization
    Q = torch.zeros(n_params, m + 1)
    alpha = torch.zeros(m)  # Diagonal of T
    beta = torch.zeros(m)   # Off-diagonal of T

    # Initialize
    q = torch.randn(n_params)
    q = q / q.norm()
    Q[:, 0] = q

    for j in range(m):
        # Hessian-vector product
        v_list = _vec_to_params(Q[:, j], params)
        hvp = hessian_vector_product(model, X, Y, loss_fn, v_list)
        w = torch.cat([h.reshape(-1) for h in hvp])

        # Compute alpha_j
        alpha[j] = torch.dot(w, Q[:, j])

        # Orthogonalize against previous two vectors
        if j == 0:
            w = w - alpha[j] * Q[:, j]
        else:
            w = w - alpha[j] * Q[:, j] - beta[j - 1] * Q[:, j - 1]

        # Full reorthogonalization for numerical stability
        for k in range(j + 1):
            coeff = torch.dot(w, Q[:, k])
            w = w - coeff * Q[:, k]

        beta_j = w.norm()

        if beta_j < 1e-12:
            # Invariant subspace found; truncate
            alpha = alpha[: j + 1]
            beta = beta[:j]
            break

        beta[j] = beta_j
        Q[:, j + 1] = w / beta_j

    # Build tridiagonal matrix T and compute its eigenvalues
    m_actual = len(alpha)
    T = torch.zeros(m_actual, m_actual)
    for i in range(m_actual):
        T[i, i] = alpha[i]
    for i in range(len(beta)):
        if i < m_actual - 1:
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

    eigenvalues = torch.linalg.eigvalsh(T).numpy()
    eigenvalues.sort()
    return eigenvalues


# ---------------------------------------------------------------------------
# 6. Loss Surface Visualization
# ---------------------------------------------------------------------------

def loss_surface_2d(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    grid_size: int = 51,
    range_scale: float = 1.0,
    directions: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D loss surface by perturbing model parameters along two
    random orthonormal directions with filter normalization.

    Args:
        model: Neural network (at the point to center the plot on).
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.
        grid_size: Number of grid points per dimension.
        range_scale: Extent of the grid (in units of parameter norm).
        directions: Optional pre-computed (dir1, dir2) as lists of tensors.

    Returns:
        (alphas, betas, losses) where alphas and betas are 1D arrays and
        losses is a 2D array of shape (grid_size, grid_size).
    """
    params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    if directions is not None:
        dir1, dir2 = directions
    else:
        # Generate two random directions
        dir1 = [torch.randn_like(p) for p in params]
        dir2 = [torch.randn_like(p) for p in params]

        # Gram-Schmidt orthogonalization
        dot = sum(torch.sum(d1 * d2) for d1, d2 in zip(dir1, dir2))
        norm1_sq = sum(torch.sum(d1 * d1) for d1 in dir1)
        dir2 = [d2 - (dot / norm1_sq) * d1 for d1, d2 in zip(dir1, dir2)]

    # Filter normalization: normalize each direction by ||theta_l|| / ||d_l||
    for i, p in enumerate(params):
        p_norm = p.norm()
        d1_norm = dir1[i].norm()
        d2_norm = dir2[i].norm()
        if d1_norm > 1e-10:
            dir1[i] = dir1[i] * (p_norm / d1_norm)
        if d2_norm > 1e-10:
            dir2[i] = dir2[i] * (p_norm / d2_norm)

    # Normalize to unit vectors after filter normalization
    total_norm1 = torch.sqrt(sum(torch.sum(d * d) for d in dir1))
    total_norm2 = torch.sqrt(sum(torch.sum(d * d) for d in dir2))
    if total_norm1 > 1e-10:
        dir1 = [d / total_norm1 for d in dir1]
    if total_norm2 > 1e-10:
        dir2 = [d / total_norm2 for d in dir2]

    alphas = np.linspace(-range_scale, range_scale, grid_size)
    betas = np.linspace(-range_scale, range_scale, grid_size)
    losses = np.zeros((grid_size, grid_size))

    model.eval()
    model_params = [p for p in model.parameters() if p.requires_grad]

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                # Set parameters to theta + a * dir1 + b * dir2
                for p, p0, d1, d2 in zip(model_params, params, dir1, dir2):
                    p.copy_(p0 + a * d1 + b * d2)

                output = model(X)
                losses[i, j] = loss_fn(output, Y).item()

        # Restore original parameters
        for p, p0 in zip(model_params, params):
            p.copy_(p0)

    model.train()
    return alphas, betas, losses


# ---------------------------------------------------------------------------
# 7. Mode Connectivity
# ---------------------------------------------------------------------------

def linear_interpolation_loss(
    model1: nn.Module,
    model2: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    n_points: int = 21,
) -> List[Tuple[float, float, float]]:
    """
    Evaluate loss along the linear path between two models.

    Args:
        model1: Starting model.
        model2: Ending model.
        X: Input tensor.
        Y: Target tensor.
        loss_fn: Loss function.
        n_points: Number of interpolation points.

    Returns:
        List of (alpha, loss, accuracy) tuples.
    """
    params1 = [p.detach().clone() for p in model1.parameters()]
    params2 = [p.detach().clone() for p in model2.parameters()]

    # Create a copy of model1 for evaluation
    import copy
    interp_model = copy.deepcopy(model1)
    interp_params = list(interp_model.parameters())

    results: List[Tuple[float, float, float]] = []

    interp_model.eval()
    with torch.no_grad():
        for alpha in np.linspace(0.0, 1.0, n_points):
            # theta(alpha) = (1-alpha)*theta1 + alpha*theta2
            for p, p1, p2 in zip(interp_params, params1, params2):
                p.copy_((1.0 - alpha) * p1 + alpha * p2)

            output = interp_model(X)
            loss_val = loss_fn(output, Y).item()

            # Compute accuracy for classification
            preds = output.argmax(dim=1)
            acc = (preds == Y).float().mean().item()
            results.append((float(alpha), loss_val, acc))

    return results


def permutation_align(
    model1: nn.Module,
    model2: nn.Module,
    X: torch.Tensor,
) -> nn.Module:
    """
    Align model2's hidden units to model1 via activation matching.

    For each hidden layer, computes the activation correlation matrix and
    solves the assignment problem using the Hungarian algorithm. Applies
    the resulting permutation to model2's weights and biases.

    Args:
        model1: Reference model.
        model2: Model to be permuted (a deep copy is returned).
        X: Input data used to compute activations.

    Returns:
        A deep copy of model2 with permuted hidden-layer weights/biases.
    """
    from scipy.optimize import linear_sum_assignment
    import copy

    model2_aligned = copy.deepcopy(model2)

    # Collect linear layers (for Sequential MLPs)
    def _get_linear_layers(model: nn.Module) -> List[nn.Linear]:
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
        return layers

    layers1 = _get_linear_layers(model1)
    layers2 = _get_linear_layers(model2_aligned)

    if len(layers1) != len(layers2):
        warnings.warn(
            "Models have different numbers of Linear layers; returning unaligned copy."
        )
        return model2_aligned

    if len(layers1) < 2:
        # No hidden layer to permute
        return model2_aligned

    # Compute activations at each hidden layer
    def _get_activations(model: nn.Module, layers: List[nn.Linear], X: torch.Tensor) -> List[torch.Tensor]:
        activations: List[torch.Tensor] = []
        hooks = []

        def _make_hook(storage: List[torch.Tensor]):
            def hook(module, input, output):
                storage.append(output.detach())
            return hook

        # Register hooks on all but the last linear layer
        for layer in layers[:-1]:
            act_storage: List[torch.Tensor] = []
            h = layer.register_forward_hook(_make_hook(act_storage))
            hooks.append((h, act_storage))

        model.eval()
        with torch.no_grad():
            model(X)

        for h, act_storage in hooks:
            h.remove()
            if act_storage:
                activations.append(act_storage[0])

        return activations

    acts1 = _get_activations(model1, layers1, X)
    acts2 = _get_activations(model2_aligned, layers2, X)

    # Align layer by layer from input to output
    for layer_idx in range(len(acts1)):
        a1 = acts1[layer_idx]  # (N, hidden_dim)
        a2 = acts2[layer_idx]  # (N, hidden_dim)

        if a1.shape[1] != a2.shape[1]:
            continue  # Skip mismatched dimensions

        hidden_dim = a1.shape[1]

        # Compute correlation (cost) matrix
        # Normalize activations
        a1_centered = a1 - a1.mean(dim=0, keepdim=True)
        a2_centered = a2 - a2.mean(dim=0, keepdim=True)

        a1_norm = a1_centered / (a1_centered.norm(dim=0, keepdim=True) + 1e-8)
        a2_norm = a2_centered / (a2_centered.norm(dim=0, keepdim=True) + 1e-8)

        # Correlation matrix: maximize correlation = minimize negative correlation
        C = (a1_norm.T @ a2_norm).numpy()
        cost = -C  # Negate for minimization

        row_ind, col_ind = linear_sum_assignment(cost)

        # Build permutation: col_ind[i] tells which neuron in model2
        # should map to neuron i
        perm = col_ind

        # Apply permutation to the current layer's output and next layer's input
        curr_layer = layers2[layer_idx]
        next_layer = layers2[layer_idx + 1]

        # Permute outgoing weights and bias of current layer
        with torch.no_grad():
            curr_layer.weight.copy_(curr_layer.weight[perm])
            if curr_layer.bias is not None:
                curr_layer.bias.copy_(curr_layer.bias[perm])

            # Permute incoming weights of next layer
            next_layer.weight.copy_(next_layer.weight[:, perm])

    return model2_aligned


# ---------------------------------------------------------------------------
# 8. NTK Computation
# ---------------------------------------------------------------------------

def compute_ntk(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Compute the empirical Neural Tangent Kernel matrix.

    K[i,j] = sum_p (dF(x_i)/dtheta_p) * (dF(x_j)/dtheta_p)

    Args:
        model: Neural network.
        X: Input tensor of shape (n, ...).

    Returns:
        NTK matrix of shape (n, n).
    """
    n = X.size(0)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    if n > 200:
        warnings.warn(f"compute_ntk: n={n} > 200. This will be slow and memory-intensive.")
    if n_params > 100_000:
        warnings.warn(f"compute_ntk: {n_params} params > 100K. This will be slow.")

    model.eval()

    # Compute per-sample Jacobians
    jacobians = torch.zeros(n, n_params)

    for i in range(n):
        model.zero_grad()
        x_i = X[i : i + 1]
        output = model(x_i)

        # For multi-output, take the sum of all outputs for Jacobian
        output_sum = output.sum()
        output_sum.backward()

        grad_vec = torch.cat([
            p.grad.detach().reshape(-1) for p in params
        ])
        jacobians[i] = grad_vec

    model.zero_grad()
    model.train()

    # K = J @ J.T
    ntk = jacobians @ jacobians.T
    return ntk


# ---------------------------------------------------------------------------
# 9. CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X1: torch.Tensor, X2: torch.Tensor) -> float:
    """
    Compute linear CKA (Centered Kernel Alignment) between two representations.

    Based on the HSIC formulation:
        CKA(X1, X2) = HSIC(X1, X2) / sqrt(HSIC(X1, X1) * HSIC(X2, X2))

    Args:
        X1: Representation matrix of shape (n, d1).
        X2: Representation matrix of shape (n, d2).

    Returns:
        CKA similarity in [0, 1].
    """
    assert X1.size(0) == X2.size(0), "X1 and X2 must have the same number of samples."
    n = X1.size(0)

    # Center the representations
    X1_c = X1 - X1.mean(dim=0, keepdim=True)
    X2_c = X2 - X2.mean(dim=0, keepdim=True)

    # Linear HSIC: trace(K1 H K2 H) = ||X2_c^T X1_c||_F^2
    # where K1 = X1 X1^T, K2 = X2 X2^T, H = I - 1/n
    hsic_12 = torch.norm(X2_c.T @ X1_c, p="fro") ** 2
    hsic_11 = torch.norm(X1_c.T @ X1_c, p="fro") ** 2
    hsic_22 = torch.norm(X2_c.T @ X2_c, p="fro") ** 2

    denom = torch.sqrt(hsic_11 * hsic_22)
    if denom < 1e-12:
        return 0.0

    return (hsic_12 / denom).item()


# ---------------------------------------------------------------------------
# 10. I/O and Plotting
# ---------------------------------------------------------------------------

def setup_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    params = {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "font.family": "serif",
        "savefig.dpi": 300,
    }
    # savefig.bbox_inches is not a valid rcParam in all matplotlib versions;
    # set it only if supported.
    if "savefig.bbox_inches" in plt.rcParams:
        params["savefig.bbox_inches"] = "tight"
    plt.rcParams.update(params)


def save_results(
    results_dict: Dict[str, Any],
    experiment_dir: str,
    config: Dict[str, Any],
) -> None:
    """
    Save experiment results and configuration.

    Saves results.json, config.json, and optionally raw_data.npy
    if results_dict contains a 'raw_data' key with a numpy array.

    Args:
        results_dict: Dictionary of results to save.
        experiment_dir: Directory to save into (created if needed).
        config: Configuration dictionary.
    """
    os.makedirs(experiment_dir, exist_ok=True)

    # Separate raw numpy data from JSON-serializable results
    json_results = {}
    for k, v in results_dict.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(experiment_dir, f"{k}.npy"), v)
        elif isinstance(v, torch.Tensor):
            np.save(os.path.join(experiment_dir, f"{k}.npy"), v.numpy())
        elif isinstance(v, (int, float, str, bool, list, dict)):
            json_results[k] = v
        else:
            # Try to convert; skip if not possible
            try:
                json_results[k] = str(v)
            except Exception:
                pass

    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Results saved to {experiment_dir}")


def load_results(
    experiment_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load experiment results and configuration from a directory.

    Args:
        experiment_dir: Directory containing results.json and config.json.

    Returns:
        (results_dict, config_dict) tuple.
    """
    results_path = os.path.join(experiment_dir, "results.json")
    config_path = os.path.join(experiment_dir, "config.json")

    with open(results_path, "r") as f:
        results = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load any .npy files
    for fname in os.listdir(experiment_dir):
        if fname.endswith(".npy"):
            key = fname[:-4]
            results[key] = np.load(os.path.join(experiment_dir, fname))

    return results, config


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str,
) -> None:
    """
    Save a matplotlib figure as both PDF and PNG at 300 DPI.

    Args:
        fig: Matplotlib figure object.
        name: Base filename (without extension).
        output_dir: Directory to save into (created if needed).
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")

    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Figure saved: {pdf_path}, {png_path}")


def get_hardware_info() -> Dict[str, str]:
    """
    Return a dictionary with hardware and software environment information.

    Returns:
        Dict with platform, processor, cores, ram, device, pytorch_version,
        python_version.
    """
    try:
        import psutil
        ram_gb = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except ImportError:
        ram_gb = "unknown (install psutil)"

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cores": str(os.cpu_count()),
        "ram": ram_gb,
        "device": DEVICE,
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
        "cuda_available": str(torch.cuda.is_available()),
    }


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("utils_v1.py -- Smoke Test")
    print("=" * 60)

    # Hardware info
    hw = get_hardware_info()
    print("\n--- Hardware Info ---")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    # Reproducibility
    set_seed(42)
    print("\n--- Seed set to 42 ---")

    # Create model
    model = make_model("tiny_mlp", input_dim=20, output_dim=5, hidden=10)
    n_params = count_parameters(model)
    print(f"\n--- Model: tiny_mlp ---")
    print(f"  Parameters: {n_params}")
    print(f"  Architecture:\n{model}")

    # Generate data
    print("\n--- Generating gaussian_mixture dataset ---")
    X_train, Y_train, X_test, Y_test = make_dataset(
        "gaussian_mixture", n_train=200, n_test=50, d=20, K=5, separation=2.0,
    )
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")
    print(f"  Classes: {Y_train.unique().tolist()}")

    # Train
    print("\n--- Training for 50 steps ---")
    loss_fn = nn.CrossEntropyLoss()
    log = train(model, X_train, Y_train, lr=0.01, n_steps=50, loss_fn=loss_fn, log_every=10)
    print(f"  Final loss: {log[-1]['loss']:.6f}")

    # Hessian
    print("\n--- Hessian Analysis ---")
    top_eig = top_hessian_eigenvalue(model, X_train, Y_train, loss_fn, n_iter=30)
    print(f"  Top Hessian eigenvalue (power iteration): {top_eig:.6f}")

    print("\n--- Full Hessian (tiny model) ---")
    H = full_hessian(model, X_train, Y_train, loss_fn)
    eigs_full = torch.linalg.eigvalsh(H).numpy()
    print(f"  Hessian shape: {H.shape}")
    print(f"  Top 5 eigenvalues: {eigs_full[-5:]}")
    print(f"  Bottom 5 eigenvalues: {eigs_full[:5]}")
    print(f"  Num negative eigenvalues: {(eigs_full < -1e-6).sum()}")

    print("\n--- Lanczos eigenvalues ---")
    eigs_lanczos = hessian_eigenvalues_lanczos(model, X_train, Y_train, loss_fn, n_iter=50)
    print(f"  Lanczos estimates (top 5): {eigs_lanczos[-5:]}")

    # NTK on small subset
    print("\n--- NTK (10 samples) ---")
    ntk = compute_ntk(model, X_train[:10])
    print(f"  NTK shape: {ntk.shape}")
    print(f"  NTK trace: {ntk.trace().item():.4f}")

    # CKA
    print("\n--- Linear CKA ---")
    R1 = torch.randn(50, 20)
    R2 = R1 + 0.1 * torch.randn(50, 20)  # Slightly perturbed
    cka_val = linear_cka(R1, R2)
    print(f"  CKA(R1, R1+noise): {cka_val:.4f} (should be close to 1.0)")

    # XOR dataset
    print("\n--- XOR 2D Dataset ---")
    X_xor, Y_xor, X_xor_test, Y_xor_test = make_dataset("xor_2d", n_train=500, n_test=100)
    print(f"  X_xor: {X_xor.shape}, classes: {Y_xor.unique().tolist()}")

    # Concentric spheres
    print("\n--- Concentric Spheres ---")
    X_cs, Y_cs, _, _ = make_dataset("concentric_spheres", n_train=300, d=10)
    print(f"  X_cs: {X_cs.shape}, classes: {Y_cs.unique().tolist()}")

    # Architecture listing
    print("\n--- Architecture Parameter Counts ---")
    for arch, kw in [
        ("tiny_mlp", {"hidden": 10}),
        ("mlp_2layer", {"hidden": 256}),
        ("mlp_deep", {"hidden": 128, "depth": 4}),
        ("mlp_2layer_nobias", {"hidden": 256}),
        ("mlp_deep_nobias", {"hidden": 128, "depth": 4}),
    ]:
        m = make_model(arch, input_dim=20, output_dim=5, **kw)
        print(f"  {arch:25s}: {count_parameters(m):>8d} params")

    cnn = make_model("simple_cnn", input_dim=0, output_dim=10)
    print(f"  {'simple_cnn':25s}: {count_parameters(cnn):>8d} params")

    resnet = make_model("resnet_tiny", input_dim=0, output_dim=10, in_channels=1)
    print(f"  {'resnet_tiny':25s}: {count_parameters(resnet):>8d} params")

    print("\n" + "=" * 60)
    print("Smoke test PASSED.")
    print("=" * 60)
