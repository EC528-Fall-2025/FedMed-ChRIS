#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from chris_plugin import chris_plugin, PathMapper

# Imports for MNIST classifier code package. See source code in /MNIST_root
# Custom imports
from MNIST_root.config import TrainConfig, EvalConfig
from MNIST_root.data import mnist_loaders
from MNIST_root.engine import train_model
from MNIST_root.models import SimpleCNN
from MNIST_root.utils import get_device, load_checkpoint, MNIST_MEAN_STD, resolve_path, preprocess_image, _eval

from torchvision import transforms
from PIL import Image
import torch
import os

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
ChRIS Plugin: MNIST Classifier for Federated Learning Validation
"""


parser = ArgumentParser(description=("Train/Evaluate/Predict an MNIST classifier as a ChRIS plugin.\n"
                                    "Usage pattern follows ChRIS spec:\n"
                                    "  chrNIST [--mode] [--options...] inputdir/ outputdir/\n"
                                    "Where `inputdir` may contain images (for --mode predict) and\n"
                                    "`outputdir` will receive checkpoints and result files."),
                        formatter_class=ArgumentDefaultsHelpFormatter)

# Top level argument to specify mode, hyperparameter input arguments depend on this.
parser.add_argument("--mode",
                    choices=["train", "eval", "predict"],
                    required=True,
                    help="Which operation to run. The choices are 'train', 'eval', or 'predict ")

# Training options (specify hyperparameters)
parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Number of training epochs.")
parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="Batch size for training.")
parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="Learning rate.")
parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="Random seed.")
parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers, help="Number of data loader workers.")
parser.add_argument("--device", type=str, default=TrainConfig.device, choices=["auto", "cpu", "cuda", "mps"], help="Device to use: 'auto', 'cpu', 'cuda' (NVIDIA GPU), or 'mps' (Apple Silicon).")
parser.add_argument("--amp", action="store_true", default=TrainConfig.amp, help="Enable mixed precision if supported.")
parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay, help="Weight decay (L2 regularization).")

# eval and predict options
parser.add_argument("--weights", type=str, help="Path to checkpoint (.ckpt). If relative, resolved path against inputdir.")
parser.add_argument("--image", type=str, help="Image file (PNG/JPG). If relative, resolved path against inputdir.")

parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')

def print_dir_tree(root: Path, label: str) -> None:
    """
    Print a full recursive listing of `root`.
    Shows directories with trailing '/' and files with byte size.
    """
    print(f"\n[{label}] root={root}")
    if not root.exists():
        print("  (does not exist)")
        return
    # print the root itself
    print("  [D] .")
    try:
        for p in sorted(root.rglob("*"), key=lambda x: str(x).lower()):
            rel = p.relative_to(root)
            if p.is_dir():
                print(f"  [D] {rel}/")
            else:
                try:
                    size = p.stat().st_size
                except Exception as e:
                    size = f"stat-error:{e}"
                print(f"  [F] {rel}  ({size} bytes)")
    except Exception as e:
        print(f"  [error] failed walking {root}: {e}")

def prep_dataset_root(outputdir: Path) -> None:
    """
    Ensure torchvision's `root="data"` resolves to a writable path.
    We move CWD to outputdir and pre-create ./data under it.
    """
    try:
        (outputdir / "data").mkdir(parents=True, exist_ok=True)
        os.chdir(outputdir)
        print(f"[setup] CWD set to {Path.cwd()} ; dataset root will be ./data")
    except Exception as e:
        print(f"[setup][warn] could not switch CWD to outputdir: {e}")

# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='MNIST Classifier ChRIS Plugin',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='1Gi',      # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
# Single entrypoint to this plugin. See README.md for usage.
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)
    print_dir_tree(inputdir, "inputdir")
    print_dir_tree(outputdir, "outputdir")

    mode = options.mode
    device = get_device(options.device)

    # TRAINING MODE 
    if mode == "train":
        # Assign the CL arguments to config object (call constructor)
        cfg = TrainConfig(
            epochs=options.epochs,
            batch_size=options.batch_size,
            lr=options.lr,
            seed=options.seed,
            num_workers=options.num_workers,
            out_dir=str(outputdir),
            device=options.device,
            amp=options.amp,
            weight_decay=options.weight_decay,
        )
        prep_dataset_root(outputdir)
        train_loader, test_loader = mnist_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        history, best_path = train_model(cfg, (train_loader, test_loader), SimpleCNN)

        # create a .txt summary file in outputdir (note to self: maybe add this as debug mode only)
        (outputdir / "train_metrics.txt").write_text(
            f"best_ckpt={best_path}\n"
            f"last_train_loss={history['train_loss'][-1]:.4f}\n"
            f"last_val_loss={history['validation_loss'][-1]:.4f}\n"
            f"last_val_acc={history['validation_accuracy'][-1]:.2f}\n"
            f"learning rate={cfg.lr:.5f}\nbatch_size={cfg.batch_size}\nepochs={cfg.epochs}\n"
        )
        print(f"[train] Wrote checkpoint and metrics to: {outputdir}")

    # EVALUATION MODE
    elif mode == "eval":
        weights = resolve_path(options.weights, inputdir)
        if weights is None or not weights.exists():
            raise FileNotFoundError("Please provide --weights (file must exist). "
                                    "Relative paths are resolved against inputdir.")
        model = SimpleCNN().to(device)
        load_checkpoint(model, str(weights), map_location=device)
        prep_dataset_root(outputdir)
        _, test_loader = mnist_loaders(batch_size=options.batch_size, num_workers=options.num_workers)

        # Model evaluation call from utils.py
        loss, acc = _eval(model, test_loader, device)

        (outputdir / "eval.txt").write_text(f"loss={loss:.4f}\nacc={acc:.2f}\n")
        print(f"[eval] Wrote eval metrics to: {outputdir}")

    # PREDICTING MODE (PNG/JPG)
    elif mode == "predict":
        weights = resolve_path(options.weights, inputdir)
        image = resolve_path(options.image, inputdir)
        if weights is None or not weights.exists():
            raise FileNotFoundError("Please provide --weights (file must exist).")      # different from outputdir from template
        if image is None or not image.exists():
            raise FileNotFoundError("Please provide --image (file must exist).")        # Not properly implemented yet

        model = SimpleCNN().to(device)
        load_checkpoint(model, str(weights), map_location=device)
        x = preprocess_image(image).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item()) * 100.0

        (outputdir / "prediction.txt").write_text(f"digit={pred}\nconfidence={conf:.2f}\n")
        print(f"[predict] Wrote prediction to: {outputdir}")

if __name__ == '__main__':
    main()
