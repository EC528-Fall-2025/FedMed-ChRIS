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
import csv

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
parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
parser.add_argument("--lr", type=float, default=TrainConfig.lr)
parser.add_argument("--seed", type=int, default=TrainConfig.seed)
parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
parser.add_argument("--device", type=str, default=TrainConfig.device, choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--amp", action="store_true", default=TrainConfig.amp, help="Enable mixed precision if supported.")
parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)

# eval and predict options
parser.add_argument("--weights", type=str, help="Path to checkpoint (.ckpt). If relative, resolved path against inputdir.")
parser.add_argument("--image", type=str, help="Image file (PNG/JPG). If relative, resolved path against inputdir.") # Predicting on a signle image
parser.add_argument("--pattern", type=str, default="**/*.png",
                    help="Glob pattern (relative to inputdir) for batch prediction, e.g. '**/*.png'. "
                         "Used only when --image is NOT given.")            # This is for predicting on a batch or folder of images
parser.add_argument("--suffix", type=str, default=".pred.txt",
                    help="Suffix for per-file outputs in batch mode, e.g., '.pred.txt' or '.json'.")

parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='MNIST Classifier ChRIS Plugin',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='2Gi',      # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=1              # set min_gpu_limit=1 to enable GPU
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
        _, test_loader = mnist_loaders(batch_size=options.batch_size, num_workers=options.num_workers)

        # Model evaluation call from utils.py
        loss, acc = _eval(model, test_loader, device)

        (outputdir / "eval.txt").write_text(f"loss={loss:.4f}\nacc={acc:.2f}\n")
        print(f"[eval] Wrote eval metrics to: {outputdir}")

    # PREDICTING MODE (PNG/JPG)
    elif mode == "predict":
        # Resolve weights (absolute path instead of using relative to inputfir)
        weights = resolve_path(options.weights, inputdir)
        if weights is None or not weights.exists():
            raise FileNotFoundError("Please provide --weights (file must exist). "
                                    "Relative paths are resolved against inputdir.")

        # Load model
        model = SimpleCNN().to(device)
        load_checkpoint(model, str(weights), map_location=device)
        model.eval()

        # SINGLE-IMAGE MODE 
        if options.image:
            image_path = resolve_path(options.image, inputdir)
            print(f"Predict on single image mode. Looking for image at: {image_path}")
            if image_path is None or not image_path.exists():
                raise FileNotFoundError("Please provide --image (file must exist). "
                                        "Relative paths are resolved against inputdir.")
            x = preprocess_image(image_path).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item()) * 100.0

            (outputdir / "prediction.txt").write_text(f"file={image_path.name}\ndigit={pred}\nconfidence={conf:.2f}\ninput path={image_path}")
            print(f"[predict] Single image done. See: {outputdir/'prediction.txt'}")
            return

        # DIRECTORY (BATCH) PREDICTION MODE 
        # PathMapper to mirror input->output paths and write per file results
        # Example: input: /in/d1/a.png  => output: /out/d1/a.png.pred.txt
        mapper = PathMapper.file_mapper(
            input_dir=inputdir,            # source dir
            output_dir=outputdir,           # dest dir
            glob=options.pattern,
            suffix=options.suffix
        )

        # Collect vals for a summary CSV
        rows = []
        num_files = 0

        for input_file, output_file in mapper:
            # Skip non-files
            if not input_file.is_file():
                continue
            num_files += 1

            # Predict on image
            x = preprocess_image(input_file).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item()) * 100.0

            # Write per-file output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(f"Predicted digit={pred}\nconfidence={conf:.2f}\n")

            # Appending for the summary
            rows.append((str(input_file.relative_to(inputdir)), pred, conf))

        if num_files == 0:
            raise FileNotFoundError(
                f"No images matched pattern '{options.pattern}' under {inputdir}. "
                "Adjust --pattern (e.g. '**/*.png' or '**/*.[pj][pn]g')."
            )

        # Write the summary CSV 
        summary_csv = outputdir / "results.csv"
        with summary_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["relative_path", "predicted_digit", "confidence_percent"])
            w.writerows(rows)

        print(f"[predict] Batch done: {num_files} files")
        print(f"[predict] Per-file outputs under: {outputdir} (suffix='{options.suffix}')")
        print(f"[predict] Summary CSV: {summary_csv}")

if __name__ == '__main__':
    main()
