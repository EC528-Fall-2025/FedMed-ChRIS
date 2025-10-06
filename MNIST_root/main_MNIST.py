'''
This is the main script for offline testing only. It does not run with the plugin or in the container.
Use this script only for testing the actual MNIST classifier source code. Otherwise, this file is deprecated.
'''

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image

from .config import TrainConfig, EvalConfig
from .data import mnist_loaders
from .engine import train_model
from .models import SimpleCNN
from .utils import get_device, load_checkpoint, MNIST_MEAN_STD, _eval, preprocess_image


# I defined a new parser in the app.py ChRIS plugin entry-point. I left this here for reference, but I also made
# some changes to the new one in app.py. This is deprecated
def parse_args():
    p = argparse.ArgumentParser(description="MNIST classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    # training args
    pt = sub.add_parser("train", help="Train a model")
    pt.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    pt.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    pt.add_argument("--lr", type=float, default=TrainConfig.lr)
    pt.add_argument("--seed", type=int, default=TrainConfig.seed)
    pt.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    pt.add_argument("--out-dir", type=str, default=TrainConfig.out_dir)
    pt.add_argument("--device", type=str, default=TrainConfig.device, choices=["auto","cpu","cuda","mps"])
    pt.add_argument("--amp", action="store_true", help="Enable mixed precision (default off)", default=TrainConfig.amp)
    pt.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)

    # evaluation args
    pe = sub.add_parser("eval", help="Evaluate trained model")
    pe.add_argument("--weights", type=str, required=True)
    pe.add_argument("--batch-size", type=int, default=EvalConfig.batch_size)
    pe.add_argument("--num-workers", type=int, default=EvalConfig.num_workers)
    pe.add_argument("--device", type=str, default=EvalConfig.device, choices=["auto","cpu","cuda","mps"])

    # predict args
    pp = sub.add_parser("predict", help="Predict a single image of PNG or JPG)")
    pp.add_argument("--weights", type=str, required=True)
    pp.add_argument("--image", type=str, required=True)
    pp.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])

    return p.parse_args()

# This function isn't called in the ChRIS plugin; however, I kept it here for isolated testing of Training
def main_MNIST():
    '''
    This function isn't called in the ChRIS plugin; however, I kept it here for isolated testing of Training.
    Look in app.py for most recent ChRIS plugin code.
    '''
    args = parse_args()

    if args.cmd == "train":
        # Build config and loaders from the CL arguments (specified by ChRIS documentation)
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            num_workers=args.num_workers,
            out_dir=args.out_dir,
            device=args.device,
            amp=args.amp,
            weight_decay=args.weight_decay,
        )
        train_loader, test_loader = mnist_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        history, best_path = train_model(cfg, (train_loader, test_loader), SimpleCNN)
        print(f"Training finished. Best checkpoint: {best_path}")
        print(f"Last Epoch: loss={history['train_loss'][-1]:.4f}, "
              f"val_loss={history['validation_loss'][-1]:.4f}, "
              f"val_acc={history['validation_accuracy'][-1]:.2f}%")

    elif args.cmd == "eval":
        device = get_device(args.device)
        model = SimpleCNN().to(device)
        load_checkpoint(model, args.weights, map_location=device)
        _, test_loader = mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
        loss, acc = _eval(model, test_loader, device)
        print(f"Eval loss: {loss:.4f}, acc: {acc:.2f}%")

    elif args.cmd == "predict":
        device = get_device(args.device)
        model = SimpleCNN().to(device)
        load_checkpoint(model, args.weights, map_location=device)
        x = preprocess_image(args.image).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item()) * 100.0
        print(f"Prediction digit: {pred} (confidence: {conf:.2f}%)")

if __name__ == "__main__":
    main_MNIST()
