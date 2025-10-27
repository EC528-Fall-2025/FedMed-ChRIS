#!/usr/bin/env python
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin
import json

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

# For medical data
import medmnist
from medmnist import INFO

__version__ = '1.1.0'

DISPLAY_TITLE = r"""
         ChRIS Plugin: OpenFL Interactive API - PneumoniaMNIST
"""

# ---------------------------
# Simple CNN Model
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, 64, 1, 1] regardless of H,W
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------
# Local "OpenFL-style" Data Interface
# ---------------------------
class MedMNISTDataInterface:
    """Shards a given base_dataset among collaborators."""
    def __init__(self, collaborator_name, num_collaborators, base_dataset, batch_size=64):
        self.collaborator_name = collaborator_name
        self.num_collaborators = num_collaborators
        self.base_dataset = base_dataset
        self.batch_size = batch_size

    def get_train_loader(self, batch_size=None):
        ds = self.base_dataset
        total = len(ds)
        shard = total // self.num_collaborators
        idx = int(self.collaborator_name.replace('site', '')) - 1
        start = idx * shard
        end = start + shard if idx < self.num_collaborators - 1 else total
        sampler = SubsetRandomSampler(list(range(start, end)))
        return DataLoader(ds, batch_size=batch_size or self.batch_size, sampler=sampler)

    def get_train_data_size(self):
        return len(self.base_dataset) // self.num_collaborators


class TaskRunner:
    """Minimal task runner for train/validate loops."""
    def __init__(self, model=None, in_channels=1, num_classes=2, class_weights=None, lr=1e-3, weight_decay=1e-4):
        self.model = model if model is not None else SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Adam tends to converge faster here
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def validate(self, batch_generator):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in batch_generator:
                data = data.to(self.device)
                target = target.squeeze().long().to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        accuracy = 100.0 * correct / total if total else 0.0
        return {'val_loss': val_loss / max(len(batch_generator), 1), 'val_accuracy': accuracy}


def run_openfl_interactive(options: Namespace, outputdir: Path):
    """Run local FedAvg-style training with MedMNIST Pneumonia."""
    print(" Using OpenFL Interactive API (Local Simulation)\n")
    print(" Preparing PneumoniaMNIST dataset...")
    print("  Note: Using local simulation mode (no director node required)")
    print("   For distributed OpenFL, please set up director and envoy nodes\n")
    return run_pytorch_federated_with_openfl_style(options, outputdir)


def _build_medmnist_pneumonia(size=28, download=True):
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])  # e.g., medmnist.PneumoniaMNIST

    # Keep grayscale and scale to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = DataClass(split='train', download=download, transform=transform, size=size)
    val_ds   = DataClass(split='val',   download=download, transform=transform, size=size)
    test_ds  = DataClass(split='test',  download=download, transform=transform, size=size)

    in_channels = 1          # BW
    num_classes = len(info['label'])  # 2 for binary
    return train_ds, val_ds, test_ds, in_channels, num_classes


def run_pytorch_federated_with_openfl_style(options: Namespace, outputdir: Path):
    print(" Setting up federated simulation on PneumoniaMNIST (BW)...\n")

    train_dataset, val_dataset, test_dataset, in_ch, num_classes = _build_medmnist_pneumonia(
        size=28, download=True
    )

    # Compute class weights once (handle imbalance)
    labels_tensor = torch.tensor(train_dataset.labels).squeeze().long()
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    class_weights = class_counts.sum() / (class_counts + 1e-9)

    # Create per-site data interfaces
    data_interfaces = []
    for i in range(options.clients):
        collab_name = f"site{i+1}"
        data_interfaces.append(MedMNISTDataInterface(
            collaborator_name=collab_name,
            num_collaborators=options.clients,
            base_dataset=train_dataset,
            batch_size=options.batch_size
        ))
        print(f"   Created data interface for {collab_name}")

    # Global task runner (aggregator)
    global_task = TaskRunner(in_channels=in_ch, num_classes=num_classes, class_weights=class_weights)

    print("\n Starting federated learning...\n")
    for round_num in range(options.rounds):
        print(f"{'='*60}\n Round {round_num + 1}/{options.rounds}\n{'='*60}")

        client_state_dicts = []

        for client_id, data_interface in enumerate(data_interfaces):
            print(f"   Site {client_id + 1} - Training...")

            # Local task with broadcasted weights
            local_task = TaskRunner(in_channels=in_ch, num_classes=num_classes, class_weights=class_weights)
            local_task.model.load_state_dict(global_task.model.state_dict())

            # Train for full shard, possibly multiple epochs
            for epoch in range(options.local_epochs):
                train_loader = data_interface.get_train_loader(batch_size=options.batch_size)
                total_loss, batches = 0.0, 0
                local_task.model.train()
                for data, target in train_loader:
                    data = data.to(local_task.device)
                    target = target.squeeze().long().to(local_task.device)

                    local_task.optimizer.zero_grad()
                    output = local_task.model(data)
                    loss = local_task.criterion(output, target)
                    loss.backward()
                    local_task.optimizer.step()

                    total_loss += loss.item()
                    batches += 1
                avg_loss = total_loss / batches if batches > 0 else 0.0
                print(f"     └─ Local epoch {epoch+1}/{options.local_epochs} avg loss: {avg_loss:.4f}")

            client_state_dicts.append(local_task.model.state_dict())

        # FedAvg aggregation
        print("   Aggregating models (FedAvg)...")
        global_dict = global_task.model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([sd[key].float() for sd in client_state_dicts]).mean(0)
        global_task.model.load_state_dict(global_dict)

        # Validate on MedMNIST validation split each round
        print("   Validating global model...")
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        val_metrics = global_task.validate(val_loader)
        print(f"     └─ Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")

    # Final evaluation on test split
    print(f"\n{'='*60}\n Final Evaluation\n{'='*60}")
    global_task.model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(global_task.device)
            target = target.squeeze().long().to(global_task.device)
            output = global_task.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    final_accuracy = 100.0 * correct / total if total else 0.0
    print(f" Final Test Accuracy: {final_accuracy:.2f}%\n")

    torch.save(global_task.model.state_dict(), outputdir / "final_model.pt")
    return {"final_accuracy": final_accuracy, "framework": "OpenFL Interactive API (Local Simulation)"}


# ---------------------------
# CLI Parser
# ---------------------------
parser = ArgumentParser(
    description='Run federated learning on PneumoniaMNIST using PyTorch',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('-r', '--rounds', type=int, default=3, help='Number of federation rounds')
parser.add_argument('-c', '--clients', type=int, default=2, help='Number of collaborators')
parser.add_argument('--local-epochs', type=int, default=2, help='Local epochs per client per round')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size per client')
parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')

# ---------------------------
# ChRIS Plugin
# ---------------------------
@chris_plugin(
    parser=parser,
    title='OpenFL PneumoniaMNIST Federated Learning',
    category='AI/ML',
    min_memory_limit='1Gi',
    min_cpu_limit='2000m',
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    print(DISPLAY_TITLE)
    print(f" Configuration:")
    print(f"   ├─ Rounds: {options.rounds}")
    print(f"   ├─ Collaborators: {options.clients}")
    print(f"   ├─ Local epochs: {options.local_epochs}")
    print(f"   ├─ Batch size: {options.batch_size}")
    print(f"   └─ Output: {outputdir}\n")
    
    try:
        result = run_openfl_interactive(options, outputdir)
        
        # Save summary
        summary = {
            "version": __version__,
            "framework": result["framework"],
            "rounds": options.rounds,
            "collaborators": options.clients,
            "local_epochs": options.local_epochs,
            "batch_size": options.batch_size,
            "final_accuracy": result["final_accuracy"],
            "status": "completed"
        }
        
        (outputdir / "federation_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n Complete! Results in {outputdir}\n")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()