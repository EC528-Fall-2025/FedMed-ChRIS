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

__version__ = '1.0.9'

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
        self.gap = nn.AdaptiveAvgPool2d(1) # -> [B, 64, 1, 1] regardless of H,W
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
# OpenFL Interactive API Implementation (Local Simulation)
# ---------------------------
class MedMNISTDataInterface():
    """OpenFL-style DataInterface that shards a given base_dataset among collaborators."""
    def __init__(self, collaborator_name, num_collaborators, base_dataset, batch_size=32, **kwargs):
        super().__init__(**kwargs)
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

    def get_valid_loader(self, batch_size=32):
        # We keep validation outside via the main loop; not used here.
        return DataLoader(self.base_dataset, batch_size=batch_size, shuffle=False)

    def get_train_data_size(self):
        return len(self.base_dataset) // self.num_collaborators


class TaskRunner():
    """OpenFL TaskInterface for training and validation (local use)."""

    def __init__(self, model=None, in_channels=1, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.model = model if model is not None else SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, batch_generator):
        """Train for one epoch (not used in main loop, kept for completeness)."""
        self.model.train()
        losses = []

        for data, target in batch_generator:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        return {'train_loss': avg_loss}

    def validate(self, batch_generator):
        """Validate the model"""
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
                total += len(target)

        accuracy = 100.0 * correct / total if total else 0.0
        return {'val_loss': val_loss / max(len(batch_generator), 1), 'val_accuracy': accuracy}

def run_openfl_interactive(num_clients, num_rounds, outputdir):
    """Run federated learning using OpenFL Interactive API (local simulation)"""
    
    print(" Using OpenFL Interactive API (Local Simulation)\n")
    print(" Preparing PneumoniaMNIST dataset...")
    # Local simulation only (no director/envoys needed)
    print("  Note: Using local simulation mode (no director node required)")
    print("   For distributed OpenFL, please set up director and envoy nodes\n")
    
    return run_pytorch_federated_with_openfl_style(num_clients, num_rounds, outputdir)

def _build_medmnist_pneumonia(size=28, download=True):
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])  # e.g., medmnist.PneumoniaMNIST

    # Transform: keep grayscale and scale to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = DataClass(split='train', download=download, transform=transform, size=size)
    val_ds   = DataClass(split='val',   download=download, transform=transform, size=size)
    test_ds  = DataClass(split='test',  download=download, transform=transform, size=size)

    in_channels = 1          # BW
    num_classes = len(info['label'])  # should be 2 for binary
    return train_ds, val_ds, test_ds, in_channels, num_classes

def run_pytorch_federated_with_openfl_style(num_clients, num_rounds, outputdir):
    print(" Setting up federated simulation on PneumoniaMNIST (BW)...\n")

    train_dataset, val_dataset, test_dataset, in_ch, num_classes = _build_medmnist_pneumonia(
        size=28, download=True
    )

    # Create per-site data interfaces
    data_interfaces = []
    for i in range(num_clients):
        collab_name = f"site{i+1}"
        data_interfaces.append(MedMNISTDataInterface(
            collaborator_name=collab_name,
            num_collaborators=num_clients,
            base_dataset=train_dataset,
            batch_size=32
        ))
        print(f"   Created data interface for {collab_name}")

    # Global task runner (aggregator)
    global_task = TaskRunner(in_channels=in_ch, num_classes=num_classes)

    print("\n Starting federated learning...\n")
    for round_num in range(num_rounds):
        print(f"{'='*60}\n Round {round_num + 1}/{num_rounds}\n{'='*60}")

        client_state_dicts = []

        for client_id, data_interface in enumerate(data_interfaces):
            print(f"   Site {client_id + 1} - Training...")

            # Local task with broadcasted weights
            local_task = TaskRunner(in_channels=in_ch, num_classes=num_classes)
            local_task.model.load_state_dict(global_task.model.state_dict())

            train_loader = data_interface.get_train_loader(batch_size=32)

            total_loss, batches = 0.0, 0
            local_task.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 20:  # cap for speed, like before
                    break

                data = data.to(local_task.device)
                # MedMNIST returns labels with shape [B, 1]; CE expects [B], so squeeze.
                target = target.squeeze().long().to(local_task.device)

                local_task.optimizer.zero_grad()
                output = local_task.model(data)
                loss = local_task.criterion(output, target)
                loss.backward()
                local_task.optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / batches if batches > 0 else 0.0
            print(f"     └─ Loss: {avg_loss:.4f}")

            client_state_dicts.append(local_task.model.state_dict())

        # FedAvg aggregation
        print("   Aggregating models (FedAvg)...")
        global_dict = global_task.model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([sd[key].float() for sd in client_state_dicts]).mean(0)
        global_task.model.load_state_dict(global_dict)

        # Validate on MedMNIST test split
        print("   Validating global model...")
        val_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        val_metrics = global_task.validate(val_loader)
        print(f"     └─ Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")

    # Final evaluation
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


# (Removed separate PyTorch fallback; unified on local simulation path)

# ---------------------------
# CLI Parser
# ---------------------------
parser = ArgumentParser(
    description='Run federated learning on PneumoniaMNIST using PyTorch',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('-r', '--rounds', type=int, default=3, help='Number of federation rounds')
parser.add_argument('-c', '--clients', type=int, default=2, help='Number of collaborators')
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
    print(f"   └─ Output: {outputdir}\n")
    
    try:
        result = run_openfl_interactive(options.clients, options.rounds, outputdir)
        
        # Save summary
        summary = {
            "version": __version__,
            "framework": result["framework"],
            "rounds": options.rounds,
            "collaborators": options.clients,
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