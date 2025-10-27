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
from torchvision import datasets, transforms

# OpenFL Interactive API (libraries assumed present in container/venv)
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface

__version__ = '1.0.9'

DISPLAY_TITLE = r"""
         ChRIS Plugin: OpenFL Interactive API - MNIST
"""

# ---------------------------
# Simple CNN Model
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------
# OpenFL Interactive API Implementation (Local Simulation)
# ---------------------------
class MNISTDataset(DataInterface):
    """OpenFL DataInterface for MNIST (used for local sharding only)."""

    def __init__(self, collaborator_name, num_collaborators, **kwargs):
        super().__init__(**kwargs)
        self.collaborator_name = collaborator_name
        self.num_collaborators = num_collaborators

    def get_train_loader(self, batch_size=32):
        """Return training data loader for this collaborator"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        # Split data among collaborators
        total_size = len(dataset)
        shard_size = total_size // self.num_collaborators
        collab_idx = int(self.collaborator_name.replace('site', '')) - 1

        start_idx = collab_idx * shard_size
        end_idx = start_idx + shard_size if collab_idx < self.num_collaborators - 1 else total_size

        indices = list(range(start_idx, end_idx))
        sampler = SubsetRandomSampler(indices)

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def get_valid_loader(self, batch_size=32):
        """Return validation data loader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_train_data_size(self):
        """Return number of training samples"""
        total_size = 60000
        shard_size = total_size // self.num_collaborators
        return shard_size

    def get_valid_data_size(self):
        """Return number of validation samples"""
        return 10000


class MNISTTaskRunner(TaskInterface):
    """OpenFL TaskInterface for training and validation (local use)."""

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model if model is not None else SimpleCNN()
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
                data, target = data.to(self.device), target.to(self.device)
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
    print(" Preparing MNIST dataset...")
    
    # Download dataset first
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    datasets.MNIST('./data', train=True, download=True, transform=transform)
    datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Local simulation only (no director/envoys needed)
    print("  Note: Using local simulation mode (no director node required)")
    print("   For distributed OpenFL, please set up director and envoy nodes\n")
    
    return run_pytorch_federated_with_openfl_style(num_clients, num_rounds, outputdir)

def run_pytorch_federated_with_openfl_style(num_clients, num_rounds, outputdir):
    """Federated learning using OpenFL's TaskInterface pattern"""
    print(" Setting up federated simulation...\n")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create data interfaces for each collaborator
    data_interfaces = []
    for i in range(num_clients):
        collab_name = f"site{i+1}"
        data_interface = MNISTDataset(
            collaborator_name=collab_name,
            num_collaborators=num_clients
        )
        data_interfaces.append(data_interface)
        print(f"   Created data interface for {collab_name}")
    
    # Global task runner (aggregator's model)
    global_task = MNISTTaskRunner()
    
    # Federated training rounds
    print("\n Starting federated learning...\n")
    for round_num in range(num_rounds):
        print(f"{'='*60}")
        print(f" Round {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")
        
        # Collect client models for aggregation
        client_state_dicts = []
        
        for client_id, data_interface in enumerate(data_interfaces):
            print(f"   Site {client_id + 1} - Training...")
            
            # Create local task runner with global weights
            local_task = MNISTTaskRunner()
            local_task.model.load_state_dict(global_task.model.state_dict())
            
            # Get data loader
            train_loader = data_interface.get_train_loader(batch_size=32)
            
            # Train locally
            total_loss = 0
            batches = 0
            local_task.model.train()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 20:  # Limit batches for demo
                    break
                
                data, target = data.to(local_task.device), target.to(local_task.device)
                local_task.optimizer.zero_grad()
                output = local_task.model(data)
                loss = local_task.criterion(output, target)
                loss.backward()
                local_task.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches if batches > 0 else 0
            print(f"     └─ Loss: {avg_loss:.4f}")
            
            # Collect model for aggregation
            client_state_dicts.append(local_task.model.state_dict())
        
        # Federated averaging
        print(f"   Aggregating models (FedAvg)...")
        global_dict = global_task.model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [sd[key].float() for sd in client_state_dicts]
            ).mean(0)
        global_task.model.load_state_dict(global_dict)
        
        # Validation
        print(f"   Validating global model...")
        val_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        val_metrics = global_task.validate(val_loader)
        print(f"     └─ Validation Accuracy: {val_metrics['val_accuracy']:.2f}%")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f" Final Evaluation")
    print(f"{'='*60}")
    
    global_task.model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1000)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(global_task.device)
            target = target.to(global_task.device)
            output = global_task.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    final_accuracy = 100. * correct / total
    print(f" Final Test Accuracy: {final_accuracy:.2f}%\n")
    
    # Save model
    torch.save(global_task.model.state_dict(), outputdir / "final_model.pt")
    
    return {
        "final_accuracy": final_accuracy,
        "framework": "OpenFL Interactive API (Local Simulation)"
    }

# (Removed separate PyTorch fallback; unified on local simulation path)

# ---------------------------
# CLI Parser
# ---------------------------
parser = ArgumentParser(
    description='Run federated learning on MNIST using OpenFL or PyTorch',
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
    title='OpenFL MNIST Federated Learning',
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