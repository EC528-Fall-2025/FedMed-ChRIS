#!/usr/bin/env python
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin
import json
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# OpenFL Interactive API
try:
    from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, FLExperiment
    from openfl.interface.interactive_api.federation import Federation
    from openfl.component.collaborator import Collaborator
    OPENFL_AVAILABLE = True
except ImportError:
    OPENFL_AVAILABLE = False
    #print("  OpenFL Interactive API not available, will use PyTorch fallback")

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
if OPENFL_AVAILABLE:
    class MNISTDataset(DataInterface):
        """OpenFL DataInterface for MNIST"""
        
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
        """OpenFL TaskInterface for training and validation"""
        
        def __init__(self, model=None, **kwargs):
            super().__init__(**kwargs)
            self.model = model if model is not None else SimpleCNN()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            self.criterion = nn.CrossEntropyLoss()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        
        def train_epoch(self, batch_generator):
            """Train for one epoch"""
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
            
            return {'train_loss': np.mean(losses)}
        
        def validate(self, batch_generator):
            """Validate the model"""
            self.model.eval()
            val_loss = 0
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
            
            accuracy = 100. * correct / total
            return {'val_loss': val_loss / len(batch_generator), 'val_accuracy': accuracy}

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
    
    # For local simulation, we'll use the PyTorch fallback approach
    # since Federation() requires a director node connection
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

# ---------------------------
# PyTorch Fallback
# ---------------------------
def run_pytorch_federated(num_clients, num_rounds, outputdir):
    """Fallback: Pure PyTorch federated learning"""
    print(" Using PyTorch implementation\n")
    print(" Downloading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Split data
    client_data_size = len(train_dataset) // num_clients
    client_datasets = torch.utils.data.random_split(
        train_dataset, 
        [client_data_size] * num_clients
    )
    
    # Initialize model
    global_model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    
    # Training rounds
    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f" Round {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")
        
        client_models = []
        
        for client_id in range(num_clients):
            print(f"   Site {client_id + 1} - Training...")
            
            client_model = SimpleCNN()
            client_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
            
            loader = DataLoader(client_datasets[client_id], batch_size=32, shuffle=True)
            client_model.train()
            
            total_loss = 0
            batches = 0
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= 20:
                    break
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
            
            print(f"     └─ Loss: {total_loss/batches:.4f}")
            client_models.append(client_model)
        
        # Aggregate
        print(f"   Aggregating...")
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [m.state_dict()[key].float() for m in client_models]
            ).mean(0)
        global_model.load_state_dict(global_dict)
    
    # Final evaluation
    print(f"\n Final evaluation...")
    global_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1000)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    final_accuracy = 100. * correct / total
    print(f"✓ Accuracy: {final_accuracy:.2f}%")
    
    torch.save(global_model.state_dict(), outputdir / "final_model.pt")
    
    return {
        "final_accuracy": final_accuracy,
        "framework": "PyTorch"
    }

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
        if OPENFL_AVAILABLE:
            result = run_openfl_interactive(options.clients, options.rounds, outputdir)
        else:
            result = run_pytorch_federated(options.clients, options.rounds, outputdir)
        
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