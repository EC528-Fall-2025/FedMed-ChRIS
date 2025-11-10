"""PneumoniaMNIST Federated Learning: Task definitions."""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO


class SimpleCNN(nn.Module):
    """Simple CNN for PneumoniaMNIST (grayscale, binary classification)."""
    
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
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


# Global dataset cache
_train_dataset = None
_val_dataset = None
_test_dataset = None


def _initialize_datasets():
    """Initialize MedMNIST datasets (cached globally)."""
    global _train_dataset, _val_dataset, _test_dataset
    
    if _train_dataset is None:
        data_flag = 'pneumoniamnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        _train_dataset = DataClass(split='train', download=True, transform=transform, size=28)
        _val_dataset = DataClass(split='val', download=True, transform=transform, size=28)
        _test_dataset = DataClass(split='test', download=True, transform=transform, size=28)
    
    return _train_dataset, _val_dataset, _test_dataset


def load_data(partition_id: int, num_partitions: int, batch_size: int = 64):
    """Load partitioned PneumoniaMNIST data.
    
    Args:
        partition_id: ID of the partition (0-indexed)
        num_partitions: Total number of partitions
        batch_size: Batch size for DataLoader
    
    Returns:
        trainloader, valloader: DataLoaders for training and validation
    """
    train_ds, val_ds, _ = _initialize_datasets()
    
    # Partition training data
    total = len(train_ds)
    shard_size = total // num_partitions
    start_idx = partition_id * shard_size
    end_idx = start_idx + shard_size if partition_id < num_partitions - 1 else total
    
    train_indices = list(range(start_idx, end_idx))
    train_subset = Subset(train_ds, train_indices)
    
    # Use full validation set for each client (or partition it if preferred)
    # For now, using full validation set
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader


def get_test_loader(batch_size: int = 64):
    """Get test data loader for final evaluation."""
    _, _, test_ds = _initialize_datasets()
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def train(net, trainloader, epochs, device, lr=1e-3, weight_decay=1e-4):
    """Train the model on the training set.
    
    Args:
        net: Neural network model
        trainloader: Training data loader
        epochs: Number of training epochs
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    
    Returns:
        avg_trainloss: Average training loss
    """
    net.to(device)
    
    # Compute class weights for imbalanced dataset
    train_dataset = trainloader.dataset
    if isinstance(train_dataset, Subset):
        # Extract labels from subset
        base_dataset = train_dataset.dataset
        indices = train_dataset.indices
        labels = torch.tensor([base_dataset.labels[i] for i in indices]).squeeze().long()
    else:
        labels = torch.tensor(train_dataset.labels).squeeze().long()
    
    class_counts = torch.bincount(labels, minlength=2).float()
    class_weights = class_counts.sum() / (class_counts + 1e-9)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    net.train()
    running_loss = 0.0
    total_batches = 0
    
    for _ in range(epochs):
        for data, target in trainloader:
            data = data.to(device)
            target = target.squeeze().long().to(device)
            
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
    
    avg_trainloss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set.
    
    Args:
        net: Neural network model
        testloader: Test/validation data loader
        device: Device to evaluate on
    
    Returns:
        loss: Average loss
        accuracy: Accuracy on test set
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    
    correct, loss = 0, 0.0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.squeeze().long().to(device)
            
            outputs = net(data)
            loss += criterion(outputs, target).item()
            
            pred = outputs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    
    return loss, accuracy


def get_weights(net):
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)