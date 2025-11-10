"""PneumoniaMNIST Federated Learning: Client application."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from flower_pneumonia.task import (
    SimpleCNN,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


class FlowerClient(NumPyClient):
    """Flower client for PneumoniaMNIST training."""
    
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Train model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary
        
        Returns:
            Updated parameters, number of samples, metrics dictionary
        """
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        """Evaluate model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary
        
        Returns:
            Loss, number of samples, metrics dictionary
        """
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Create and configure a Flower client.
    
    Args:
        context: Flower context containing node and run configuration
    
    Returns:
        FlowerClient instance
    """
    # Load model
    net = SimpleCNN(in_channels=1, num_classes=2)
    
    # Get partition configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load partitioned data
    batch_size = context.run_config.get("batch-size", 64)
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    
    # Get training configuration
    local_epochs = context.run_config["local-epochs"]
    
    print(f"Client {partition_id + 1}/{num_partitions}: "
          f"Training samples = {len(trainloader.dataset)}, "
          f"Validation samples = {len(valloader.dataset)}")
    
    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)