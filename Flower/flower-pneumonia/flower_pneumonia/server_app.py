"""PneumoniaMNIST Federated Learning: Server application."""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flower_pneumonia.task import SimpleCNN, get_weights, get_test_loader, test


def server_fn(context: Context):
    """Create and configure the Flower server.
    
    Args:
        context: Flower context containing run configuration
    
    Returns:
        ServerAppComponents with strategy and config
    """
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    min_clients = context.run_config.get("min-available-clients", 2)
    
    # Initialize model with PneumoniaMNIST architecture
    net = SimpleCNN(in_channels=1, num_classes=2)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    
    print(f"Starting federated learning:")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Min clients: {min_clients}")
    
    # Define strategy with optional server-side evaluation
    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = SimpleCNN(in_channels=1, num_classes=2)
        
        # Set model parameters
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        from collections import OrderedDict
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        
        # Evaluate on test set
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testloader = get_test_loader(batch_size=128)
        loss, accuracy = test(net, testloader, device)
        
        print(f"Round {server_round} - Server-side test accuracy: {accuracy:.4f}")
        
        return loss, {"accuracy": accuracy}
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_clients,
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,  # Enable server-side evaluation
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)