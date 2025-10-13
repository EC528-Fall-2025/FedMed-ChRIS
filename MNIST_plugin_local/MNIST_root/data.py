from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Standard mnist loader, found code on tutorial web page.
def mnist_loaders(batch_size: int, num_workers: int = 2, root: Path = Path("data")):
    tfm = transforms.Compose([
        transforms.ToTensor(),                              # [0,1] normaized here
        transforms.Normalize((0.1307,), (0.3081,))          # mean/std deviation for MNIST
    ])
    root = Path(root)
    train_ds = datasets.MNIST(root=str(root), train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
