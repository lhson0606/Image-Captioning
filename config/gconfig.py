import torch

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")