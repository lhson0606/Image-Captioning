import torch

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

    # Check if the device is set to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if the device is set to CPU
    if device.type == 'cpu':
        print("Device is set to CPU.")
    else:
        print("Device is not set to CPU.")