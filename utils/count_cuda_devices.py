import torch

if __name__ == "__main__":
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)