from mnist_gpu_lab import *

if __name__ == "__main__":
    train_loader, test_loader = loaders("./data", 128, 12000)
    metrics = train(ViT(), train_loader, test_loader, 2, 1e-3)
    print("ViT", metrics)
