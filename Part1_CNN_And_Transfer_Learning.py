from mnist_gpu_lab import *

if __name__ == "__main__":
    train_loader, test_loader = loaders("./data", 128, 12000)
    transfer_train, transfer_test = transfer_loaders("./data", 32, 12000)
    results = {}
    for name, model in (("FCNN", FCNN()), ("CNN", CNN())):
        results[name] = train(model, train_loader, test_loader, 2, 1e-3)
    results["Faster R-CNN (synthetic boxes)"] = train_faster_rcnn(faster_rcnn_for_mnist(), detection_loader("./data", 8, 256), 1)
    for name in ("vgg16", "alexnet"):
        results[name] = train(transfer_model(name), transfer_train, transfer_test, 1, 1e-4)
    for name, metrics in results.items():
        print(name, metrics)
