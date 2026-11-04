# Lab 2 - Computer Vision with PyTorch

This project follows the structure of [Lab 1](https://github.com/Mohamed-Aite/Lab1_DL)
and applies PyTorch to computer-vision models on the MNIST dataset. The work is
organized as a Colab/Kaggle-ready notebook and a reusable Python implementation.

## Dataset

The experiments use the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
`torchvision.datasets.MNIST` downloads it automatically at runtime, so no dataset
or checkpoint is stored in this repository.

## Project Structure

| File | Description |
|---|---|
| `Part1_MNIST_Deep_Learning/MNIST_GPU_Lab.ipynb` | Complete lab notebook with GPU setup, experiments, metrics and interpretation |
| `Part1_MNIST_Deep_Learning/mnist_gpu_lab.py` | Reusable implementation of CNN, ViT, transfer models and Faster R-CNN target adapter |

## Part 1: MNIST Classification and Detection

### FCNN baseline

A fully connected neural network is included as a baseline. It flattens each
28x28 image and uses two regularized dense hidden layers before the ten-class
output, making the benefit of convolutional spatial inductive bias easier to
interpret.

### CNN classifier

The CNN uses two padded 3x3 convolution layers, batch normalization, ReLU,
2x2 max pooling, dropout and fully connected layers. AdamW, label smoothing,
weight decay and CUDA are used when available. The notebook reports test loss,
accuracy, macro-F1 and elapsed training time.

### Faster R-CNN

MNIST is an image-classification dataset and contains no bounding-box
annotations. The lab therefore makes the adaptation explicit: it trains an
11-class Faster R-CNN on one synthetic full-image box per digit, then reports
the detector loss, label accuracy, macro-F1 and training time. These numbers
are only a protocol demonstration; a genuine object-detection study should use
meaningful digit boxes and report mAP/IoU.

### Transfer learning with VGG16 and AlexNet

The pretrained-model section resizes MNIST to 224x224, converts grayscale to
three channels, applies ImageNet normalization, replaces the ten-class output
head, and fine-tunes VGG16 and AlexNet. Set the torchvision `weights` argument
to the relevant `DEFAULT` weights in an internet-enabled Colab run for strict
ImageNet initialization.

## Part 2: Vision Transformer

The ViT is implemented from scratch with 7x7 image patches, a learnable class
token, positional embeddings, Transformer encoder blocks and a ten-class
classification head. Its loss, accuracy, macro-F1 and training time are
compared with the CNN and transfer-learning models.

## Running the lab

Open `Part1_MNIST_Deep_Learning/MNIST_GPU_Lab.ipynb` in Google Colab or Kaggle
and enable a GPU runtime. For a short command-line smoke run:

```bash
python "Part1_MNIST_Deep_Learning/mnist_gpu_lab.py" --epochs 2 --limit 12000
```

For final reported values, remove `--limit`, increase the number of epochs,
record the returned `loss`, `accuracy`, `macro_f1` and `seconds` values, and
repeat runs with the same seed/configuration.

## Interpretation and key takeaways

CNNs are usually the strongest accuracy/compute baseline on native 28x28 MNIST,
while the FCNN provides a useful non-convolutional reference.
ViT demonstrates patch-token modelling, but generally needs more tuning and
data to outperform a well-designed CNN on this small image size. VGG16 and
AlexNet expose transfer learning and larger-backbone costs; their 224x224
preprocessing is disproportionate for MNIST. Faster R-CNN is only the right
choice when the task has meaningful object boxes.

This lab provided practical experience with GPU device management, convolutional
architectures, detection targets, transfer learning, Transformer tokenization,
regularization, metric selection and critical comparison of models.

## References

- PyTorch classifier tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Torchvision Faster R-CNN: https://pytorch.org/vision/stable/models/faster_rcnn.html
- Dosovitskiy et al., *An Image is Worth 16x16 Words*: https://arxiv.org/abs/2010.11929
- ViT tutorial: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
