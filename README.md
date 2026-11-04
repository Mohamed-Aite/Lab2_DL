# Lab 2 - Computer Vision with PyTorch

This project was completed for the Deep Learning course at the Faculté des
Sciences et Techniques de Tanger. The objective was to use PyTorch and GPU
computing to build and compare different computer-vision architectures on the
MNIST handwritten-digit dataset.

The dataset is available on
[Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

## Project files

| File | Description |
|---|---|
| `Part1_MNIST_Deep_Learning/MNIST_GPU_Lab.ipynb` | Notebook containing the experiments and results |
| `Part1_MNIST_Deep_Learning/mnist_gpu_lab.py` | Python implementation of the models |

## Part 1: CNN, Faster R-CNN and transfer learning

### CNN classifier

I first built a CNN to classify the handwritten digits. The network uses two
convolution layers, batch normalization, ReLU activation, max-pooling layers,
dropout and fully connected layers. The convolution layers help the model
recognize local shapes such as edges and curves in the digits.

The model uses AdamW as optimizer, a small weight decay for regularization and
label smoothing to make the training more stable. When a GPU is available,
the data and the model are moved to CUDA to reduce the training time.

I also included a fully connected neural network as a simple baseline. Unlike
the CNN, it first flattens the image and does not use the spatial structure of
the pixels. This makes it useful for seeing why convolution is helpful for
image classification.

### Faster R-CNN

Faster R-CNN is normally used to find objects and draw bounding boxes around
them. The original MNIST dataset only gives one label for each image and does
not provide bounding boxes. To still study the architecture, I created a
demonstration using one synthetic full-image box for each digit.

This comparison must be interpreted carefully. The Faster R-CNN result is not
the same as a real object-detection experiment because the boxes are synthetic.
A proper detection study would need images containing several digits or real
annotations for the position of each digit. The detector section reports its
loss, digit-label accuracy, macro-F1 and training time, while a real detection
task should also use metrics such as IoU and mAP.

### VGG16 and AlexNet

I then fine-tuned the pretrained VGG16 and AlexNet networks. Since these
models were originally trained on ImageNet, the grayscale MNIST images are
converted to three channels and resized to 224x224 pixels. Their final
classification layers are replaced with new layers that predict the ten MNIST
digits.

These models show how transfer learning works, but they are much larger than
necessary for MNIST. The resizing and extra parameters also make them more
expensive than the small CNN.

### Comparison

The models are compared using test loss, accuracy, macro-F1 score and training
time. The CNN is expected to provide the best balance between accuracy and
computation for native 28x28 MNIST images. The fully connected model is a
useful baseline, but it does not use image structure as effectively. VGG16 and
AlexNet can benefit from pretrained features, although their size is not
really justified for this small dataset. Faster R-CNN is not directly
comparable unless meaningful detection annotations are available.

## Part 2: Vision Transformer

For the second part, I implemented a Vision Transformer from scratch following
the provided tutorial and the original ViT paper.

The image is divided into 7x7 patches. Each patch is converted into an
embedding, and a class token and positional embeddings are added. Transformer
encoder blocks then learn relationships between the different patches. The
class token is finally used to predict one of the ten digits.

The ViT is compared with the models from Part 1 using the same loss, accuracy,
macro-F1 and training-time measurements. A ViT is powerful because it can
learn relationships between distant image regions, but MNIST images are very
small and simple. For this reason, a CNN will often be easier to train and
more efficient, while the ViT usually requires more tuning to show an
advantage.

## Conclusion and personal synthesis

This lab helped me understand the complete workflow of a computer-vision
project with PyTorch: preparing the data, defining neural-network layers,
choosing hyperparameters, using regularization, training on a GPU and
comparing models with several metrics.

The main lesson is that the most complex model is not automatically the best
one. The architecture must match the data and the objective. For MNIST
classification, a small CNN is usually the most suitable solution. VGG16 and
AlexNet demonstrate transfer learning, ViT demonstrates Transformer-based
vision, and Faster R-CNN shows why the annotation format must match the task.

## References

- MNIST dataset:
  https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- PyTorch classifier tutorial:
  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Torchvision Faster R-CNN:
  https://pytorch.org/vision/stable/models/faster_rcnn.html
- Dosovitskiy et al., *An Image is Worth 16x16 Words*:
  https://arxiv.org/abs/2010.11929
- Vision Transformer tutorial:
  https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
