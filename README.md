# Image Classifier
This tutorial shows how to classify images of flowers.

## Official Docker images for TensorFlow

Docker pull command:

```
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter
```

Running containers:

```
docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter
```

## Download flower dataset


## Training a model

Running the command:

```
python train_image_classifier.py
```
