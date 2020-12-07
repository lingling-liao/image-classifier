# image-classifier
This tutorial shows how to classify images of flowers.

## Official Docker images for the machine learning framework TensorFlow

Docker Pull Command: `docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter`

Running Containers: `docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter`

## Download flower dataset
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

## Training a model
Running the command: python train_image_classifier.py
