from tensorflow.keras import applications, optimizers, losses


def preprocess_input(architecture):
    """Preprocesses a tensor or Numpy array encoding a batch of images.
    """
    dicts = {
        'densenet': applications.densenet,
        'efficientnet': applications.efficientnet,
#        'imagenet_utils': applications.imagenet_utils,
        'inception_resnet_v2': applications.inception_resnet_v2,
        'inception_v3': applications.inception_v3,
        'mobilenet': applications.mobilenet,
        'mobilenet_v2': applications.mobilenet_v2,
        'nasnet': applications.nasnet,
        'resnet': applications.resnet,
        'resnet50': applications.resnet50,
        'resnet_v2': applications.resnet_v2,
        'vgg16': applications.vgg16,
        'vgg19': applications.vgg19,
        'xception': applications.xception,
    }
    return dicts[architecture].preprocess_input


def instantiates_architecture(architecture, kwargs):
    dicts = {
        'DenseNet121': applications.DenseNet121,
        'DenseNet169': applications.DenseNet169,
        'DenseNet201': applications.DenseNet201,
        'EfficientNetB0': applications.EfficientNetB0,
        'EfficientNetB1': applications.EfficientNetB1,
        'EfficientNetB2': applications.EfficientNetB2,
        'EfficientNetB3': applications.EfficientNetB3,
        'EfficientNetB4': applications.EfficientNetB4,
        'EfficientNetB5': applications.EfficientNetB5,
        'EfficientNetB6': applications.EfficientNetB6,
        'EfficientNetB7': applications.EfficientNetB7,
        'InceptionResNetV2': applications.InceptionResNetV2,
        'InceptionV3': applications.InceptionV3,
        'MobileNet': applications.MobileNet,
        'MobileNetV2': applications.MobileNetV2,
        'NASNetLarge': applications.NASNetLarge,
        'NASNetMobile': applications.NASNetMobile,
        'ResNet101': applications.ResNet101,
        'ResNet101V2': applications.ResNet101V2,
        'ResNet152': applications.ResNet152,
        'ResNet152V2': applications.ResNet152V2,
        'ResNet50': applications.ResNet50,
        'ResNet50V2': applications.ResNet50V2,
        'VGG16': applications.VGG16,
        'VGG19': applications.VGG19,
        'Xception': applications.Xception,
    }
    return dicts[architecture](**kwargs)


def implements_optimizer(optimizer, kwargs):
    dicts = {
        'Adadelta': optimizers.Adadelta,
        'Adagrad': optimizers.Adagrad,
        'Adam': optimizers.Adam,
        'Adamax': optimizers.Adamax,
        'Ftrl': optimizers.Ftrl,
        'Nadam': optimizers.Nadam,
#        'Optimizer': optimizers.Optimizer,
        'RMSprop': optimizers.RMSprop,
        'SGD': optimizers.SGD,
    }
    return dicts[optimizer](**kwargs)


def computes_loss(loss, kwargs):
    dicts = {
        'BinaryCrossentropy': losses.BinaryCrossentropy,
        'CategoricalCrossentropy': losses.CategoricalCrossentropy,
        'CategoricalHinge': losses.CategoricalHinge,
        'CosineSimilarity': losses.CosineSimilarity,
        'Hinge': losses.Hinge,
        'Huber': losses.Huber,
        'KLDivergence': losses.KLDivergence,
        'LogCosh': losses.LogCosh,
#        'Loss': losses.Loss,
        'MeanAbsoluteError': losses.MeanAbsoluteError,
        'MeanAbsolutePercentageError': losses.MeanAbsolutePercentageError,
        'MeanSquaredError': losses.MeanSquaredError,
        'MeanSquaredLogarithmicError': losses.MeanSquaredLogarithmicError,
        'Poisson': losses.Poisson,
        'Reduction': losses.Reduction,
        'SparseCategoricalCrossentropy': losses.SparseCategoricalCrossentropy,
        'SquaredHinge': losses.SquaredHinge,
    }
    return dicts[loss](**kwargs)
