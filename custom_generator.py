import tensorflow as tf

import io
import numpy as np
import os
import warnings

from PIL import Image as pil_image

from keras_preprocessing.image import array_to_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DirectoryIterator, DataFrameIterator

_INTERPOLATION_METHODS = [
    'area',
    'bicubic',
    'bilinear',
    'gaussian',
    'lanczos3',
    'lanczos5',
    'mitchellcubic',
    'nearest',
]


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest', data_format='channels_last',
             dtype='float32'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported.
            Default: "nearest".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    with open(path, 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
        if color_mode == 'grayscale':
            # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
            # convert it to an 8-bit grayscale image.
            if img.mode not in ('L', 'I;16', 'I'):
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_INTERPOLATION_METHODS)))
                np_img = img_to_array(img, dtype=dtype, data_format=data_format)
                resize_tensor = tf.image.resize(
                    np_img, width_height_tuple, interpolation)
                img = array_to_img(
                    resize_tensor, dtype=dtype, data_format=data_format)
        return img


class TFResizeImageDataGenerator(ImageDataGenerator):
    
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 interpolation_order=1,
                 dtype='float32'):
        super().__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            interpolation_order=interpolation_order,
            dtype=dtype)
    
    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return TFResizeDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dtype=self.dtype
        )
    
    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            x_col="filename",
                            y_col="class",
                            weight_col=None,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            validate_filenames=True,
                            **kwargs):
        if 'has_ext' in kwargs:
            warnings.warn('has_ext is deprecated, filenames in the dataframe have '
                          'to match the exact filenames in disk.',
                          DeprecationWarning)
        if 'sort' in kwargs:
            warnings.warn('sort is deprecated, batches will be created in the'
                          'same order than the filenames provided if shuffle'
                          'is set to False.', DeprecationWarning)
        if class_mode == 'other':
            warnings.warn('`class_mode` "other" is deprecated, please use '
                          '`class_mode` "raw".', DeprecationWarning)
            class_mode = 'raw'
        if 'drop_duplicates' in kwargs:
            warnings.warn('drop_duplicates is deprecated, you can drop duplicates '
                          'by using the pandas.DataFrame.drop_duplicates method.',
                          DeprecationWarning)
        
        return DataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            dtype=self.dtype
        )


class TFResizeDirectoryIterator(DirectoryIterator):
    
    def _get_batches_of_transformed_samples(self, index_array):
        return _get_batches_of_transformed_samples(self, index_array)


class TFResizeDataFrameIterator(DataFrameIterator):
    
    def _get_batches_of_transformed_samples(self, index_array):
        return _get_batches_of_transformed_samples(self, index_array)


def _get_batches_of_transformed_samples(Iterator, index_array):
    """Gets a batch of transformed samples.
    # Arguments
        index_array: Array of sample indices to include in batch.
    # Returns
        A batch of transformed samples.
    """
    batch_x = np.zeros((len(index_array),) + Iterator.image_shape, dtype=Iterator.dtype)
    # build batch of image data
    # Iterator.filepaths is dynamic, is better to call it once outside the loop
    filepaths = Iterator.filepaths
    for i, j in enumerate(index_array):
        img = load_img(filepaths[j],
                       color_mode=Iterator.color_mode,
                       target_size=Iterator.target_size,
                       interpolation=Iterator.interpolation)
        x = img_to_array(img, data_format=Iterator.data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if Iterator.image_data_generator:
            params = Iterator.image_data_generator.get_random_transform(x.shape)
            x = Iterator.image_data_generator.apply_transform(x, params)
            x = Iterator.image_data_generator.standardize(x)
        batch_x[i] = x
    # optionally save augmented images to disk for debugging purposes
    if Iterator.save_to_dir:
        for i, j in enumerate(index_array):
            img = array_to_img(batch_x[i], Iterator.data_format, scale=True)
            fname = '{prefix}_{index}_{hash}.{format}'.format(
                prefix=Iterator.save_prefix,
                index=j,
                hash=np.random.randint(1e7),
                format=Iterator.save_format)
            img.save(os.path.join(Iterator.save_to_dir, fname))
    # build batch of labels
    if Iterator.class_mode == 'input':
        batch_y = batch_x.copy()
    elif Iterator.class_mode in {'binary', 'sparse'}:
        batch_y = np.empty(len(batch_x), dtype=Iterator.dtype)
        for i, n_observation in enumerate(index_array):
            batch_y[i] = Iterator.classes[n_observation]
    elif Iterator.class_mode == 'categorical':
        batch_y = np.zeros((len(batch_x), len(Iterator.class_indices)),
                           dtype=Iterator.dtype)
        for i, n_observation in enumerate(index_array):
            batch_y[i, Iterator.classes[n_observation]] = 1.
    elif Iterator.class_mode == 'multi_output':
        batch_y = [output[index_array] for output in Iterator.labels]
    elif Iterator.class_mode == 'raw':
        batch_y = Iterator.labels[index_array]
    else:
        return batch_x
    if Iterator.sample_weight is None:
        return batch_x, batch_y
    else:
        return batch_x, batch_y, Iterator.sample_weight[index_array]
