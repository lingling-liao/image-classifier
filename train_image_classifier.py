import tensorflow as tf

import built_in
import custom_generator


class TrainKerasApplicationsModel:
    
    def __init__(self):
        # For self.set_up_data
        self.train_dir = './train_data'
        self.valid_dir = './valid_data'
        self.train_generator_kwargs = {}
        self.valid_generator_kwargs = {}
        self.train_data_kwargs = {}
        self.valid_data_kwargs = {}
        # For self.compose_model
        self.preprocess = 'xception'
        self.architecture = 'Xception'
        self.architecture_kwargs = {
            'input_shape': (256, 256, 3),
            'include_top': False,
        }
        self.num_classes = 5
        self.trainable = False
        self.fine_tune_at = 0
        self.dense_kwargs = {
            'activation': 'softmax',
        }
        self.training = False
        self.dropout_rate = .20
        # For self.compile_model
        self.optimizer = 'Adam'
        self.optimizer_kwargs = {
            'learning_rate': 0.001,
        }
        self.loss = 'CategoricalCrossentropy'
        self.loss_kwargs = {}
        self.metrics = [
            'accuracy',
        ]
        # For self.train_model
        self.log_dir = './logs'
        self.checkpoint_dir = './checkpoint'
        self.patience = None
        self.model_fit_kwargs = {
            'epochs': 5,
        }
        self.saved_model_path = './saved_model'
    
    def set_up_data(self, data_dir, generator_kwargs, data_kwargs):
        """Inputs are suitably resized for the selected module.
        
        Dataset augmentation (i.e., random distortions of an image each time it
        is read) improves training.
        """
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **generator_kwargs)
        return generator.flow_from_directory(data_dir, **data_kwargs)
    
    def compose_model(self):
        """Load in the pretrained base model (and pretrained weights).
        
        Stack the classification layers on top.
        """
        # Rescale pixel values
        preprocess_input = built_in.preprocess_input(self.preprocess)
        
        # Create the base model from the pre-trained convnets
        base_model = built_in.instantiates_architecture(
            self.architecture, self.architecture_kwargs)
        input_shape = base_model.input.shape[1:]
        
        # Transfer Learning:
        # If trainable is False, freeze the convolutional base.
        
        # Fine Tuning:
        # If trainable is True, un-freeze the top layers of the model.
        
        base_model.trainable = self.trainable
        if self.trainable:
            # Freeze all the layers before the 'fine_tune_at' layer
            for layer in base_model.layers[:self.fine_tune_at]:
                layer.trainable = False
        
        # Add a classification head
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(
            self.num_classes, **self.dense_kwargs)
        
        # Build a model by chaining together the data augmentation, rescaling,
        # base_model and feature extractor layers using the Keras Functional.
        inputs = tf.keras.Input(shape=input_shape)
        x = preprocess_input(inputs)
        x = base_model(x, training=self.training)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = prediction_layer(x)
        return tf.keras.Model(inputs, outputs)
    
    def compile_model(self, model):
        """Configures the model for training.
        """
        optimizer = built_in.implements_optimizer(
            self.optimizer, self.optimizer_kwargs)
        loss = built_in.computes_loss(self.loss, self.loss_kwargs)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=self.metrics)
        return model
    
    def train_model(self, model, train_data, valid_data):
        """Trains the model for a fixed number of epochs.
        """
        if train_data.class_indices != valid_data.class_indices:
            raise ValueError('valid_data.class_indices is different from '
                             'train_data.class_indices')
        
        # Enable visualizations for TensorBoard.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1, write_images=True)
        
        # To see the output, you can download and view the TensorBoard logs at
        # the terminal. $ tensorboard --logdir=./logs/ --host=0.0.0.0
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_dir,
            save_best_only=True,
            monitor='val_loss',
            mode='min')
        callbacks = [checkpoint, tensorboard_callback]
        
        # Stop training when a monitored metric has stopped improving.
        if self.patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', mode='min', patience=self.patience)
            callbacks = callbacks.append(early_stopping)
        
        steps_per_epoch = train_data.samples // train_data.batch_size
        validation_steps = valid_data.samples // valid_data.batch_size
        
        history = model.fit(train_data,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_data,
                            validation_steps=validation_steps,
                            callbacks=callbacks,
                            **self.model_fit_kwargs).history
        return model, history
    
    def save_model(self, model):
        """Exports the Trackable object obj to SavedModel format.
        """
        tf.saved_model.save(model, self.saved_model_path)
    
    def load_model(self):
        """Loads a model saved via model.save().
        """
        return tf.keras.models.load_model(self.saved_model_path)
    
    def evaluate(self, model, valid_data):
        """Returns the loss value & metrics values for the model in test mode.
        """
        return model.evaluate(valid_data)
    
    def predict(self, model, valid_data):
        """Generates output predictions for the input samples.
        """
        return model.predict(valid_data)
    

class TrainTFResizeModel(TrainKerasApplicationsModel):
    
    def set_up_data(self, data_dir, generator_kwargs, data_kwargs):
        generator = custom_generator.TFResizeImageDataGenerator(
            **generator_kwargs)
        return generator.flow_from_directory(data_dir, **data_kwargs)
    
    def compose_model(self, interpolation='nearest'):
        """Add image resizing layer.
        """
        # Rescale pixel values
        preprocess_input = built_in.preprocess_input(self.preprocess)
        
        # Create the base model from the pre-trained convnets
        base_model = built_in.instantiates_architecture(
            self.architecture, self.architecture_kwargs)
        _, height, width, channel = base_model.input.shape
        
        # Transfer Learning:
        # If trainable is False, freeze the convolutional base.
        
        # Fine Tuning:
        # If trainable is True, un-freeze the top layers of the model.
        
        base_model.trainable = self.trainable
        if self.trainable:
            # Freeze all the layers before the 'fine_tune_at' layer
            for layer in base_model.layers[:self.fine_tune_at]:
                layer.trainable = False
        
        # Resizing layer
        resize = tf.keras.layers.experimental.preprocessing.Resizing(
            height, width, interpolation=interpolation, name='resize')
        
        # Add a classification head
        flatten_layer = tf.keras.layers.Flatten()
        prediction_layer = tf.keras.layers.Dense(
            self.num_classes, **self.dense_kwargs)
        
        # Build a model by chaining together the data augmentation, rescaling,
        # base_model and feature extractor layers using the Keras Functional.
        inputs = tf.keras.layers.Input([None, None, channel])
        x = resize(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=self.training)
        x = flatten_layer(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = prediction_layer(x)
        return tf.keras.Model(inputs, outputs)
    
    def pretrained_model(self, model):
        resize = model.get_layer(name='resize')
        target_height, target_width = resize.target_height, resize.target_width
        
        self.num_classes = model.get_layer(index=-1).kernel.shape[1]
        
        functional_layer = model.get_layer(index=-4)
        functional_layer.trainable = self.trainable
        print('Layer(type): %s(Functional)' % functional_layer._name)
        
        for layer in functional_layer.layers[:self.fine_tune_at]:
            layer.trainable = False
        return model, target_height, target_width


if __name__ == '__main__':
    tkam = TrainKerasApplicationsModel()
    
    # Set up data
    tkam.train_dir = './flower_photos'
    tkam.train_generator_kwargs = dict(validation_split=.20)
    tkam.train_data_kwargs = dict(
        subset="training", shuffle=True)
    
    tkam.valid_dir = './flower_photos'
    tkam.valid_generator_kwargs = tkam.train_generator_kwargs
    tkam.valid_data_kwargs = dict(
        subset="validation", shuffle=False)
    
    train_data = tkam.set_up_data(
        tkam.train_dir, tkam.train_generator_kwargs, tkam.train_data_kwargs)
    valid_data = tkam.set_up_data(
        tkam.valid_dir, tkam.valid_generator_kwargs, tkam.valid_data_kwargs)
    print('class_indices:', train_data.class_indices)
    
    # Compose_model
    tkam.num_classes = train_data.num_classes
    tkam.model_fit_kwargs = dict(epochs=10)
    model = tkam.compose_model()
    
    # Compile model
    model = tkam.compile_model(model)
    
    # Train model
    model, _ = tkam.train_model(model, train_data, valid_data)
    
    # Save model
    tkam.save_model(model)
