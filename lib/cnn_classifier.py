# externals
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2


class CNNClassifier:
    """
    A convolutional neural network (CNN) classifier for multi-class image classification
    """

    def __init__(self, X:np.array, Y:np.array, input_shape:tuple, num_classes:int) -> None:
        """
        :np.array X: Training input data
        :np.array Y: Training labels corresponding to input data
        :tuple input_shape: Shape of the input images (height, width, channels)
        :int num_classes: Number of target classes for classification
        :keras.models model: Compiled Keras model for classification
        :float learning_rate: Initial learning rate for the optimizer
        :tf.keras.optimizers.Optimizer optimizer: Optimizer for model training
        :str loss: Loss function used during training
        :tf.keras.callbacks.LearningRateScheduler lr_scheduler: Callback for learning rate scheduling
        :tf.keras.callbacks.ModelCheckpoint checkpoint_cb: Callback for saving the best model during training
        :tf.keras.callbacks.EarlyStopping early_stopping_cb: Callback to stop training early when validation metrics stop improving
        :keras.callbacks.History history: Training history, including loss and accuracy
        """

        self.X = X
        self.Y = Y
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = self.build_model()
        
        self.learning_rate = 0.0005
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = 'sparse_categorical_crossentropy'

        self.lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.exponential_decay())
        self.checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)
        self.early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

        self.history = None


    def build_model(self) -> keras.models:
        """
        Constructs the CNN model architecture

        :returns keras.models: Keras CNN model
        """

        model=keras.Sequential()

        model.add(keras.layers.Conv2D(
            256, 
            kernel_size=(3,3), 
            input_shape=self.input_shape, 
            kernel_regularizer=l2(0.01)
            ))
        model.add(keras.layers.LeakyReLU(alpha=0.01))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(keras.layers.Conv2D(
            128, 
            kernel_size=(3,3), 
            kernel_regularizer=l2(0.01)
            ))
        model.add(keras.layers.LeakyReLU(alpha=0.01))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(keras.layers.Conv2D(
            64, 
            kernel_size=(3,3), 
            kernel_regularizer=l2(0.01)
            ))
        model.add(keras.layers.LeakyReLU(alpha=0.01))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(
            128, 
            activation='relu'
            ))
        model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        return model
    

    def exponential_decay(self, epochs_to_decay:int = 100) -> float:
        """
        Defines an exponential decay function for adjusting the learning rate during training

        :int epochs_to_decay: Number of epochs over which the learning rate decreases

        :returns float: The updated learning rate for the current epoch
        """

        def decay(epoch):
            return self.learning_rate * 0.99 ** (epoch / epochs_to_decay)
        
        return decay
    

    def compile_model(self) -> None:
        """
        Compiles the CNN model
        """

        self.model.compile(optimizer = self.optimizer, 
                           loss = self.loss, 
                           metrics = ['accuracy'])

    
    def train_model(self, epochs:int = 50, batch_size:int = 32) -> keras.models:
        """
        Trains the CNN model on the input data

        :int epochs: Number of epochs to train the model
        :int batch_size: Batch size for training

        :returns keras.models: Trained Keras model with updated weights
        """

        self.compile_model()

        history = self.model.fit(

            self.X,
            self.Y, 

            validation_split=0.1, 
            epochs=epochs,
            batch_size=batch_size,

            verbose=1,

            callbacks=[
                self.checkpoint_cb, 
                self.early_stopping_cb, 
                self.lr_scheduler
            ]
        )

        self.history = history
    

    def get_history(self):
        """
        Retrieves the training history of the model

        :returns keras.callbacks.History: History object containing loss and accuracy metrics
        """

        return self.history
    

    def get_model(self):
        """
        Retrieves the trained CNN model

        :returns keras.models: The trained CNN model instance
        """

        return self.model
