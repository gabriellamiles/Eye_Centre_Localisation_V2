import os
import datetime

import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras.callbacks import TensorBoard, ModelCheckpoint

class VGG_model:
    def __init__(
            self, 
            input_shape = None,
            learning_rate = 1e-4,
            loss = "mse"
            ):
        
        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = None

        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = None

        if loss is not None:
            self.loss = loss
        else:
            self.loss = None

        # initialise other parameters
        self.batch_size = 32
        self.epochs = 10

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        # build and compile model
        self.build_model()
        self.compile_model()

        # construct callbacks
        self.initiate_callbacks()

    def build_model(self):

        vgg = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=self.input_shape))

        # freeze all VGG layers so they will *not* be updated during the
        # training process
        vgg.trainable = False

        # flatten the max-pooling output of VGG
        flatten = vgg.output
        flatten = Flatten()(flatten)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=vgg.input, outputs=bboxHead)
        self.keyword = 'vgg'
    
    def compile_model(self):

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def initiate_callbacks(self):
        """ Function to initiate callbacks, allowing recording of intermediate training results
        which can be visualised on TensorBoard. """

        subFolderLog = self.keyword + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboardPath = os.path.join(os.getcwd(), "models", subFolderLog)
        checkpointPath = os.path.join(os.getcwd(), "models" , subFolderLog)
        checkpointPath = checkpointPath + "/" + self.keyword + "-weights-improve-{epoch:02d}-{val_loss:02f}.hdf5"
        
        self.callbacks = [
            TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ]

    def train_model(self, train_data, train_labels, val_data, val_labels):

        hist = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )
