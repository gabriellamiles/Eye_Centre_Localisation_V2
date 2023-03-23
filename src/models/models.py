import os
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, Xception, InceptionResNetV2
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError

from keras.callbacks import TensorBoard, ModelCheckpoint

class DL_model():
    def __init__(
        self, 
        input_shape = None,
        test_num = None,
        output_parameters = None,
        batch_size=None,
        learning_rate = 1e-4,
        loss = "mse",
        ):
    
        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = None

        if test_num is not None:
            self.test_num = str(test_num)
        else:
            self.test_num = None    

        if output_parameters is not None:
            self.output_parameters = output_parameters
        else:
            self.output_parameters = None

        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = None

        if loss is not None:
            self.loss = loss
        else:
            self.loss = None

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = None

        self.epochs = 25

    def initiate_callbacks(self):
        """ Function to initiate callbacks, allowing recording of intermediate training results
        which can be visualised on TensorBoard. """

        self.subFolderLog = "test_" + self.test_num + "_" + self.keyword + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        tensorboardPath = os.path.join(os.getcwd(), "models", self.subFolderLog)
        checkpointPath = os.path.join(os.getcwd(), "models" , self.subFolderLog, "{epoch:02d}-{val_loss:.4f}.hdf5")
        
        self.callbacks = [
            TensorBoard(log_dir=tensorboardPath, histogram_freq=0, write_graph=True, write_images=True),
            ModelCheckpoint(filepath=checkpointPath, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        ]

    def plot_loss_curves(self):
        """Plots graph of training and validation loss """

        # loss plot
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(os.path.join(os.getcwd(), "models", self.subFolderLog, "loss_plot.png"))
        # plt.show()

class VGG_model(DL_model):

    """ Input shape default is (224, 224, 3) """

    def __init__(self, input_shape=None, test_num = None, output_parameters=None, batch_size=None, learning_rate=0.0001, loss="mse"):
        super().__init__(input_shape, test_num, output_parameters, batch_size, learning_rate, loss)

        self.keyword = "vgg"

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        self.initiate_callbacks()
        
        # build and compile model
        self.build_model()
        self.compile_model()

    def build_model(self):
        input_tensor=Input(shape=self.input_shape)
        x = tf.keras.applications.vgg16.preprocess_input(input_tensor)

        vgg = VGG16(weights="imagenet", include_top=False)
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        vgg.trainable = False

        x = vgg(x)
        # flatten the max-pooling output of VGG
        flatten = Flatten()(x)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(self.output_parameters, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=input_tensor, outputs=bboxHead)
    
    def compile_model(self):

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def train_model(self, train_generator, val_generator):

        self.hist = self.model.fit(
            train_generator,
            validation_data=val_generator,
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )
    
    def predict_model(self, predict_generator):

        y_pred = self.model.predict(
            predict_generator,
            batch_size=self.batch_size,
            verbose=1
        )

        y_pred = y_pred*960 

        return y_pred


class Inception_model(DL_model):
    """ Input to inception model has default shape of (299, 299, 3) and input pixels of between -1 and 1. """

    def __init__(self, input_shape=None, test_num = None, output_parameters=None, batch_size=None, learning_rate=0.0001, loss="mse"):
        super().__init__(input_shape, test_num, output_parameters, batch_size, learning_rate, loss)

        self.keyword = "inception"

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        self.initiate_callbacks()
        
        # build and compile model
        self.build_model()
        self.compile_model()

    def build_model(self):


        input_tensor=Input(shape=self.input_shape)
        x = tf.keras.applications.inception_v3.preprocess_input(input_tensor)

        inception = InceptionV3(weights="imagenet", include_top=False)
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        inception.trainable = False

        x = inception(x)
        # flatten the max-pooling output of inception
        flatten = Flatten()(x)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(self.output_parameters, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=input_tensor, outputs=bboxHead)
    
    def compile_model(self):

        self.model.compile(loss=self.loss, 
                           optimizer=self.opt,
                           metrics=MeanSquaredError())

    def train_model(self, train_data, train_labels, val_data, val_labels):

        self.hist = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )

    def predict_model(self, predict_generator):

        y_pred = self.model.predict(
            predict_generator,
            batch_size=self.batch_size,
            verbose=1
        )

        y_pred = y_pred*960 

        return y_pred

class Xception_model(DL_model):
    """ Input to inception model has default shape of (299, 299, 3) and input pixels of between -1 and 1. """

    def __init__(self, input_shape=None, test_num = None, output_parameters=None, batch_size=None, learning_rate=0.0001, loss="mse"):
        super().__init__(input_shape, test_num, output_parameters, batch_size, learning_rate, loss)

        self.keyword = "xception"

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        self.initiate_callbacks()
        
        # build and compile model
        self.build_model()
        self.compile_model()

    def build_model(self):

        input_tensor=Input(shape=self.input_shape)
        x = tf.keras.applications.xception.preprocess_input(input_tensor)
        
        xception = Xception(weights="imagenet", include_top=False,
        )
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        xception.trainable = False

        x = xception(x)

        # flatten the max-pooling output of inception
        flatten = Flatten()(x)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(self.output_parameters, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=input_tensor, outputs=bboxHead)
    
    def compile_model(self):

        self.model.compile(loss=self.loss, 
                           optimizer=self.opt,
                           metrics=MeanSquaredError())

    def train_model(self, train_data, train_labels, val_data, val_labels):

        self.hist = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )
    def predict_model(self, predict_generator):

        y_pred = self.model.predict(
            predict_generator,
            batch_size=self.batch_size,
            verbose=1
        )

        y_pred = y_pred*960 

        return y_pred
class ResNet50_model(DL_model):

    """ Input shape default is (224, 224, 3) """

    def __init__(self, input_shape=None, test_num = None, output_parameters=None, batch_size=None, learning_rate=0.0001, loss="mse"):
        super().__init__(input_shape, test_num, output_parameters, batch_size, learning_rate, loss)

        self.keyword = "resnet50"

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        self.initiate_callbacks()
        
        # build and compile model
        self.build_model()
        self.compile_model()

    def build_model(self):


        input_tensor=Input(shape=self.input_shape)
        x = tf.keras.applications.resnet50.preprocess_input(input_tensor)

        resnet = ResNet50(weights="imagenet", include_top=False)
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        resnet.trainable = False

        x = resnet(x)

        # flatten the max-pooling output of VGG
        flatten = Flatten()(x)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(self.output_parameters, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=input_tensor, outputs=bboxHead)
    
    def compile_model(self):

        self.model.compile(loss=self.loss, optimizer=self.opt)

    def train_model(self, train_data, train_labels, val_data, val_labels):

        self.hist = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )

    def predict_model(self, predict_generator):

        y_pred = self.model.predict(
            predict_generator,
            batch_size=self.batch_size,
            verbose=1
        )

        y_pred = y_pred*960 

        return y_pred

class InceptionResNetV2_model(DL_model):
    """ Input to inception model has default shape of (299, 299, 3) and input pixels of between -1 and 1. """

    def __init__(self, input_shape=None, test_num = None, output_parameters=None, batch_size=None, learning_rate=0.0001, loss="mse"):
        super().__init__(input_shape, test_num, output_parameters, batch_size, learning_rate, loss)

        self.keyword = "inception_resnet_v2"

        # construct optimizer
        self.opt = Adam(learning_rate=self.learning_rate)

        self.initiate_callbacks()
        
        # build and compile model
        self.build_model()
        self.compile_model()

    def build_model(self):


        input_tensor=Input(shape=self.input_shape)
        x = tf.keras.applications.inception_resnet_v2.preprocess_input(input_tensor)

        inception_resnet_v2 = InceptionResNetV2(weights="imagenet", include_top=False)
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        inception_resnet_v2.trainable = False

        x = inception_resnet_v2(x)

        # flatten the max-pooling output of inception
        flatten = Flatten()(x)

        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(self.output_parameters, activation="sigmoid")(bboxHead)
        
        # construct the model we will fine-tune for bounding box regression
        self.model = Model(inputs=input_tensor, outputs=bboxHead)
    
    def compile_model(self):

        self.model.compile(loss=self.loss, 
                           optimizer=self.opt,
                           metrics=MeanSquaredError())

    def train_model(self, train_data, train_labels, val_data, val_labels):

        self.hist = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            batch_size = self.batch_size,
            epochs = self.epochs,
            callbacks = self.callbacks,
            verbose = 1
        )

    def predict_model(self, predict_generator):

        y_pred = self.model.predict(
            predict_generator,
            batch_size=self.batch_size,
            verbose=1
        )

        y_pred = y_pred*960 

        return y_pred
