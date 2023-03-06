from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def load_model(size=None):

    if size == None:
        size = (224, 224,3)

    vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=size))
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
    bboxHead = Dense(2, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)
    keyword = 'vgg'

    return model, keyword
