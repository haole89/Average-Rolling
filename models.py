import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam


class Extractor():
    def __init__(self, nb_classes, img_size):
        """        
        nb_classes: the number of classes to predict
        img_size: the size of image e.g., for imagenet 224x224      
        """
        # set defaults        
        self.nb_classes = nb_classes
        self.img_size = img_size
        self.model = self.Rmodel()
        
        print(self.model.summary())
    
    def Rmodel(self):
        # load ResNet50 CNN without including the top-layer

        input_tensor = Input(shape=(self.img_size, self.img_size, 3))

        baseModel = NASNetMobile(
            weights="imagenet",
            include_top=False,
            input_tensor= input_tensor,
            input_shape= (self.img_size, self.img_size, 3)

        )
        # construct the head of the model that will be placed on top of the 
        # the base model

        x = baseModel.output
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten(name="Flatten")(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(self.nb_classes, activation="softmax")(x)

        model = Model(inputs=input_tensor, outputs=output)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the training process
        for layer in baseModel.layers:
            layer.trainable = False
        
        # compile our model (this needs to be done after our setting our
        # layers to being non-trainable)
        print("[INFO] compiling model...")
        # trains fast as Adam, generalizes well as SGD. momentum=0.9
        opt = SGD(learning_rate= 1e-4, decay=1e-4, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model






