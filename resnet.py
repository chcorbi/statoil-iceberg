from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from keras.models import Model

class ResNetModel(object):
    def __init__(self, options):
        self.build()

    def build(self):
        base_model = ResNet50(include_top=False)
        y = base_model.layers[-1].get_output_at(0)
        y = GlobalAveragePooling2D()(y)
        y = Dense(1)(y)
        output = Activation('sigmoid')(y)
        self.model = Model(inputs=[base_model.input],
                           outputs=[output])
