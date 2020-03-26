import keras.layers as layers
import keras
import keras.backend as K

# import tensorflow.keras.layers as layers
# import tensorflow.keras.backend as K
# import tensorflow.keras as keras
# import tensorflow as tf

class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return K.dot(inputs, self.w) + self.b


class MLPBlock(layers.Layer):

  def __init__(self):
    super(MLPBlock, self).__init__()
    self.linear_1 = Linear(32)
    self.linear_2 = Linear(32)
    self.linear_3 = Linear(1)

  def call(self, inputs):
    x = self.linear_1(inputs)
    x = keras.activations.relu(x)
    x = self.linear_2(x)
    x = keras.activations.relu(x)
    return self.linear_3(x)


input_tf = layers.Input(shape=(10,))
output = MLPBlock()(input_tf)
model = keras.models.Model(inputs=input_tf, outputs=output)
model.summary()
