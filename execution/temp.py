import tensorflow as tf
import timeit

@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


# x = tf.random.uniform((3, 3))
# y = tf.random.uniform((3, 3))
#
# print(simple_nn_layer(x, y))
# print(simple_nn_layer)

class CustomModel(tf.keras.models.Model):
  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      return input_data // 2


class CustomModel_1(tf.keras.models.Model):
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      return input_data // 2


x = tf.constant([-2, -4])

model = CustomModel()
model(x)

model_1 = CustomModel_1()
model_1(x)

print("Time execution eager: {}".format(timeit.timeit(lambda: model_1(x), number=100)))
print("Time execution tf.function: {}".format(timeit.timeit(lambda: model(x), number=100)))