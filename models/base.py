import tensorflow.keras as keras


def get_baseline_model():
    x = keras.layers.Input(shape=(28, 28, 1))
    mid = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    mid = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.Activation(activation="relu")(mid)
    mid = keras.layers.MaxPool2D()(mid)
    mid = keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(mid)
    mid = keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.Activation(activation="relu")(mid)
    mid = keras.layers.MaxPool2D()(mid)
    mid = keras.layers.Flatten()(mid)
    y_logit = keras.layers.Dense(10, name='logit')(mid)

    return keras.models.Model(
        inputs=x,
        outputs=y_logit
    )