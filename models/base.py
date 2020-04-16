import tensorflow.keras as keras


def get_baseline_model(width=28, height=28, n_channel=1):
    x = keras.layers.Input(shape=(width, height, n_channel))
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
        outputs=y_logit,
        name="Model"
    )