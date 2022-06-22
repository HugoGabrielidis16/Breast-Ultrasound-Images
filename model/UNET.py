import tensorflow as tf


def block_conv_down(x, dims):
    x = tf.keras.layers.Conv2D(dims, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(dims, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    p = tf.keras.layers.Dropout(0.2)(p)
    return x, p


def bottleneck(x, dims):
    x = tf.keras.layers.Conv2D(dims, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(dims, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def block_conv_up(x, p, dims):
    p = tf.keras.layers.UpSampling2D(2)(p)
    u = tf.keras.layers.Concatenate(axis=-1)([x, p])

    u = tf.keras.layers.Conv2D(dims, (3, 3), padding="same")(u)
    u = tf.keras.layers.BatchNormalization()(u)
    u = tf.keras.layers.Activation("relu")(u)
    u = tf.keras.layers.Dropout(0.1)(u)

    u = tf.keras.layers.Conv2D(dims, (3, 3), padding="same")(u)
    u = tf.keras.layers.BatchNormalization()(u)
    u = tf.keras.layers.Activation("relu")(u)
    u = tf.keras.layers.Dropout(0.1)(u)

    return u


def UNET(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    x1, p1 = block_conv_down(input, 64)
    # print(f" x1 shape : {x1.shape}")
    x2, p2 = block_conv_down(p1, 128)
    # print(f" x2 shape : {x2.shape}")
    x3, p3 = block_conv_down(p2, 256)
    # print(f" x3 shape : {x3.shape}")
    x4, p4 = block_conv_down(p3, 512)
    # print(f" x4 shape : {x4.shape}")

    p5 = bottleneck(p4, 1024)
    # print(f" p5 shape : {p5.shape}")
    p6 = block_conv_up(x4, p5, 512)
    # print(f" p6 shape : {p6.shape}")
    p7 = block_conv_up(x3, p6, 256)
    # print(f" p7 shape : {p7.shape}")
    p8 = block_conv_up(x2, p7, 128)
    # print(f" p8 shape : {p8.shape}")
    p9 = block_conv_up(x1, p8, 64)

    # print(f" p9 shape : {p9.shape}")

    output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(p9)
    # output = tf.keras.layers.Conv2D(2,(1,1), activation = "softmax")
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = UNET((256, 256, 1))
    # print(model.summary())
