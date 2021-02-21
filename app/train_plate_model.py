import pathlib

import numpy as np
import tensorflow as tf

from app.common import char_dict


async def train_model():
    all_image_path = [str(p) for p in pathlib.Path('./dataset/labeled').glob('*/*')]

    n = len(all_image_path)
    x_train, y_train = [], []
    for i in range(n):
        path = all_image_path[i]
        print('正在读取 {}'.format(all_image_path[i]))
        img = tf.io.read_file(path + '/plate.jpeg')
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [80, 240])
        img /= 255.0

        plate = pathlib.Path(path).name
        label = [char_dict[name] for name in plate[0:8]]
        if len(label) == 7:
            label.append(65)
        x_train.append(img)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = [np.array(y_train)[:, i] for i in range(8)]

    input_layer = tf.keras.layers.Input((80, 240, 3))
    x = input_layer
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)

    for i in range(3):
        x = tf.keras.layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = [tf.keras.layers.Dense(66, activation='softmax', name='c%d' % (i + 1))(x) for i in range(8)]

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50)
