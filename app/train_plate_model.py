import pathlib

import tensorflow as tf

from app.common import char_dict


def load_and_process_image(image_path, l0, l1, l2, l3, l4, l5, l6, l7):
    image = tf.io.read_file(image_path + '/plate.jpeg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [80, 240])
    image /= 255.0
    return image, (l0, l1, l2, l3, l4, l5, l6, l7)


async def train_model():
    # 读取数据集
    all_image_path = [str(p) for p in pathlib.Path('./dataset/labeled').glob('*/*')]
    batch_size = 64
    image_count = len(all_image_path)
    label0, label1, label2, label3, label4, label5, label6, label7 = [], [], [], [], [], [], [], []

    for p in all_image_path:
        print('正在读取 {}'.format(p))
        plate = pathlib.Path(p).name
        label0.append(char_dict[plate[0]])
        label1.append(char_dict[plate[1]])
        label2.append(char_dict[plate[2]])
        label3.append(char_dict[plate[3]])
        label4.append(char_dict[plate[4]])
        label5.append(char_dict[plate[5]])
        label6.append(char_dict[plate[6]])
        if len(plate) == 7:
            label7.append(65)
        else:
            label7.append(char_dict[plate[7]])

    image_path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
    label0 = tf.data.Dataset.from_tensor_slices(label0)
    label1 = tf.data.Dataset.from_tensor_slices(label1)
    label2 = tf.data.Dataset.from_tensor_slices(label2)
    label3 = tf.data.Dataset.from_tensor_slices(label3)
    label4 = tf.data.Dataset.from_tensor_slices(label4)
    label5 = tf.data.Dataset.from_tensor_slices(label5)
    label6 = tf.data.Dataset.from_tensor_slices(label6)
    label7 = tf.data.Dataset.from_tensor_slices(label7)

    ds = (
        tf.data.Dataset.zip((image_path_ds, label0, label1, label2, label3, label4, label5, label6, label7))
            .map(load_and_process_image)
            .cache()
            .shuffle(buffer_size=image_count)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

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

    model.fit(ds, epochs=50)
