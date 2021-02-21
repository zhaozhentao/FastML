import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File

from app.common import locate, index_to_char, create_mask

app = FastAPI()
detect_model = tf.keras.models.load_model('models/zc.h5')
recognition_model = tf.keras.models.load_model('models/plate.h5')


@app.post("/")
def recognize(file: bytes = File(...)):
    begin = time.time()
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = img / 255.0

    result = detect_model.predict(np.array([img]))

    mask = create_mask(result)
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.asarray(mask)
    img = np.asarray(img)

    plate_image = locate(img, mask)
    plate_chars = recognition_model.predict(np.array([plate_image]))
    plate = []
    for cs in plate_chars:
        plate.append(index_to_char[np.argmax(cs)])

    print('耗时: {}'.format(time.time() - begin))

    return {'plate': ''.join(plate)}
