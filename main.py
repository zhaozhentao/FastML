import asyncio
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File

from app.baidu import recognize_with_baidu
from app.common import locate, index_to_char, create_mask, index_to_chinese
from train_plate_model import train_model_by_system_call

app = FastAPI()
detect_model = tf.keras.models.load_model('models/zc.h5')
recognition_model = tf.keras.models.load_model('models/plate.h5')


@app.post("/")
async def recognize(file: bytes = File(...)):
    begin = time.time()
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = img / 255.0
    result = detect_model.predict(np.array([img]))
    print('车牌mask耗时{}'.format(time.time() - begin))

    locate_begin = time.time()
    mask = create_mask(result)
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.asarray(mask)
    img = np.asarray(img)
    plate_image = locate(img, mask)
    print('车牌旋转耗时{}'.format(time.time() - locate_begin))

    ocr_begin = time.time()
    plate_chars = recognition_model.predict(np.array([plate_image]))
    plate = []
    for idx, cs in enumerate(plate_chars):
        char_index = np.argmax(cs)
        if idx == 0:
            plate.append(index_to_chinese[char_index])
        else:
            plate.append(index_to_char[char_index])
    print('ocr耗时{}'.format(time.time() - ocr_begin))

    predict_plate = ''.join(plate)
    asyncio.create_task(recognize_with_baidu(predict_plate, file, mask))

    print('耗时: {}'.format(time.time() - begin))
    return {'plate': predict_plate}


@app.get('/')
async def train():
    asyncio.create_task(train_model_by_system_call())
