from typing import Optional

from fastapi import FastAPI
import numpy as np
import tensorflow as tf

detect_model = tf.keras.models.load_model('zc.h5')
recognition_model = tf.keras.models.load_model('plate.h5')

app = FastAPI()


@app.post("/")
def recognize():
    pass

