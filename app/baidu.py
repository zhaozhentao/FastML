import base64
import json
import time

import numpy as np
import requests
import tensorflow as tf

import config
from app.common import locate


async def recognize_with_baidu(predict_plate, file, mask):
    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}

    # get token
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(
        config.Settings().client_id, config.Settings().client_secret)
    res = requests.get(host, headers=headers).text
    token = json.loads(res)['access_token']

    # recognize
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate?access_token=" + token
    image = base64.b64encode(file)
    respond = requests.post(url, data={'image': image, 'multi_detect': 'false'}, headers=headers)
    respond.encoding = 'utf-8'
    words_result = json.loads(respond.text)

    if 'words_result' not in words_result.keys():
        print('%s未识别')
        return

    baidu_result = words_result['words_result']['number']
    if predict_plate == baidu_result:
        print('识别结果一致')
        return

    print('识别结果不一致，百度识别结果:{}'.format(baidu_result))
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize(img, [416, 416])
    img = np.asarray(img)
    plate_image = locate(img, mask)
    file_path = './dataset/error/' + time.strftime("%Y_%m_%d_%H", time.localtime()) + '/' + baidu_result + '/plate.jpeg'
    tf.io.write_file(file_path, tf.image.encode_jpeg(plate_image))
