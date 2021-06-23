#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np

# Set the content type
endpoint = 'http://1754f79d-186b-48e3-a6d6-0b6b528aad12.northeurope.azurecontainer.io/score'
path_img = '../data/external/figure05.jpeg'
threshold = 0.4

img = Image.open(path_img).convert('RGB')
# x = TF.to_tensor(img)
# x = x.reshape(1, *x.shape)
# js = json.dumps(x.tolist())

# data = {
#     'image': js,
#     'threshold':threshold
# }

# response = requests.post(endpoint, data=json.dumps(data))

json_image = json.dumps(np.array(img).tolist())
data = {
    'image': json_image,
    'threshold': threshold
}
headers = {'Content-Type':'application/json'}

response = requests.post(endpoint, data=json.dumps(data), headers=headers)

print(response)
# print('I hear voices: ', response.content)
img = Image.fromarray(np.array(json.loads(response.content), dtype='uint8'))
img.save('result.jpeg')
# def send_request():
#     payload = {"param_1": "value_1", "param_2": "value_2"}
#     files = {
#         'json': (None, json.dumps(payload), 'application/json'),
#         'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
#     }

#     r = requests.post(url, files=files)
#     print(r.content)


# plt.figure(figsize=(20, 30))
# plt.imshow(predictions)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# plt.savefig('result.jpeg')