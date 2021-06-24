#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use( 'tkagg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-image',
	            default='figure05.jpeg',
	            type=str)
	parser.add_argument('-threshold',
	            default=0.5,
	            type=float)
	args = parser.parse_args()

	# Old endpoint
	# endpoint = 'http://1754f79d-186b-48e3-a6d6-0b6b528aad12.northeurope.azurecontainer.io/score'
	
	# New endpoint
	endpoint = 'http://8081cf43-dab6-475f-8f71-77d63d686bee.northeurope.azurecontainer.io/score'
	path_img = '../../data/external/' + args.image
	threshold = args.threshold

	img = Image.open(path_img).convert('RGB')

	json_image = json.dumps(np.array(img).tolist())
	data = {
	'image': json_image,
	'threshold': threshold
	}
	headers = {'Content-Type':'application/json'}

	response = requests.post(endpoint, data=json.dumps(data), headers=headers)

	print(response)

	img = Image.fromarray(np.array(json.loads(response.content), dtype='uint8'))
	img.save('result.jpeg')

	plt.figure(figsize=(20, 30))
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.show()
