# coding=utf-8
import requests
import json

#req = requests.get('http://127.0.0.1:5000')

req = requests.post('http://127.0.0.1:5000/predict', json = {'image':'https://raw.githubusercontent.com/GeeKoders/OCR/main/chineseocr/images/testing-data/test.png', 'top': 5})

result = json.dumps(json.loads(req.content), ensure_ascii=False).encode('utf8') 
print(result)