import requests
url = 'http://127.0.0.1:5000/predict'
files = {'file': open('kaggle/test/6. Malign/actinic-cheilitis-sq-cell-lip-3.jpg', 'rb')}
req = requests.post(url, files=files)

print(req.text)