import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('/home/dima/PycharmProjects/DeployFlask/cat1.jpg','rb')})

print(resp.json())
