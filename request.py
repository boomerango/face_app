import requests
API_ENDPOINT = "http://127.0.0.1:8000/face"

def sendTags(age, gender):
    data ={"age":age,"Gender":gender}
    r = requests.post(url=API_ENDPOINT, json=data)
    response = r.text
    print(f"response {response}")