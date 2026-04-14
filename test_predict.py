import requests

url = "http://localhost:8000/predict"
with open("hpap_subset_500.h5ad", "rb") as f:
    files = {"file": f}
    print("Sending request...")
    res = requests.post(url, files=files)
    
print("Status code:", res.status_code)
if res.status_code != 200:
    print(res.text)
