import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'timestamp ':18011, 'day_of_week':4, 'temperature':71.76, 'month':7, 'hour':17, 'is_weekend':1, 'is_holiday':0 })

print(r.json())