#Send requests with the features to the server and receive the results

#request.py is going to request the server for the predictions.

import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':1.8,})
print(r.json())