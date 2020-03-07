#Code to handle POST requests and return the results

#we will use the flask web framework to handle the POST requests that we will get from the request.py.

import numpy as np
from flask import Flask, request, jsonify
import pickle

#we have created the instance of the Flask() and loaded the model into the model.
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

#we have bounded /api with the method predict(). 
# In which predict method gets the data from the json passed by the requestor. 
# model.predict() method takes input from the json and converts it into 2D numpy array the 
# results are stored into the variable named output and we return this variable 
# after converting it into the json object using flasks jsonify() method.

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)