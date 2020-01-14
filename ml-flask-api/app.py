"""
 Serving KNN model as Web API service with Python Flask
 
 Marian Witkowski
 https://www.linkedin.com/in/marianwitkowski/
"""

from flask import Flask, request, Response, render_template
from sklearn.externals import joblib
import pandas as pd
import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# load saved model from file
model = joblib.load(os.path.join(dir_path, "knn.model"))

app = Flask(__name__)

"""
    render form template
"""
@app.route("/")
def hello():
    return render_template("form.html")


"""
    put params in GET request query 
    sl - sepal length
    sw - sepal width
    pl - petal length
    pw - petal width
    
    request example: http://127.0.0.1:5000/predict?sl=5.1&sw=3.5&pl=1.4&pw=0.15
"""
@app.route("/predict" , methods=['GET'])
def predict():
    try:
        sl = float( request.args.get('sl', 0) )
        sw = float( request.args.get('sw', 0) )
        pl = float( request.args.get('pl', 0) )
        pw = float( request.args.get('pw', 0) )

        if sl<=0 or sw<=0 or pl<=0 or pw<=0:
            raise ValueError("incorrect value")

        x_test = pd.np.array([sl, sw, pl, pw]) # create test data
        y_pred = model.predict(x_test.reshape(1,-1))[0] # get predict
        if y_pred==0:
            result = "Iris setosa"
        elif y_pred==1:
            result = "Iris versicolor"
        elif y_pred==2:
            result = "Iris virginica"
        else:
            result = "uknown species"

        response =  { "code" : 0, "message" : result }
        return Response(json.dumps(response), mimetype='application/json')

    except Exception as exc:
        response =  { "code" : -1,  "message" : str(exc) }
        return Response(json.dumps(response), mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)

