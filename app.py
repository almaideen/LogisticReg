from flask import Flask
from flask import request
from flask import render_template
import pickle

#initializing Flask app
app = Flask(__name__)

#route for Homepage
@app.route('/',methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def index():
    if request.method == 'POST':
        try:
            rate_marriage = float(request.form['rate_marriage'])
            age = float(request.form['age'])
            yrs_married = float(request.form['yrs_married'])
            children = float(request.form['children'])
            religious = float(request.form['religious'])
            educ = float(request.form['educ'])
            occupation = float(request.form['occupation'])
            occupation_husb = float(request.form['occupation_husb'])

            scaler = pickle.load(open('scaler.pickle','rb'))
            x = scaler.fit_transform([[rate_marriage,age,yrs_married,children,religious,educ,occupation,occupation_husb]])

            model = pickle.load(open('LogisticAssign.pickle','rb'))
            prediction = model.predict(x)
            return render_template('results.html',prediction=prediction)
        except Exception as e:
            return 'Something is Wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)