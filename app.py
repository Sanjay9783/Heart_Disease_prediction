from flask import Flask,render_template,request
import pickle
import numpy as np

# load the model from disk
loaded_model=pickle.load(open('heart_trained_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	age = int(request.form.get('age'))
	sex = int(request.form.get('sex'))
	cp = int(request.form.get('cp'))
	trestbps = int(request.form.get('trestbps'))
	chol = int(request.form.get('chol'))
	fbs = int(request.form.get('fbs'))
	restecg = int(request.form.get('restecg'))
	thalach = int(request.form.get('thalach'))
	exang = int(request.form.get('exang'))
	oldpeak = float(request.form.get('oldpeak'))
	slope = int(request.form.get('slope'))
	ca = int(request.form.get('ca'))
	thal = int(request.form.get('thal'))


	# prediction
	result = loaded_model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,13))

	if result [0]== 1:
		result = 'The Person does not have a Heart Disease'
	else:
		result = 'The Person has Heart Disease'

	return render_template('index.html', result=result)

if __name__ == '__main__':
	app.run(debug=True)