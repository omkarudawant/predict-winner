import flask
import joblib
from pandas import DataFrame as df
from flask import request

app = flask.Flask(__name__, template_folder='templates')

model = joblib.load('model/model.joblib')


@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return flask.render_template('index.html')

	if flask.request.method == 'POST':
		FTHG = flask.request.form['FTHG']
		FTAG = flask.request.form['FTAG']
		HTHG = flask.request.form['HTHG']
		HTAG = flask.request.form['HTAG']
		HS = flask.request.form['HS']
		AS = flask.request.form['AS']
		HST = flask.request.form['HST']
		AST = flask.request.form['AST']
		HF = flask.request.form['HF']
		AF = flask.request.form['AF']
		HC = flask.request.form['HC']
		AC = flask.request.form['AC']
		HY = flask.request.form['HY']
		AY = flask.request.form['AY']
		HR = flask.request.form['HR']
		AR = flask.request.form['AR']
		data = df([[FTHG, FTAG, HTHG, HTAG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR]],
			columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
			'HC', 'AC', 'HY', 'AY', 'HR', 'AR'], dtype=float)
		data = data.loc[[0]]
		print(data)
		input = df(data)
		print(input)
		prediction = model.predict(input)
		print(f'Pred: {prediction}')
		return flask.render_template('index.html',
			original_input = {'FTHG':FTHG, 'FTAG':FTAG, 'HTHG':HTHG,
			'HTAG':HTAG, 'HS':HS, 'AS':AS, 'HST':HST, 'AST':AST, 'HF':HF, 'AF':AF,
			'HC':HC, 'AC':AC, 'HY':HY, 'AY':AY, 'HR':HR, 'AR':AR}, result=prediction)


if __name__ == '__main__':
	app.run(port=5001, debug=True)
