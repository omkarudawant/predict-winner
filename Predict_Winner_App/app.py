import flask
import pickle
from pandas import DataFrame as df
from flask import request

app = flask.Flask(__name__, template_folder='templates')

with open(f'model/XGBoost.pkl', 'rb') as model:
	xgb = pickle.load(model)


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
		prediction = xgb.predict(input)[0]
		return flask.render_template('index.html',
			original_input = {'FTHG':FTHG, 'FTAG':FTAG, 'HTHG':HTHG,
			'HTAG':HTAG, 'HS':HS, 'AS':AS, 'HST':HST, 'AST':AST, 'HF':HF, 'AF':AF,
			'HC':HC, 'AC':AC, 'HY':HY, 'AY':AY, 'HR':HR, 'AR':AR}, result=prediction)
while __name__ == '__main__':
	app.run(debug=True)
