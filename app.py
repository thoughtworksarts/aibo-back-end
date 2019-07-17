from flask import Flask, json, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/echo')
def echo():
    question = request.args['question']
    return 'You have asked: ' + question

