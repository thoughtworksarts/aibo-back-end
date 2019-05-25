from flask import Flask, json, request
from flask_cors import CORS
# from chatbot import chatbot

app = Flask(__name__)
CORS(app)

# chatbot = chatbot.Chatbot()
# chatbot.main()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/functional_chat')
def functional_chat():
    data = {'intent': 'Hardcoded response',
            'entities': [
                {
                    'type': 'name',
                    'value': 'alice'
                }
            ]}
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/wacky_chat', methods=['POST'])
def wacky_chat():
    request_json = request.json
    question = 'Hi' if request_json is None else request_json.get('question', 'Hi')
    data = {'intent': 'testing', #chatbot.answer(question),
            'entities': [
                {
                    'type': 'name',
                    'value': 'alice'
                }
            ]}
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response
