from flask import Flask, json
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/functional_chat')
def functional_chat():
    data = {'intent': 'sample intent',
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
