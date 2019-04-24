Requirements:
- Python 3.5

To setup the application, please run the following:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


In order to run the application:
```bash
export FLASK_APP=app.py
export FLASK_DEBUG=1
flask run
```

The Flask_Debug variable indicates if the application will auto reload when there is any change to the code.
A value of 1 means that it will auto reload.

At the moment there are 2 addresses available:
- `/` which is a hello world (just for diagnosis)
- `/functional_chat` which will support a simple functional chat bot