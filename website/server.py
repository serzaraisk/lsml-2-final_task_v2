from celery import Celery
from celery.result import AsyncResult
import time
from flask import Flask, request, render_template
import json
import pickle
import re

celery_app = Celery('server', backend='redis://localhost', broker='redis://localhost')
app = Flask(__name__)


def load_model(pickle_path):
    with open(pickle_path, 'rb') as f:
        raw_data = f.read()
        model = pickle.loads(raw_data)
    return model


#model = load_model('fmeter-model.pickle')


@celery_app.task
def freq(sentence):
    result = {}
    word_pattern = re.compile(r"[a-z]+")
    for match in word_pattern.finditer(sentence.lower()):
        word = match.group(0)
        result[word] = model.compute(word)
    return result


@app.route('/frequency', methods=["GET", "POST"])
def frequency_handler():
    if request.method == 'POST':
        data = request.get_json(force=True)
        sentence = data['sentence']

        task = freq.delay(sentence)

        response = {
            "task_id": task.id
        }
        return json.dumps(response)


@app.route('/frequency/<task_id>')
def frequency_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.ready():
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)


@app.route('/')  # handler for /
def start_page():
    return render_template('index.html', text='')


@app.route('/parse', methods=["GET", "POST"])
def parse_review():
    if request.method == 'POST':  # handle only POST requests
        data = request.data.decode()  # read the payload

        return data
    else:
        return "You should use only POST query"


if __name__ == '__main__':
    app.run("0.0.0.0", 8000)