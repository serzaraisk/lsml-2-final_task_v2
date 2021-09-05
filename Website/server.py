from flask import Flask, request, render_template
import json


from Website.utills import model_fn, predict_fn

app = Flask(__name__)
model_info = {'embedding_dim': 64, 'hidden_dim': 128, 'vocab_size': 5000}
model = model_fn('./', model_info)

with open("word_dict.json", "r") as read_file:
    word_dict = json.load(read_file)


@app.route('/')  # handler for /
def start_page():
    return render_template('index.html', text='')


@app.route('/parse', methods=["GET", "POST"])
def parse_review():
    if request.method == 'POST':  # handle only POST requests
        data = request.data.decode()  # read the payload
        data = str(int(predict_fn(data, model, word_dict)))
        return data
    else:
        return "You should use only POST query"


if __name__ == '__main__':
    app.run("0.0.0.0", 8000)