import json
import pickle

import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

REMEMBER_CHAT_HISTORY = False

app = Flask(__name__)
model = tf.keras.models.load_model('ml/my_llm_model.h5')
with open('ml/tokenizer', 'rb') as f:
    tokenizer = pickle.load(f)
with open('ml/config.json', 'r') as f:
    config = json.load(f)
    max_sequence_length = config['max_sequence_length']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    history = request.args.get('msg[history]').strip()
    last_msg = request.args.get('msg[last_msg]')
    request.args.getlist('msg[last_msg]')
    text = history + last_msg if REMEMBER_CHAT_HISTORY else last_msg
    response_words = []
    next_words = 15

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probabilities)
        output_word = tokenizer.index_word[predicted_index]
        response_words.append(output_word)
        text += ' ' + output_word

    return ' '.join(response_words)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
