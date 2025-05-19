from flask import Flask, request, render_template
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = ''
    user_input = ''
    if request.method == 'POST':
        user_input = request.form['sentence']
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity

        if polarity > 0.5:
            emotion = 'Very Happy 😊'
        elif polarity > 0:
            emotion = 'Happy 🙂'
        elif polarity == 0:
            emotion = 'Neutral 😐'
        elif polarity > -0.5:
            emotion = 'Sad 🙁'
        else:
            emotion = 'Very Sad 😢'

    return render_template('index.html', emotion=emotion, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
