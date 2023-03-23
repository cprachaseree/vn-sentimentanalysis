from flask import Flask, render_template, request
import pickle
import os
import sklearn
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer

app = Flask(__name__)

def tokenize_sent(sent):
    return ViTokenizer.spacy_tokenize(sent)[0]

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "app"
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

@app.route('/', methods=['post', 'get'])
def index():
    result = ''
    if request.method == 'POST':
        review = request.form.get('review')  # access the data inside
        if not review:
            return render_template('index.html', result=result)
        output_dir="models"
        vectorizer = MyCustomUnpickler(open(os.path.join(output_dir, "vectorizer.pickle"), "rb")).load()
        classifier = MyCustomUnpickler(open(os.path.join(output_dir, "classifier.pickle"), "rb")).load()
        text = [review]
        text_feat = vectorizer.transform(text)
        label2result = ["Negative", "Neutral", "Positive"]
        pred = classifier.predict(text_feat)
        print("pred:", pred)
        result = label2result[int(pred)]

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)