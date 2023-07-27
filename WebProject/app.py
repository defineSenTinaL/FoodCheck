from flask import Flask, render_template, request
from fastai.vision.all import *
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def GetLabel(fileName):
    return fileName.split('-')[0]

modelPath = "models/Food.pkl"
learn_inf = load_learner(modelPath)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('index.html', error='No file uploaded!')
        if file.filename == '':
            return render_template('index.html', error='No file selected!')
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return render_template('index.html', error='Invalid file format! Only JPG, JPEG, and PNG files are allowed.')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        try:
            food,_,preds = learn_inf.predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        except Exception as e:
            print(e)
            return render_template('index.html', error='An error occurred while processing the file!')
        return render_template('result.html', food=food, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)