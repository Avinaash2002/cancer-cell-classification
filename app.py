from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from cancer_detection1 import load_model, predict_image, annotate_image, get_label_names

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model and label names
model = load_model()
label_names = get_label_names()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction
        prediction = predict_image(model, file_path)
        label_idx = np.argmax(prediction)
        if label_idx < len(label_names):
            label = label_names[label_idx]
            prob = prediction[0][label_idx]
            
            # Annotate the image and save the result with a unique name
            result_image_path = annotate_image(file_path, label, prob)
            result_image_url = url_for('static', filename=f'uploads/result_{filename}')
            original_image_url = url_for('static', filename=f'uploads/{filename}')
            
            return render_template('result.html', result=label, original_image_url=original_image_url, result_image_url=result_image_url)
        else:
            return render_template('result.html', result="Error: Label index out of range.", original_image_url=None, result_image_url=None)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)

























