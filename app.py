from flask import Flask, request, jsonify
from scripts.compile_model import Model

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host="0.0.0.0")

model = Model()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            prediction = model.predict_class(file.stream)
            return jsonify({'prediction': prediction}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
@app.route('/')
def root():
    return "Hello World", 200