from flask import Flask, request, jsonify, render_template, redirect, url_for
from vosk import Model, KaldiRecognizer
import subprocess
import json
import pickle
import os
import tempfile
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = "/Users/manutsanan/Downloads/project-nsc/vosk-inference/model"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("ไม่พบโมเดล Vosk ที่ระบุ")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bully_records.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

with open('bully_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('bully_model.pkl', 'rb') as f:
    bully_model = pickle.load(f)

model = Model(MODEL_PATH)

class BullyRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    proba = db.Column(db.String, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    teacher_checked = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'ไม่มีไฟล์เสียง'}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        temp.flush()

    try:
        result = subprocess.run([
            'ffmpeg', '-loglevel', 'quiet', '-i', temp.name,
            '-ar', '16000', '-ac', '1', '-f', 'wav', 'pipe:1'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            return jsonify({'error': 'ffmpeg แปลงไฟล์ไม่สำเร็จ'}), 500

        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)

        if recognizer.AcceptWaveform(result.stdout):
            final_result = recognizer.Result()
        else:
            final_result = recognizer.FinalResult()

        text = json.loads(final_result).get("text", "")

        if text.strip():
            X_test = vectorizer.transform([text])
            prediction = bully_model.predict(X_test.toarray())[0]
            proba = bully_model.predict_proba(X_test.toarray())[0].tolist()

            record = BullyRecord(
                text=text,
                prediction=int(prediction),
                proba=json.dumps(proba)
            )
            db.session.add(record)
            db.session.commit()

            return jsonify({
                'text': text,
                'prediction': int(prediction),
                'proba': proba
            })
        else:
            return jsonify({'text': '', 'prediction': -1, 'proba': []})

    finally:
        os.remove(temp.name)

@app.route('/teacher')
def teacher_dashboard():
    records = BullyRecord.query.order_by(BullyRecord.timestamp.desc()).all()
    return render_template('teacher.html', records=records)

@app.route('/teacher/check/<int:record_id>', methods=['POST'])
def teacher_check(record_id):
    record = BullyRecord.query.get_or_404(record_id)
    record.teacher_checked = True
    db.session.commit()
    return redirect(url_for('teacher_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
