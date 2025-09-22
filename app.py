from flask import (
    Flask, request, render_template, redirect, url_for, flash,
    send_from_directory, session, jsonify
)
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import sqlite3
import re
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from video_processing import predict_image, predict_video  # ML logic here

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app compatibility
app.secret_key = 'my_super_secret_12345'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----- Utility Functions -----

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_otp_email(recipient_email, otp):
    sender_email = "sukruthgdgd@gmail.com"
    sender_password = "sole jrfx svou dnst"  # Gmail app password
    subject = "Your OTP for Password Reset"
    body = f"Hello,\n\nYour OTP is: {otp}\n\n- Fake Video Detection Team"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Failed to send email:", e)

# ----- Web Routes -----

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'attempts' not in session:
        session['attempts'] = 0
    show_forgot = False

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            flash("Email not registered.", "error")
            return redirect(url_for('register'))
        elif user[2] != password:
            session['attempts'] += 1
            flash("Incorrect password.", "error")
        else:
            session['user'] = email
            session['attempts'] = 0
            flash("Login successful!", "success")
            return redirect(url_for('upload'))

        if session['attempts'] >= 2:
            show_forgot = True

    return render_template('login.html', show_forgot=show_forgot)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for('register'))

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format.", "error")
            return redirect(url_for('register'))

        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$', password):
            flash("Password must be strong.", "error")
            return redirect(url_for('register'))

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')

        try:
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
            conn.commit()
            flash("Registered successfully!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists.", "error")
        conn.close()

    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        flash("Login required.", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('image') or request.files.get('video')
        if not file or not allowed_file(file.filename):
            flash("Invalid or no file uploaded.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        result = predict_image(path) if filename.lower().endswith(('jpg', 'jpeg', 'png')) else predict_video(path)
        return redirect(url_for('result', prediction=result['label'], filename=filename))

    return render_template('upload.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    filename = request.args.get('filename')
    video_url = url_for('uploaded_file', filename=filename)

    if prediction.upper() == "FAKE":
        features = ["Unnatural warping", "Blinking issues", "Lip sync mismatch", "Color boundary artifacts", "Pose inconsistency"]
    else:
        features = ["Natural face movement", "Normal blinking", "Synced lip and audio", "Even lighting", "Stable pose"]

    return render_template('result.html', prediction=prediction, filename=filename, video_url=video_url, features=features)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            otp = str(random.randint(100000, 999999))
            session['otp'] = otp
            session['reset_email'] = email
            send_otp_email(email, otp)
            flash("OTP sent to your email.", "success")
            return redirect(url_for('verify_otp'))
        else:
            flash("Email not found.", "error")

    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered = request.form['otp']
        if entered == session.get('otp'):
            return redirect(url_for('reset_password'))
        else:
            flash("Incorrect OTP.", "error")
    return render_template('verify_otp.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        password = request.form['new_password']
        email = session.get('reset_email')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET password = ? WHERE email = ?', (password, email))
        conn.commit()
        conn.close()
        flash("Password updated.", "success")
        return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

# ----- API Endpoints for Mobile App -----

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'status': 'error', 'message': 'Email not registered'}), 404
    if user[2] != password:
        return jsonify({'status': 'error', 'message': 'Incorrect password'}), 401
    return jsonify({'status': 'success', 'message': 'Login successful', 'email': email})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Email already exists'}), 409
    return jsonify({'status': 'success', 'message': 'Registered successfully'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid or missing file'}), 400

    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    result = predict_image(path) if filename.lower().endswith(('jpg', 'jpeg', 'png')) else predict_video(path)

    return jsonify({
        'status': 'success',
        'label': result['label'],
        'filename': filename
    })

# ----- Run App -----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=5000, debug=True)

