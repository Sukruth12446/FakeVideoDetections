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
from datetime import datetime
from gtts import gTTS
from flask import send_file
import tempfile
app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app compatibility
app.secret_key = 'my_super_secret_12345'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Admin credentials
ADMIN_EMAIL = "sukruth959@gmail.com"
ADMIN_PASSWORD = "Sukchinu@1234"


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


def init_database():
    """Initialize database with users table including role column"""
    try:
        # Only create database if it doesn’t exist
        if not os.path.exists('database.db'):
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()

            # Create users table
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )''')

            # Create uploads table
            cursor.execute('''CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES users (email)
            )''')

            # Check if admin already exists
            cursor.execute('SELECT * FROM users WHERE email=?', (ADMIN_EMAIL,))
            if not cursor.fetchone():
                cursor.execute('INSERT INTO users (email, password, role) VALUES (?, ?, ?)',
                               (ADMIN_EMAIL, ADMIN_PASSWORD, 'admin'))
                print("✓ Admin user created successfully")

            conn.commit()
            conn.close()
            print("✓ Database initialized successfully")
        else:
            print("✓ Database already exists — not re-initializing")

    except Exception as e:
        print(f"✗ Database initialization failed: {e}")

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

@app.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'attempts' not in session:
        session['attempts'] = 0
    show_forgot = False

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if admin login
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['user'] = email
            session['role'] = 'admin'
            session['attempts'] = 0
            flash("Admin login successful!", "success")
            return redirect(url_for('admin_dashboard'))

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
            session['role'] = user[3]  # role column
            session['attempts'] = 0
            flash("Login successful!", "success")

            # Redirect based on role
            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('upload'))

        if session['attempts'] >= 2:
            show_forgot = True

    return render_template('login.html', show_forgot=show_forgot)


@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        flash("Admin access required.", "error")
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Get all users
    cursor.execute('SELECT id, email, role FROM users')
    users = cursor.fetchall()

    # Get all uploads with user information
    cursor.execute('''
        SELECT u.id, u.user_email, u.filename, u.file_type, u.prediction_result, u.upload_time 
        FROM uploads u 
        ORDER BY u.upload_time DESC
    ''')
    uploads = cursor.fetchall()

    # Get statistics
    cursor.execute('SELECT COUNT(*) FROM uploads')
    total_uploads = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM uploads WHERE prediction_result = "REAL"')
    real_predictions = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM uploads WHERE prediction_result = "FAKE"')
    fake_predictions = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT user_email) FROM uploads')
    active_users = cursor.fetchone()[0]

    conn.close()

    return render_template('admin_dashboard.html',
                           users=users,
                           uploads=uploads,
                           total_uploads=total_uploads,
                           real_predictions=real_predictions,
                           fake_predictions=fake_predictions,
                           active_users=active_users)


@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if 'user' not in session or session.get('role') != 'admin':
        flash("Admin access required.", "error")
        return redirect(url_for('login'))

    # Prevent admin from deleting themselves
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if user and user[0] == session['user']:
        flash("Cannot delete your own account.", "error")
    else:
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        flash("User deleted successfully.", "success")

    conn.close()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_upload/<int:upload_id>')
def delete_upload(upload_id):
    if 'user' not in session or session.get('role') != 'admin':
        flash("Admin access required.", "error")
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM uploads WHERE id = ?', (upload_id,))
    conn.commit()
    conn.close()

    flash("Upload record deleted successfully.", "success")
    return redirect(url_for('admin_dashboard'))


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

        try:
            cursor.execute('INSERT INTO users (email, password, role) VALUES (?, ?, ?)',
                           (email, password, 'user'))
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

        # Store upload data in database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        file_type = 'image' if filename.lower().endswith(('jpg', 'jpeg', 'png')) else 'video'
        cursor.execute('INSERT INTO uploads (user_email, filename, file_type, prediction_result) VALUES (?, ?, ?, ?)',
                       (session['user'], filename, file_type, result['label']))
        conn.commit()
        conn.close()

        return redirect(url_for('result', prediction=result['label'], filename=filename))

    return render_template('upload.html')


@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    filename = request.args.get('filename')
    video_url = url_for('uploaded_file', filename=filename)

    if prediction.upper() == "FAKE":
        features = ["Unnatural warping", "Blinking issues", "Lip sync mismatch", "Color boundary artifacts",
                    "Pose inconsistency"]
    else:
        features = ["Natural face movement", "Normal blinking", "Synced lip and audio", "Even lighting", "Stable pose"]

    return render_template('result.html', prediction=prediction, filename=filename, video_url=video_url,
                           features=features)


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
    session.pop('role', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))


# ----- API Endpoints for Mobile App -----

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check admin login
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return jsonify({
            'status': 'success',
            'message': 'Admin login successful',
            'email': email,
            'role': 'admin'
        })

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'status': 'error', 'message': 'Email not registered'}), 404
    if user[2] != password:
        return jsonify({'status': 'error', 'message': 'Incorrect password'}), 401

    return jsonify({
        'status': 'success',
        'message': 'Login successful',
        'email': email,
        'role': user[3]
    })


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (email, password, role) VALUES (?, ?, ?)',
                       (email, password, 'user'))
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

    # Store upload data in database for API calls
    user_email = request.headers.get('User-Email')
    if user_email:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        file_type = 'image' if filename.lower().endswith(('jpg', 'jpeg', 'png')) else 'video'
        cursor.execute('INSERT INTO uploads (user_email, filename, file_type, prediction_result) VALUES (?, ?, ?, ?)',
                       (user_email, filename, file_type, result['label']))
        conn.commit()
        conn.close()

    return jsonify({
        'status': 'success',
        'label': result['label'],
        'filename': filename
    })

@app.route('/speak')
def speak():
    """Generate and return speech audio for tutorial instructions in multiple Indian languages"""
    lang = request.args.get('lang', 'en')

    # Text for each language
    texts = {
        "en": "Step 1: Open the web application. Step 2: Click on Login, register, and log in. Step 3: Click on Upload and select a video or image. Step 4: Click on Submit. Step 5: Wait for the result. Step 6: The result will show whether it is Real or Fake.",
        "hi": "चरण 1: वेब एप्लिकेशन खोलें। चरण 2: लॉगिन पर क्लिक करें, पंजीकरण करें और लॉगिन करें। चरण 3: अपलोड पर क्लिक करें और वीडियो या छवि चुनें। चरण 4: सबमिट पर क्लिक करें। चरण 5: परिणाम की प्रतीक्षा करें। चरण 6: परिणाम दिखाएगा कि यह असली है या नकली।",
        "kn": "ಹಂತ 1: ವೆಬ್ ಅಪ್ಲಿಕೇಶನ್ ತೆರೆಯಿರಿ. ಹಂತ 2: ಲಾಗಿನ್ ಕ್ಲಿಕ್ ಮಾಡಿ, ನೋಂದಣಿ ಮಾಡಿ ಮತ್ತು ಲಾಗಿನ್ ಮಾಡಿ. ಹಂತ 3: ಅಪ್‌ಲೋಡ್ ಕ್ಲಿಕ್ ಮಾಡಿ ಮತ್ತು ವಿಡಿಯೋ ಅಥವಾ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ. ಹಂತ 4: ಸಬ್ಮಿಟ್ ಕ್ಲಿಕ್ ಮಾಡಿ. ಹಂತ 5: ಫಲಿತಾಂಶಕ್ಕಾಗಿ ಕಾಯಿರಿ. ಹಂತ 6: ಫಲಿತಾಂಶದಲ್ಲಿ ನಿಜ ಅಥವಾ ನಕಲಿ ಎಂದು ತೋರಿಸಲಾಗುತ್ತದೆ.",
        "ta": "படி 1: வலைப்பயன்பாட்டைத் திறக்கவும். படி 2: உள்நுழைவு கிளிக் செய்யவும், பதிவு செய்யவும் மற்றும் உள்நுழைக. படி 3: பதிவேற்றம் கிளிக் செய்து வீடியோ அல்லது படத்தைத் தேர்ந்தெடுக்கவும். படி 4: சமர்ப்பி கிளிக் செய்யவும். படி 5: முடிவுக்காக காத்திருங்கள். படி 6: இது உண்மையா போலியானதா என்பதைக் காட்டும்.",
        "te": "దశ 1: వెబ్ అప్లికేషన్ తెరవండి. దశ 2: లాగిన్‌పై క్లిక్ చేయండి, నమోదు చేయండి మరియు లాగిన్ చేయండి. దశ 3: అప్‌లోడ్‌పై క్లిక్ చేయండి మరియు వీడియో లేదా చిత్రం ఎంచుకోండి. దశ 4: సమర్పించు క్లిక్ చేయండి. దశ 5: ఫలితానికి వేచి ఉండండి. దశ 6: ఇది నిజమా నకిలీదా అని చూపిస్తుంది.",
        "ml": "പടി 1: വെബ് അപ്ലിക്കേഷൻ തുറക്കുക. പടി 2: ലോഗിൻ ക്ലിക്ക് ചെയ്യുക, രജിസ്റ്റർ ചെയ്യുക, ലോഗിൻ ചെയ്യുക. പടി 3: അപ്‌ലോഡ് ക്ലിക്ക് ചെയ്യുക, വീഡിയോ അല്ലെങ്കിൽ ചിത്രം തിരഞ്ഞെടുക്കുക. പടി 4: സമർപ്പിക്കുക ക്ലിക്ക് ചെയ്യുക. പടി 5: ഫലത്തിനായി കാത്തിരിക്കുക. പടി 6: ഇത് യഥാർത്ഥമാണോ വ്യാജമാണോ എന്ന് കാണിക്കും."
    }

    # Use temporary file for playback
    tts = gTTS(text=texts.get(lang, texts["en"]), lang=lang)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    return send_file(temp_audio.name, mimetype='audio/mpeg')

# ----- Run App -----
if __name__ == '__main__':
    print("🚀 Starting Deepfake Detection Application...")
    init_database()  # Initialize database with admin user
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
