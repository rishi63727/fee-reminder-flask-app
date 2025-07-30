import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dateutil.relativedelta import relativedelta
from flask_mail import Mail, Message

# Initialize the Flask application
app = Flask(__name__)

# --- CONFIGURATIONS ---

# Folder to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"xls", "xlsx"}

# Flask-Mail configuration 📧
# IMPORTANT: Use an "App Password" from your Google Account for MAIL_PASSWORD
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'rishisigh2808@gmail.com'  
app.config['MAIL_PASSWORD'] = 'jdwh gxfv vlcv sgci'      
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# --- HELPER FUNCTIONS ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def send_reminder_email(student_email, student_name, due_date, status):
    """Sends a customized reminder email."""
    days_due = (due_date - pd.Timestamp.now()).days
    
    subject = f"Fee Reminder: Payment is {status}"
    body = f"Dear {student_name},\n\n"
    
    if status in ['Overdue', 'Severely Overdue']:
        body += f"This is a reminder that your fee payment is overdue by {-days_due} day(s).\n"
    else: # Due Soon
        body += f"This is a friendly reminder that your fee payment is due in {days_due} day(s).\n"
    
    body += f"Your due date is {due_date.strftime('%d-%B-%Y')}.\n\nPlease ensure you complete the payment soon.\n\nBest regards,\nYour Institution"

    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[student_email])
    msg.body = body
    
    try:
        mail.send(msg)
        print(f"✅ Email sent successfully to {student_email}")
    except Exception as e:
        print(f"❌ Error sending email to {student_email}: {e}")

# --- FLASK ROUTES ---

@app.route("/")
def index():
    """Renders the main upload page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload, processing, and sending reminders."""
    if "file" not in request.files:
        return "No file part in the request.", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected.", 400

    if not allowed_file(file.filename):
        return "Only Excel files (.xls, .xlsx) are allowed.", 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # --- Data processing logic ---
        engine = "openpyxl" if filename.endswith("xlsx") else "xlrd"
        df = pd.read_excel(filepath, engine=engine)
        
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        df['last_payment_date'] = pd.to_datetime(df['last_payment_date'])

        def calculate_due_date(row):
            if row['payment_plan'].lower() == 'annual':
                return row['last_payment_date'] + relativedelta(years=1)
            elif row['payment_plan'].lower() == 'semi-annual':
                return row['last_payment_date'] + relativedelta(months=6)
            return None
        df['due_date'] = df.apply(calculate_due_date, axis=1)

        df['days_until_due'] = (df['due_date'] - pd.Timestamp.now()).dt.days

        def get_status(days):
            if days < -15: return "Severely Overdue"
            elif days < 0: return "Overdue"
            elif days <= 30: return "Due Soon"
            else: return "Paid"
        df['status'] = df['days_until_due'].apply(get_status)

        students_to_notify = df[df['status'].isin(['Due Soon', 'Overdue', 'Severely Overdue'])].copy()

        if students_to_notify.empty:
            return "File processed successfully. No reminders to send today!", 200
            
        # --- Collect names and send emails ---
        sent_to_list = [] 
        for _, student in students_to_notify.iterrows():
            send_reminder_email(
                student_email=student['email'],
                student_name=student['student_name'],
                due_date=student['due_date'],
                status=student['status']
            )
            sent_to_list.append(student['student_name'])

        names_str = ", ".join(sent_to_list)
        response_message = f"Process complete! Sent reminders to {len(sent_to_list)} students: {names_str}"

        return response_message, 200

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return f"An error occurred during processing: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)