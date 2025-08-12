from dotenv import load_dotenv
import os
import io
import logging
import numpy as np
from functools import wraps
from contextlib import contextmanager
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, Response, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from flask_mail import Mail, Message
import uuid
import threading
import re

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Initialize and Configure the App ---
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load configuration from environment variables
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
    UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', 'uploads'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    MAIL_SERVER=os.environ.get('MAIL_SERVER', 'smtp.gmail.com'),
    MAIL_PORT=int(os.environ.get('MAIL_PORT', 465)),
    MAIL_USERNAME=os.environ.get('MAIL_USERNAME'),
    MAIL_PASSWORD=os.environ.get('MAIL_PASSWORD'),
    MAIL_USE_TLS=False,
    MAIL_USE_SSL=True,
    SESSION_TIMEOUT=int(os.environ.get('SESSION_TIMEOUT', 3600))  # 1 hour
)

# Create upload directory
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Initialize Extensions ---
mail = Mail(app)

# ----------------------------
# Schema normalization (shared with training)
# ----------------------------
ALIASES = {
    # features
    "paymentplan": "payment_plan",
    "has_payment_plan": "payment_plan",
    "scholarship": "scholarship",
    "has_scholarship": "scholarship",
    "pastlatepayments": "past_late_payments",
    "previous_late_payments": "past_late_payments",
    # target (tolerated if present)
    "late_payment": "late_payment",
    "waslate": "late_payment",
    "is_late_payment": "late_payment",
    # identifiers/optional
    "studentid": "student_id",
    "student_id": "student_id",
    "studentname": "student_name",
}

def to_snake(name: str) -> str:
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(name))
    return name.strip().lower().replace(" ", "_")

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert headers to snake_case and apply common aliases & value coercions."""
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    df.rename(columns={c: ALIASES.get(c, c) for c in df.columns}, inplace=True)

    # Normalize key feature columns if present
    if "payment_plan" in df.columns:
        df["payment_plan"] = df["payment_plan"].astype(str).str.strip().str.lower()

    if "scholarship" in df.columns:
        df["scholarship"] = (
            df["scholarship"].astype(str).str.strip().str.lower()
            .map({"y": "yes", "yes": "yes", "1": "yes", "true": "yes",
                  "n": "no", "no": "no", "0": "no", "false": "no"})
            .fillna("no")
        )

    if "past_late_payments" in df.columns:
        df["past_late_payments"] = pd.to_numeric(df["past_late_payments"], errors="coerce").fillna(0).astype(int)

    # last_payment_date is needed for due dates
    if "last_payment_date" in df.columns:
        df["last_payment_date"] = pd.to_datetime(df["last_payment_date"], errors="coerce")

    return df

# --- Simple in-memory cache (dev) ---
class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._timers = {}

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value, ttl=3600):
        self._cache[key] = value
        if key in self._timers:
            self._timers[key].cancel()
        timer = threading.Timer(ttl, lambda: self._cache.pop(key, None))
        timer.start()
        self._timers[key] = timer

    def delete(self, key):
        self._cache.pop(key, None)
        if key in self._timers:
            self._timers[key].cancel()
            del self._timers[key]

cache_manager = SimpleCache()

# --- Model Management ---
class ModelManager:
    """Singleton class to manage ML model loading and caching"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fee_prediction_model.pkl')

    def load_model(self):
        if self._model is None:
            try:
                self._model = joblib.load(self._model_path)
                logger.info("Model loaded successfully")
            except FileNotFoundError:
                logger.error(f"Model file '{self._model_path}' not found")
                raise
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        return self._model

    @property
    def model(self):
        return self.load_model()

    @property
    def is_available(self):
        try:
            return self.model is not None
        except Exception:
            return False

model_manager = ModelManager()

# --- Data Validation ---
def validate_dataframe_columns(df: pd.DataFrame):
    """
    Validate that required columns exist in the dataframe.
    We first normalize/alias headers to be tolerant to variations.
    """
    df_norm = normalize_schema(df)
    required_columns = [
        'student_name', 'email', 'last_payment_date',
        'payment_plan', 'scholarship', 'past_late_payments'
    ]
    missing = [c for c in required_columns if c not in df_norm.columns]
    if missing:
        return False, f"Missing required columns: {missing}. Found columns: {list(df_norm.columns)}"
    return True, ""

# --- Data Processing ---
class DataProcessor:
    """Optimized data processing with vectorization and caching"""

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names + values (robust)."""
        return normalize_schema(df)

    @staticmethod
    def calculate_due_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate due dates with proper pandas datetime handling"""
        # df['last_payment_date'] already coerced in normalize_schema
        df['due_date'] = pd.NaT

        # Handle plans
        plan = df['payment_plan'].astype(str).str.lower()
        annual_mask = plan == 'annual'
        semi_annual_mask = plan == 'semi-annual'
        quarterly_mask = plan == 'quarterly'
        monthly_mask = plan == 'monthly'

        if annual_mask.any():
            df.loc[annual_mask, 'due_date'] = df.loc[annual_mask, 'last_payment_date'] + pd.DateOffset(years=1)
        if semi_annual_mask.any():
            df.loc[semi_annual_mask, 'due_date'] = df.loc[semi_annual_mask, 'last_payment_date'] + pd.DateOffset(months=6)
        if quarterly_mask.any():
            df.loc[quarterly_mask, 'due_date'] = df.loc[quarterly_mask, 'last_payment_date'] + pd.DateOffset(months=3)
        if monthly_mask.any():
            df.loc[monthly_mask, 'due_date'] = df.loc[monthly_mask, 'last_payment_date'] + pd.DateOffset(months=1)

        return df

    @staticmethod
    def calculate_status(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized status calculation"""
        current_date = pd.Timestamp.now()
        df['days_until_due'] = (df['due_date'] - current_date).dt.days

        conditions = [
            df['days_until_due'].isna(),
            df['days_until_due'] < 0,
            df['days_until_due'] <= 30
        ]
        choices = ['N/A', 'Overdue', 'Due Soon']
        df['status'] = np.select(conditions, choices, default='Paid')

        return df

    @staticmethod
    def prepare_ml_features(df: pd.DataFrame, model) -> pd.DataFrame:
        """
        Prepare raw features for the pipeline. IMPORTANT:
        Do NOT one-hot here; the saved model pipeline handles it.
        """
        FEATURES = ['payment_plan', 'scholarship', 'past_late_payments']
        X = df[FEATURES].copy()

        # Safety coercions (mirror training)
        X['payment_plan'] = X['payment_plan'].astype(str).str.strip().str.lower()
        X['scholarship'] = (
            X['scholarship'].astype(str).str.strip().str.lower()
            .map({"y": "yes", "yes": "yes", "1": "yes", "true": "yes",
                  "n": "no", "no": "no", "0": "no", "false": "no"})
            .fillna("no")
        )
        X['past_late_payments'] = pd.to_numeric(X['past_late_payments'], errors='coerce').fillna(0).astype(int)

        return X

# --- Utility Functions ---
def assess_risk(probability: float) -> str:
    """Risk assessment based on probability"""
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"

def validate_file(file) -> tuple:
    """Validate uploaded file"""
    if not file or file.filename == '':
        return False, "No file selected"

    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return False, "Invalid file format. Please upload an Excel file"

    return True, ""

# --- Decorators ---
def require_model(f):
    """Decorator to ensure model is available"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not model_manager.is_available:
            return jsonify({"error": "ML model not available. Please contact administrator."}), 503
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    """Decorator for centralized error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    return decorated_function

# --- Routes ---
@app.route("/")
def index():
    """Renders the main page of the application."""
    return render_template("index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/upload", methods=["POST"])
@require_model
@handle_errors
def upload_file():
    """Handle file upload and processing"""
    file = request.files.get("file")

    # Validate file
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        # Load and normalize data
        df_raw = pd.read_excel(file, engine="openpyxl")
        logger.info(f"Loaded DataFrame shape: {df_raw.shape}")
        logger.info(f"Original columns: {list(df_raw.columns)}")

        ok, msg = validate_dataframe_columns(df_raw)
        if not ok:
            return jsonify({"error": msg}), 400

        processor = DataProcessor()
        df = processor.normalize_columns(df_raw)
        df = processor.calculate_due_dates(df)
        df = processor.calculate_status(df)

        # --- ML predictions (no get_dummies; pipeline handles encoding) ---
        try:
            X_predict = processor.prepare_ml_features(df, model_manager.model)
            probabilities_late = model_manager.model.predict_proba(X_predict)[:, 1]
            df['Risk Tier'] = [assess_risk(p) for p in probabilities_late]
        except Exception as e:
            logger.exception("ML prediction error")
            # Fallback to Medium Risk if you want non-fatal behavior
            df['Risk Tier'] = 'Medium Risk'

        # Session & persistence
        session_id = str(uuid.uuid4())
        processed_filename = f"{session_id}.csv"
        processed_filepath = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        # Save processed data (handle NaT formatting safely)
        df_save = df.copy()
        if 'due_date' in df_save.columns:
            dd = df_save['due_date']
            # convert to string safely
            df_save['due_date'] = dd.dt.strftime('%Y-%m-%d %H:%M:%S')
            df_save['due_date'] = df_save['due_date'].fillna('N/A')
        df_save.to_csv(processed_filepath, index=False)

        cache_manager.set(f"session:{session_id}", processed_filepath, ttl=3600)
        session['session_id'] = session_id

        # Prepare response table
        output_df = df[['student_name', 'status', 'Risk Tier']].copy()
        results_list = output_df.to_dict(orient='records')

        # Cache results for export
        import json
        cache_manager.set(f"results:{session_id}", output_df.to_json(orient='split'), ttl=3600)

        logger.info(f"Successfully processed {len(df)} student records")
        return jsonify({"data": results_list}), 200

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/prepare_emails', methods=['POST'])
@handle_errors
def prepare_emails():
    """Prepare and send emails to students"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No session found. Please upload a file first."}), 404

    processed_filepath = cache_manager.get(f"session:{session_id}")
    if not processed_filepath or not os.path.exists(processed_filepath):
        return jsonify({"error": "No analysis data found. Please upload a file first."}), 404

    # Check email config
    if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
        return jsonify({"error": "Email configuration incomplete. Please set MAIL_USERNAME and MAIL_PASSWORD."}), 400

    # Read processed data
    df = pd.read_csv(processed_filepath)
    try:
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    except Exception:
        df['due_date'] = pd.NaT

    sent_emails_log = []
    students_to_notify = df[df['status'] != 'Paid']

    email_templates = {
        "High Risk": "Urgent: Action Required for Your Upcoming Fee Payment",
        "Medium Risk": "Reminder: Your Fee Payment is Approaching",
        "Low Risk": "Friendly Reminder: Upcoming Fee Payment"
    }

    successful_sends = 0
    failed_sends = 0

    for _, student in students_to_notify.iterrows():
        try:
            due_date_str = "N/A" if pd.isna(student.get('due_date')) else student['due_date'].strftime('%d-%B-%Y')
            subject = email_templates.get(student.get('Risk Tier', 'Medium Risk'), 'Fee Payment Reminder')

            # Email body variants
            tier = student.get('Risk Tier', 'Medium Risk')
            if tier == 'High Risk':
                body = f"""Dear {student.get('student_name','Student')},

URGENT: Action Required

This is an urgent reminder that your fee payment is due on {due_date_str}.
Your account has been flagged as high risk for late payment.

Please make your payment immediately to avoid any penalties.

If you have already paid, please disregard this message.

Best regards,
Finance Department
Your Institution"""
            elif tier == 'Low Risk':
                body = f"""Dear {student.get('student_name','Student')},

Friendly Payment Reminder

This is a friendly reminder that your fee payment is due on {due_date_str}.
Please ensure your payment is processed by the due date.

If you have already paid, thank you and please disregard this message.

Best regards,
Finance Department
Your Institution"""
            else:
                body = f"""Dear {student.get('student_name','Student')},

Important Payment Reminder

This is a reminder that your fee payment is due on {due_date_str}.
Please ensure your payment is processed before the due date.

If you have already paid, please disregard this message.

Best regards,
Finance Department
Your Institution"""

            email_details = {
                'recipient': student.get('email'),
                'subject': subject,
                'body': body,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'risk_tier': tier,
                'student_name': student.get('student_name')
            }

            try:
                msg = Message(
                    subject=subject,
                    recipients=[student.get('email')],
                    body=body,
                    sender=app.config['MAIL_USERNAME']
                )
                mail.send(msg)
                email_details['status'] = 'sent'
                successful_sends += 1
                logger.info(f"Email sent to {student.get('email')}")
            except Exception as email_error:
                email_details['status'] = 'failed'
                email_details['error'] = str(email_error)
                failed_sends += 1
                logger.error(f"Failed to send email to {student.get('email')}: {email_error}")

            sent_emails_log.append(email_details)

        except Exception as e:
            logger.error(f"Error processing email for {student.get('email', 'unknown')}: {e}")
            failed_sends += 1
            sent_emails_log.append({
                'recipient': student.get('email', 'unknown'),
                'subject': 'Error',
                'body': 'Error processing email',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'failed',
                'error': str(e),
                'risk_tier': student.get('Risk Tier', 'unknown'),
                'student_name': student.get('student_name', 'unknown')
            })

    # Store email log
    import json
    cache_manager.set(f"emails:{session_id}", json.dumps(sent_emails_log), ttl=3600)

    total_emails = len(students_to_notify)
    if failed_sends == 0:
        message = f"Successfully sent {successful_sends} emails to students."
        status_code = 200
    elif successful_sends == 0:
        message = f"Failed to send all {total_emails} emails. Check email configuration and logs."
        status_code = 500
    else:
        message = f"Sent {successful_sends} emails successfully, {failed_sends} failed. Check logs for details."
        status_code = 207

    logger.info(f"Email sending completed: {successful_sends} successful, {failed_sends} failed")

    return jsonify({
        "message": message,
        "successful_sends": successful_sends,
        "failed_sends": failed_sends,
        "total_students": total_emails,
        "log_url": url_for('mail_log')
    }), status_code

@app.route('/mail_log')
def mail_log():
    """Display email log with send status"""
    session_id = session.get('session_id')
    if not session_id:
        sent_emails = []
        stats = {'total': 0, 'sent': 0, 'failed': 0}
    else:
        sent_emails_str = cache_manager.get(f"emails:{session_id}")
        if sent_emails_str:
            import json
            try:
                sent_emails = json.loads(sent_emails_str)
                stats = {
                    'total': len(sent_emails),
                    'sent': len([e for e in sent_emails if e.get('status') == 'sent']),
                    'failed': len([e for e in sent_emails if e.get('status') == 'failed'])
                }
            except Exception:
                sent_emails = []
                stats = {'total': 0, 'sent': 0, 'failed': 0}
        else:
            sent_emails = []
            stats = {'total': 0, 'sent': 0, 'failed': 0}

    return render_template('mail_log.html', emails=sent_emails, stats=stats)

@app.route('/export')
@handle_errors
def export_file():
    """Export results to Excel"""
    session_id = session.get('session_id')
    if not session_id:
        return "No session found. Please upload a file first.", 404

    results_json = cache_manager.get(f"results:{session_id}")
    if not results_json:
        return "No data available to export. Please upload a file first.", 404

    df = pd.read_json(results_json, orient='split')

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Fee_Risk_Predictions')
    output.seek(0)

    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment;filename=fee_risk_predictions.xlsx"}
    )

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_available": model_manager.is_available,
        "timestamp": datetime.now().isoformat()
    })

# --- Error Handlers ---
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == "__main__":
    # Test model loading on startup
    try:
        model_manager.load_model()
        logger.info("✅ Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")

    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
