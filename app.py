"""
Reliable Document Processing Service - Flask application using stable libraries.
Replaces textract with reliable alternatives: PyMuPDF, pdfplumber, python-docx, openpyxl, python-pptx.
"""

import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from celery import Celery
import redis

# Import our reliable modules with graceful failure handling
try:
    from reliable_extractor import extract_document_text, get_supported_formats
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Document processing not available: {e}")
    DOCUMENT_PROCESSING_AVAILABLE = False

# Import raster detection module
try:
    from pdf_raster_detector import detect_pdf_raster_images, is_raster_detection_available
    RASTER_DETECTION_AVAILABLE = is_raster_detection_available()
except ImportError as e:
    logging.warning(f"Raster detection not available: {e}")
    RASTER_DETECTION_AVAILABLE = False

# Import OCR module
try:
    from ocr_processor import initialize_ocr_processor, is_ocr_available
    OCR_LANGUAGES = os.environ.get('OCR_LANGUAGES', 'eng+rus')
    initialize_ocr_processor(OCR_LANGUAGES)
    OCR_AVAILABLE = is_ocr_available()
except ImportError as e:
    logging.warning(f"OCR processing not available: {e}")
    OCR_AVAILABLE = False

try:
    from redis_manager import redis_manager
    from circuit_breaker import with_circuit_breaker, CircuitBreakerOpenException
    from graceful_shutdown import shutdown_manager, graceful_shutdown_middleware
    from monitoring import metrics_collector, create_monitoring_endpoints
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    # Create minimal fallback objects
    redis_manager = None
    shutdown_manager = None
    metrics_collector = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 52428800))  # 50MB
app.config['UPLOAD_FOLDER'] = '/tmp'

# API Key for authentication
# Raster detection configuration
RASTER_DETECTION_ENABLED = os.environ.get("RASTER_DETECTION_ENABLED", "true").lower() == "true"
DEFAULT_MIN_IMAGE_SIZE = tuple(map(int, os.environ.get("DEFAULT_MIN_IMAGE_SIZE", "100,100").split(",")))
DEFAULT_MAX_IMAGE_SIZE = tuple(map(int, os.environ.get("DEFAULT_MAX_IMAGE_SIZE", "5000,5000").split(",")))
DEFAULT_RATIO_THRESHOLD = float(os.environ.get("DEFAULT_RATIO_THRESHOLD", "0.5"))

# API Key for authentication
API_KEY = os.environ.get('API_KEY', 'default_dev_key')

# Celery configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# Initialize Celery
celery = Celery(
    app.import_name,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app']
)

# Celery configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    beat_schedule={
        'cleanup-temp-files': {
            'task': 'app.cleanup_temp_files',
            'schedule': 3600.0,  # Run every hour
        },
    }
)

def require_api_key(f):
    """Decorator to require API key authentication."""
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@celery.task(bind=True, name='app.process_document')
def process_document(self, file_path, task_id, ocr_enabled=False):
    """Celery task to process document asynchronously."""
    try:
        logger.info(f"Processing document: {file_path}, OCR: {ocr_enabled}")
        
        if not DOCUMENT_PROCESSING_AVAILABLE:
            raise Exception("Document processing not available - missing dependencies")
        
        # Apply circuit breaker if available
        if ENHANCED_FEATURES_AVAILABLE:
            def extract_with_ocr(fp):
                return extract_document_text(fp, ocr_enabled=ocr_enabled)
            extract_func = with_circuit_breaker(extract_with_ocr)
        else:
            extract_func = lambda fp: extract_document_text(fp, ocr_enabled=ocr_enabled)
        
        # Extract text
        result = extract_func(file_path)
        
        # Update metrics if available
        if metrics_collector and hasattr(metrics_collector, 'record_request'):
            metrics_collector.record_request(success=True, response_time=1.0)
        
        logger.info(f"Successfully processed document: {file_path}")
        return {
            'status': 'completed',
            'text': result['text'],
            'metadata': result['metadata'],
            'task_id': task_id
        }
        
    except CircuitBreakerOpenException:
        logger.error(f"Circuit breaker open for document processing: {file_path}")
        return {
            'status': 'failed',
            'error': 'Document processing temporarily unavailable',
            'task_id': task_id
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'task_id': task_id
        }
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup {file_path}: {cleanup_error}")

@celery.task(name='app.cleanup_temp_files')
def cleanup_temp_files():
    """Celery task to clean up old temporary files."""
    try:
        import time
        import glob
        
        temp_dir = '/tmp'
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour ago
        
        pattern = os.path.join(temp_dir, 'doc_*')
        cleaned_count = 0
        
        for file_path in glob.glob(pattern):
            try:
                if os.path.getctime(file_path) < hour_ago:
                    os.remove(file_path)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return {'cleaned_files': cleaned_count, 'timestamp': datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    try:
        # Check Redis connection if available
        redis_status = "unknown"
        if redis_manager:
            try:
                redis_manager.ping()
                redis_status = "healthy"
            except Exception:
                redis_status = "unhealthy"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'document_processing': DOCUMENT_PROCESSING_AVAILABLE,
            'enhanced_features': ENHANCED_FEATURES_AVAILABLE,
            'ocr_available': OCR_AVAILABLE,
            'ocr_languages': OCR_LANGUAGES if OCR_AVAILABLE else None,
            'redis': redis_status,
            'supported_formats': get_supported_formats() if DOCUMENT_PROCESSING_AVAILABLE else []
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/formats', methods=['GET'])
def list_supported_formats():
    """List supported document formats."""
    if not DOCUMENT_PROCESSING_AVAILABLE:
        return jsonify({
            'error': 'Document processing not available',
            'supported_formats': []
        }), 503
    
    return jsonify({
        'supported_formats': get_supported_formats(),
        'extractor': 'reliable_extractor',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/convert', methods=['POST'])
@require_api_key
def convert_document():
    """Convert document to text."""
    try:
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return jsonify({
                'error': 'Document processing not available - missing dependencies'
            }), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check async mode
        async_mode = request.args.get('async', 'false').lower() == 'true'
        
        # Check OCR mode
        ocr_enabled = request.args.get('ocr', 'false').lower() == 'true'
        
        # Validate OCR request
        if ocr_enabled and not OCR_AVAILABLE:
            return jsonify({
                'error': 'OCR not available - Tesseract not installed or pytesseract missing'
            }), 503
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        temp_filename = f"doc_{file_id}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        file.save(file_path)
        logger.info(f"Saved uploaded file: {file_path}")
        
        if async_mode:
            # Process asynchronously
            task = process_document.delay(file_path, file_id, ocr_enabled)
            return jsonify({
                'task_id': task.id,
                'status': 'processing',
                'message': 'Document processing started',
                'ocr_enabled': ocr_enabled
            })
        else:
            # Process synchronously
            try:
                if ENHANCED_FEATURES_AVAILABLE:
                    def extract_with_ocr(fp):
                        return extract_document_text(fp, ocr_enabled=ocr_enabled)
                    extract_func = with_circuit_breaker(extract_with_ocr)
                else:
                    extract_func = lambda fp: extract_document_text(fp, ocr_enabled=ocr_enabled)
                
                result = extract_func(file_path)
                
                # Update metrics if available
                if metrics_collector and hasattr(metrics_collector, 'record_request'):
                    metrics_collector.record_request(success=True, response_time=1.0)
                
                return jsonify({
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'status': 'completed'
                })
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup {file_path}: {cleanup_error}")
    
    except CircuitBreakerOpenException:
        return jsonify({
            'error': 'Document processing temporarily unavailable',
            'status': 'circuit_breaker_open'
        }), 503
    except Exception as e:
        logger.error(f"Error in convert_document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/task/<task_id>', methods=['GET'])
@require_api_key
def get_task_status(task_id):
    """Get status of an async task."""
    try:
        task = process_document.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'task_id': task_id,
                'status': 'pending',
                'message': 'Task is waiting to be processed'
            }
        elif task.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'status': 'processing',
                'message': 'Task is being processed'
            }
        elif task.state == 'SUCCESS':
            result = task.result
            response = {
                'task_id': task_id,
                'status': result.get('status', 'completed'),
                'text': result.get('text', ''),
                'metadata': result.get('metadata', {}),
                'message': 'Task completed successfully'
            }
        else:  # FAILURE
            response = {
                'task_id': task_id,
                'status': 'failed',
                'error': str(task.info),
                'message': 'Task failed'
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return jsonify({
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/cleanup', methods=['POST'])
@require_api_key
def manual_cleanup():
    """Manually trigger cleanup of temporary files."""
    try:
        task = cleanup_temp_files.delay()
        return jsonify({
            "message": "Cleanup task started",
            "task_id": task.id
        })
    except Exception as e:
        logger.error(f"Error starting cleanup: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/detect-raster", methods=["POST"])
@require_api_key
def detect_raster_images():
    """Detect raster images in PDF files."""
    try:
        if not RASTER_DETECTION_AVAILABLE:
            return jsonify({
                "error": "Raster detection not available - missing PyMuPDF"
            }), 503
        
        # Check if file is present
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Check if PDF
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported for raster detection"}), 400
        
        # Check if raster detection is enabled
        if not RASTER_DETECTION_ENABLED:
            return jsonify({"error": "Raster detection is disabled"}), 503
        # Parse settings from query parameters
        settings = {}
        # Apply environment variable defaults
        settings["min_image_size"] = DEFAULT_MIN_IMAGE_SIZE
        settings["max_image_size"] = DEFAULT_MAX_IMAGE_SIZE
        settings["ratio_threshold"] = DEFAULT_RATIO_THRESHOLD
        
        # Image size settings
        min_width = request.args.get("min_width", type=int)
        min_height = request.args.get("min_height", type=int)
        if min_width and min_height:
            settings["min_image_size"] = (min_width, min_height)
        
        max_width = request.args.get("max_width", type=int)
        max_height = request.args.get("max_height", type=int)
        if max_width and max_height:
            settings["max_image_size"] = (max_width, max_height)
        
        # Coverage settings
        check_ratio = request.args.get("check_image_ratio", "true").lower() == "true"
        settings["check_image_ratio"] = check_ratio
        
        ratio_threshold = request.args.get("ratio_threshold", type=float)
        if ratio_threshold is not None:
            settings["ratio_threshold"] = ratio_threshold
        
        # Metadata settings
        include_metadata = request.args.get("include_metadata", "false").lower() == "true"
        settings["include_metadata"] = include_metadata
        
        # Timeout setting
        timeout = request.args.get("timeout", type=int)
        if timeout is not None:
            settings["timeout_seconds"] = timeout
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        temp_filename = f"raster_{file_id}_{filename}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
        
        file.save(file_path)
        logger.info(f"Saved PDF for raster detection: {file_path}")
        
        try:
            # Apply circuit breaker if available
            if ENHANCED_FEATURES_AVAILABLE:
                detect_func = with_circuit_breaker(detect_pdf_raster_images)
            else:
                detect_func = detect_pdf_raster_images
            
            result = detect_func(file_path, settings)
            
            # Metrics not tracked for raster detection currently
            
            return jsonify({
                "status": "completed",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup {file_path}: {cleanup_error}")
    
    except CircuitBreakerOpenException:
        return jsonify({
            "error": "Raster detection temporarily unavailable",
            "status": "circuit_breaker_open"
        }), 503
    except Exception as e:
        logger.error(f"Error in detect_raster_images: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Register monitoring endpoints if available
if ENHANCED_FEATURES_AVAILABLE and metrics_collector:
    create_monitoring_endpoints(app)

# Register graceful shutdown if available
if ENHANCED_FEATURES_AVAILABLE and shutdown_manager:
    graceful_shutdown_middleware(app)

if __name__ == '__main__':
    logger.info("Starting Reliable Document Processing Service")
    logger.info(f"Document processing available: {DOCUMENT_PROCESSING_AVAILABLE}")
    logger.info(f"Enhanced features available: {ENHANCED_FEATURES_AVAILABLE}")
    logger.info(f"Raster detection available: {RASTER_DETECTION_AVAILABLE}")
    logger.info(f"OCR available: {OCR_AVAILABLE}")
    if OCR_AVAILABLE:
        logger.info(f"OCR languages: {OCR_LANGUAGES}")
    if DOCUMENT_PROCESSING_AVAILABLE:
        logger.info(f"Supported formats: {get_supported_formats()}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)