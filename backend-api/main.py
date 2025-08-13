from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json
import shutil
import time
from pathlib import Path
import asyncio
from typing import Optional, Dict, Any
import tempfile
import logging
import sys

# Setup frozen imports for PyInstaller bundle FIRST
if getattr(sys, 'frozen', False):
    # Use the new frozen module loader for better module import handling
    from frozen_module_loader import setup_frozen_module_imports
    setup_frozen_module_imports()
else:
    # Not frozen, but still setup the path-based imports
    from services.frozen_imports import setup_frozen_imports
    setup_frozen_imports()

# Now import the local modules after path setup
from services.pdf_processor import PDFProcessor
from services.tts_service import TTSService
from services.chatterbox_service import ChatterboxTTSService
from services.whisperx_service import WhisperXService
from services.pipeline_manager import PipelineManager
from models.book_models import BookInfo, BookUploadData, Result

# Check for test imports command line argument (for build testing)
if len(sys.argv) > 1 and sys.argv[1] == '--test-imports':
    print("Testing module imports...")
    try:
        # Re-import all critical modules to verify they work
        from models.book_models import BookInfo, BookUploadData, Result
        from services.pdf_processor import PDFProcessor
        from services.tts_service import TTSService
        from services.chatterbox_service import ChatterboxTTSService
        from services.whisperx_service import WhisperXService
        from services.pipeline_manager import PipelineManager
        print("✅ All module imports successful!")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during import test: {e}")
        sys.exit(1)

# Configure logging
# By default, Python's logging.basicConfig() sends *all* log records to
# STDERR.  In the Electron wrapper this caused *every* log line — even
# benign INFO lines — to be captured as an error and prefixed with
# "API Server Error".  We fix that by explicitly routing different log
# levels to different streams:
#   • INFO (and DEBUG)  → STDOUT  ➜ shown as normal output
#   • WARNING & above  → STDERR  ➜ surfaced as real errors
#
# This keeps the console clean and avoids confusing "API Server Error"
# prefixes for non-error messages.

# Fix for PyInstaller unified build logging issues
try:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Formatter shared by both handlers
    _fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Handler for stdout (DEBUG/INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(_fmt)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    # Handler for stderr (WARNING and above)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(_fmt)

    # Replace any existing handlers (e.g., from basicConfig) with ours
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
except Exception as e:
    # Fallback logging configuration for PyInstaller builds
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(f"Warning: Using fallback logging configuration due to: {e}")

# Initialize logger
logger = logging.getLogger(__name__)

def _get_app_config_dir() -> str:
    """Return per-user config dir (Roaming on Windows, ~/.arbooks elsewhere)."""
    if sys.platform == "win32":
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        return os.path.join(app_data, 'arBooks')
    return os.path.join(os.path.expanduser('~'), '.arbooks')


def _get_config_file_path() -> str:
    return os.path.join(_get_app_config_dir(), 'config.json')


def _load_config() -> dict:
    try:
        cfg_path = _get_config_file_path()
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
    return {}


def _save_config(cfg: dict) -> None:
    try:
        cfg_dir = _get_app_config_dir()
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = _get_config_file_path()
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save config: {e}")


# Configurable voice samples directory (env -> saved config -> default)
def get_voice_samples_dir() -> str:
    """Get the voice samples directory, creating it if it doesn't exist.

    Priority:
    1) ARBOOKS_VOICE_SAMPLES_DIR env var
    2) Saved config in user's app config file
    3) Platform default
    """
    env_dir = os.environ.get('ARBOOKS_VOICE_SAMPLES_DIR')
    if env_dir:
        voice_samples_dir = env_dir
    else:
        cfg = _load_config()
        cfg_dir = cfg.get('voice_samples_dir')
        if cfg_dir:
            voice_samples_dir = cfg_dir
        else:
            # Default to user's app data directory
            if sys.platform == "win32":
                app_root = os.environ.get('APPDATA', os.path.expanduser('~'))
                voice_samples_dir = os.path.join(app_root, 'arBooks', 'voice_samples')
            else:
                voice_samples_dir = os.path.join(os.path.expanduser('~'), '.arbooks', 'voice_samples')

    os.makedirs(voice_samples_dir, exist_ok=True)
    return voice_samples_dir

# Initialize voice samples directory
VOICE_SAMPLES_DIR = get_voice_samples_dir()
logger.info(f"Voice samples directory: {VOICE_SAMPLES_DIR}")

app = FastAPI(title="arBooks API", version="1.0.0")

# Add CORS middleware for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
try:
    # Try to mount the frontend dist directory
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "arbooks-desktop", "dist-electron")
    if os.path.exists(frontend_path):
        app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")
        print(f"Frontend assets mounted from: {frontend_path}")
    else:
        print(f"Frontend path not found: {frontend_path}")
except Exception as e:
    print(f"Warning: Could not mount frontend assets: {e}")

# Initialize services
pdf_processor = PDFProcessor()
tts_service = TTSService()
whisperx_service = WhisperXService()
pipeline_manager = PipelineManager()

# Store for tracking active processing jobs
active_jobs: Dict[str, Any] = {}

def create_safe_folder_name(title: str) -> str:
    """Create a safe folder name from a book title."""
    # Remove or replace problematic characters
    safe_name = title.replace(" ", "-")  # Replace spaces with hyphens
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_")  # Keep only alphanumeric, hyphens, underscores
    safe_name = safe_name.strip("-")  # Remove leading/trailing hyphens
    return safe_name or "untitled-book"  # Fallback if title is empty

# -------------------------
#  New TTS raw-text endpoint
# -------------------------
@app.post("/api/tts/generate")
async def generate_tts_from_text(text: str = Form(...)):
    """Generate speech audio directly from raw text.

    Returns JSON with success flag and path to generated audio file. The audio is
    created as a temporary file on the server and should be fetched (or streamed)
    separately by the client using the returned path.
    """
    try:
        # Create a temporary file path for the WAV output
        tmp_dir = tempfile.gettempdir()
        audio_path = os.path.join(tmp_dir, f"tts_output_{hash(text)}.wav")

        result = await tts_service.generate_speech_from_text(text, audio_path)
        if result.get("success"):
            return Result(success=True, result={"audioPath": result["output_path"]})
        else:
            return Result(success=False, error=result.get("error", "Unknown TTS error"))
    except Exception as e:
        logger.error(f"Error generating TTS from raw text: {e}")
        return Result(success=False, error=str(e))

@app.get("/")
async def root():
    """Serve the frontend HTML or return API info if frontend not available."""
    try:
        # Try to serve the frontend index.html
        frontend_path = os.path.join(os.path.dirname(__file__), "..", "arbooks-desktop", "dist-electron", "index.html")
        if os.path.exists(frontend_path):
            with open(frontend_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return {"message": "arBooks API Server", "status": "running", "frontend": "not found"}
    except Exception as e:
        return {"message": "arBooks API Server", "status": "running", "error": str(e)}

@app.get("/health")
async def health_check():
    # Use lightweight Chatterbox check to avoid heavy model loads during health probes
    chatterbox_status = "available" if await tts_service.chatterbox_service.is_available(quick_only=True) else "unavailable"
    
    # Check system requirements
    system_status = await pipeline_manager.validate_system_requirements()
    
    return {
        "status": "healthy", 
        "services": {
            "pdf_processor": "active",
            "tts_service": "active", 
            "chatterbox_tts": chatterbox_status,
            "whisperx_service": "active",
            "pipeline_manager": "active"
        },
        "system_validation": system_status.get("results", {}) if system_status.get("success") else None
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint for quick health checks."""
    return {"status": "ok", "message": "Server is running"}

@app.get("/api/tts/chatterbox/info")
async def get_chatterbox_info():
    """Get detailed information about the Chatterbox TTS model."""
    try:
        model_info = await tts_service.chatterbox_service.get_model_info()
        return {"chatterbox_tts": model_info}
    except Exception as e:
        return {
            "chatterbox_tts": {
                "available": False,
                "error": str(e)
            }
        }

@app.post("/api/tts/voice-clone")
async def generate_voice_cloned_tts(
    text: str = Form(...),
    audio_prompt_file: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5)
):
    """Generate speech with voice cloning from an audio prompt file.
    
    Args:
        text: Text to convert to speech
        audio_prompt_file: Audio file to use as voice reference for cloning
        exaggeration: Emotion exaggeration level (0.0-1.0), default 0.5
        cfg_weight: Classifier-free guidance weight (0.0-1.0), default 0.5
    
    Returns:
        JSON with success flag and path to generated audio file
    """
    try:
        # Validate audio file format
        content_type = audio_prompt_file.content_type or ""
        if not content_type.startswith('audio/'):
            # Allow WAV files even if content-type is not set properly
            filename = audio_prompt_file.filename or ""
            if not filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                return Result(success=False, error="Uploaded file must be an audio file (.wav, .mp3, .flac, .ogg)")
        
        # Save uploaded audio prompt to temporary file, preserve original extension when possible
        tmp_dir = tempfile.gettempdir()
        original_name = (audio_prompt_file.filename or "voice_prompt").lower()
        _, ext = os.path.splitext(original_name)
        if ext not in [
            ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".webm", ".opus"
        ]:
            ext = ".wav"
        audio_prompt_path = os.path.join(tmp_dir, f"voice_prompt_{hash(text)}{ext}")
        
        with open(audio_prompt_path, "wb") as f:
            content = await audio_prompt_file.read()
            f.write(content)
        
        # Create output path for generated speech
        output_path = os.path.join(tmp_dir, f"voice_cloned_tts_{hash(text)}.wav")
        
        # Generate speech with voice cloning
        result = await tts_service.chatterbox_service.generate_speech(
            text=text,
            output_path=output_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=audio_prompt_path
        )
        
        # Clean up temporary audio prompt file
        try:
            os.unlink(audio_prompt_path)
        except:
            pass
        
        if result.get("success"):
            return Result(success=True, result={"audioPath": result["output_path"]})
        else:
            return Result(success=False, error=result.get("error", "Unknown voice cloning error"))
            
    except Exception as e:
        logger.error(f"Error generating voice cloned TTS: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/list")
async def list_books(directory_path: str = Form(...)):
    """List all books in a directory"""
    try:
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        books = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                info_file = os.path.join(item_path, 'info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        book_info = json.load(f)
                        books.append(book_info)
        
        return Result(success=True, result=books)
    except Exception as e:
        logger.error(f"Error listing books: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/read")
async def read_book(book_path: str = Form(...)):
    """Read book data including text mappings and audio file info"""
    try:
        if not os.path.exists(book_path):
            raise HTTPException(status_code=404, detail="Book path not found")
        
        text_mappings_path = os.path.join(book_path, 'text_mappings.json')
        if not os.path.exists(text_mappings_path):
            raise HTTPException(status_code=404, detail="Text mappings not found")
        
        with open(text_mappings_path, 'r', encoding='utf-8') as f:
            text_mappings = json.load(f)
        
        # Check which audio file exists
        audio_dir = os.path.join(book_path, 'audio')
        regular_audio_path = os.path.join(audio_dir, 'output.wav')
        voice_cloned_audio_path = os.path.join(audio_dir, 'voice_cloned_output.wav')
        
        audio_file = None
        if os.path.exists(voice_cloned_audio_path):
            audio_file = 'voice_cloned_output.wav'
        elif os.path.exists(regular_audio_path):
            audio_file = 'output.wav'
        
        return Result(success=True, result={
            "textMappings": text_mappings,
            "audioFile": audio_file
        })
    except Exception as e:
        logger.error(f"Error reading book: {e}")
        return Result(success=False, error=str(e))

@app.get("/api/files/data")
async def get_file_data(file_path: str):
    """Get file data (for images, audio, etc.)"""
    try:
        logger.info(f"Requesting file data for: {file_path}")
        
        # Check if file path is valid
        if not file_path or file_path.strip() == "":
            logger.error("Empty file path provided")
            raise HTTPException(status_code=400, detail="File path is required")
        
        # Normalize the path
        normalized_path = os.path.normpath(file_path)
        logger.info(f"Normalized path: {normalized_path}")
        
        if not os.path.exists(normalized_path):
            logger.error(f"File not found: {normalized_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.isfile(normalized_path):
            logger.error(f"Path is not a file: {normalized_path}")
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        logger.info(f"Successfully serving file: {normalized_path}")
        return FileResponse(normalized_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file data: {e}")
        logger.error(f"File path was: {file_path}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/books/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(...),
    description: str = Form(...),
    book_path: str = Form(...)
):
    """Upload PDF file"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create filename and path
        filename = file.filename.replace(" ", "-")
        pdf_path = os.path.join(book_path, filename)
        
        # Save the uploaded file
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Uploaded {filename} to {pdf_path}")
        return Result(success=True, result=f"Uploaded {filename} successfully")
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/convert-pdf")
async def convert_pdf_to_markdown(
    filename: str = Form(...),
    title: str = Form(...),
    book_path: str = Form(...)
):
    """Convert PDF to markdown"""
    try:
        pdf_path = os.path.join(book_path, filename)
        # Use title for folder name instead of filename to avoid conflicts
        folder_name = create_safe_folder_name(title)
        output_dir = os.path.join(book_path, folder_name)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # Convert PDF to markdown
        result = await pdf_processor.convert_pdf_to_markdown(pdf_path, output_dir)
        
        if result["success"]:
            return Result(success=True, result=f"Converted {filename} successfully")
        else:
            return Result(success=False, error=result["error"])
    
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/generate-tts")
async def generate_tts(
    filename: str = Form(...),
    title: str = Form(...),
    book_path: str = Form(...)
):
    """Generate TTS audio from markdown"""
    try:
        # Use title for folder name instead of filename to avoid conflicts
        folder_name = create_safe_folder_name(title)
        output_dir = os.path.join(book_path, folder_name)
        markdown_path = os.path.join(output_dir, "pdf_result.md")
        audio_path = os.path.join(output_dir, "audio", "output.wav")
        
        # Generate TTS
        result = await tts_service.generate_speech(markdown_path, audio_path)
        
        if result["success"]:
            return Result(success=True, result=f"Generated TTS for {filename} successfully")
        else:
            return Result(success=False, error=result["error"])
    
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/generate-voice-cloned-tts")
async def generate_voice_cloned_tts_for_book(
    filename: str = Form(...),
    title: str = Form(...),
    book_path: str = Form(...),
    voice_prompt_file: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5)
):
    """Generate voice-cloned TTS audio from markdown"""
    try:
        # Use title for folder name instead of filename to avoid conflicts
        folder_name = create_safe_folder_name(title)
        output_dir = os.path.join(book_path, folder_name)
        markdown_path = os.path.join(output_dir, "pdf_result.md")
        audio_path = os.path.join(output_dir, "audio", "voice_cloned_output.wav")
        
        # Save voice prompt file temporarily
        temp_voice_path = os.path.join(output_dir, "temp_voice_prompt.wav")
        with open(temp_voice_path, "wb") as f:
            content = await voice_prompt_file.read()
            f.write(content)
        
        # Generate voice-cloned TTS
        result = await tts_service.chatterbox_service.generate_speech_from_file(
            markdown_path, 
            audio_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=temp_voice_path
        )
        
        # Clean up temp file
        if os.path.exists(temp_voice_path):
            os.unlink(temp_voice_path)
        
        if result["success"]:
            return Result(success=True, result=f"Generated voice-cloned TTS for {filename} successfully")
        else:
            return Result(success=False, error=result["error"])
    
    except Exception as e:
        logger.error(f"Error generating voice-cloned TTS: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/transcribe-audio")
async def transcribe_audio(
    filename: str = Form(...),
    title: str = Form(...),
    book_path: str = Form(...),
    engine: str = Form("whisperx")  # Default to WhisperX
):
    """Transcribe audio with WhisperX"""
    try:
        # Use title for folder name instead of filename to avoid conflicts
        folder_name = create_safe_folder_name(title)
        output_dir = os.path.join(book_path, folder_name)
        audio_path = os.path.join(output_dir, "audio", "output.wav")
        markdown_path = os.path.join(output_dir, "pdf_result.md")
        
        # Setup WhisperX environment if needed
        env_setup = await whisperx_service.setup_environment()
        if not env_setup:
            return Result(success=False, error="WhisperX environment setup failed")
        
        # Transcribe audio
        result = await whisperx_service.transcribe_and_align(audio_path, markdown_path, output_dir)
        
        if result["success"]:
            return Result(success=True, result=f"Transcribed audio for {filename} successfully")
        else:
            return Result(success=False, error=result["error"])
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/transcribe/whisperx")
async def transcribe_with_whisperx(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
    output_format: str = Form("json")  # json, srt, vtt, tsv, txt
):
    """Direct WhisperX transcription endpoint for any audio file"""
    try:
        # Create temporary directory for this transcription
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded audio file
            audio_path = os.path.join(temp_dir, audio_file.filename)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            
            # Transcribe with WhisperX
            result = await whisperx_service.transcribe_audio(audio_path, temp_dir, language)
            
            if not result["success"]:
                return Result(success=False, error=result["error"])
            
            # Return requested format
            if output_format.lower() == "json":
                return Result(success=True, result={
                    "words": result["words"],
                    "segments": result["segments"],
                    "language": result["language"],
                    "engine": "whisperx"
                })
            elif output_format.lower() == "srt":
                srt_path = os.path.join(temp_dir, "transcription.srt")
                if os.path.exists(srt_path):
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        return JSONResponse({"success": True, "result": f.read(), "format": "srt"})
            elif output_format.lower() == "vtt":
                vtt_path = os.path.join(temp_dir, "transcription.vtt")
                if os.path.exists(vtt_path):
                    with open(vtt_path, 'r', encoding='utf-8') as f:
                        return JSONResponse({"success": True, "result": f.read(), "format": "vtt"})
            elif output_format.lower() == "tsv":
                tsv_path = os.path.join(temp_dir, "transcription.tsv")
                if os.path.exists(tsv_path):
                    with open(tsv_path, 'r', encoding='utf-8') as f:
                        return JSONResponse({"success": True, "result": f.read(), "format": "tsv"})
            elif output_format.lower() == "txt":
                txt_path = os.path.join(temp_dir, "transcription.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        return JSONResponse({"success": True, "result": f.read(), "format": "txt"})
            
            # Default to JSON if file not found
            return Result(success=True, result={
                "words": result["words"],
                "segments": result["segments"],
                "language": result["language"],
                "engine": "whisperx"
            })
    
    except Exception as e:
        logger.error(f"Error in WhisperX transcription: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/whisperx/setup")
async def setup_whisperx():
    """Setup WhisperX environment"""
    try:
        result = await whisperx_service.setup_environment()
        if result:
            return Result(success=True, result="WhisperX environment setup completed successfully")
        else:
            return Result(success=False, error="Failed to setup WhisperX environment")
    except Exception as e:
        logger.error(f"Error setting up WhisperX: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/save-metadata")
async def save_metadata(
    filename: str = Form(...),
    title: str = Form(...),
    author: str = Form(...),
    description: str = Form(...),
    book_path: str = Form(...)
):
    """Save book metadata to info.json file"""
    try:
        # Use title for folder name instead of filename to avoid conflicts
        folder_name = create_safe_folder_name(title)
        output_dir = os.path.join(book_path, folder_name)
        
        # Create book info
        book_info = BookInfo(
            title=title,
            author=author,
            description=description,
            folder=folder_name,
            cover=""
        )
        
        # Save metadata
        info_path = os.path.join(output_dir, "info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(book_info.dict(), f, indent=2)
        
        logger.info(f"{filename} has been processed successfully!")
        return Result(success=True, result="Saved metadata")
    
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/process-complete")
async def process_book_complete(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(...),
    description: str = Form(...),
    book_path: str = Form(...),
    voice_cloning_enabled: str = Form("false"),
    voice_sample_id: str = Form(None),
    voice_cloning_mode: str = Form("none"),  # "settings_sample", "direct_upload", "none"
    exaggeration: str = Form("0.5"),
    cfg_weight: str = Form("0.5"),
    voice_prompt_file: UploadFile = File(None)
):
    """Process a complete book through the entire pipeline with progress tracking"""
    try:
        # Debug logging to see what we receive
        logger.info(f"DEBUG: Received file.filename: '{file.filename}'")
        logger.info(f"DEBUG: Received title: '{title}'")
        logger.info(f"DEBUG: Received author: '{author}'")
        logger.info(f"DEBUG: Received description: '{description[:50]}...'")
        logger.info(f"DEBUG: Received book_path: '{book_path}'")
        logger.info(f"🎵 DEBUG Voice Cloning Params:")
        logger.info(f"  - voice_cloning_enabled: '{voice_cloning_enabled}'")
        logger.info(f"  - voice_cloning_mode: '{voice_cloning_mode}'")
        logger.info(f"  - voice_sample_id: '{voice_sample_id}'")
        logger.info(f"  - exaggeration: '{exaggeration}'")
        logger.info(f"  - cfg_weight: '{cfg_weight}'")
        logger.info(f"  - voice_prompt_file: {voice_prompt_file.filename if voice_prompt_file else 'None'}")
        
        # Check if filename exists
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
            
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create job ID for tracking
        safe_filename = file.filename or "unknown_file.pdf"
        job_id = f"{safe_filename}_{int(time.time())}"
        
        logger.info(f"DEBUG: Generated job_id: '{job_id}'")
        
        # Save the uploaded file
        filename = safe_filename.replace(" ", "-")
        pdf_path = os.path.join(book_path, filename)
        
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create book data object
        book_data = BookUploadData(
            name=filename,
            pdfData=None,  # Already saved to disk
            title=title,
            author=author,
            description=description,
            bookPath=book_path
        )
        
        # Start processing in background
        async def process_with_progress():
            # Initialize progress tracking with safe values
            safe_title = title or "your book"
            safe_filename = filename or "unknown_file.pdf"
            
            logger.info(f"[{job_id}] Starting background processing for '{safe_title}' ({safe_filename})")
            
            active_jobs[job_id] = {
                "stage": "starting", 
                "progress": 0, 
                "message": f"Started processing '{safe_title}' ({safe_filename})",
                "status": "processing",
                "filename": filename,
                "title": title
            }
            
            logger.info(f"[{job_id}] Initial job status set: {active_jobs[job_id]}")
            
            def update_progress(stage: str, percentage: int, message: str):
                logger.info(f"[{job_id}] Progress update called: {percentage}% - {stage}: {message}")
                active_jobs[job_id] = {
                    "stage": stage,
                    "progress": percentage,
                    "message": message,
                    "status": "processing" if percentage < 100 else "completed",
                    "filename": filename,
                    "title": title
                }
                logger.info(f"[{job_id}] Job status updated: {active_jobs[job_id]}")
            
            try:
                logger.info(f"[{job_id}] About to call pipeline_manager.process_book")
                
                # Add a small delay to allow frontend to start polling
                await asyncio.sleep(1)
                
                # Test progress callback immediately
                update_progress("initializing", 5, f"Initializing processing for '{safe_title}'...")
                
                # Prepare voice cloning options
                voice_cloning_opts = None
                if voice_cloning_mode != "none":
                    voice_cloning_opts = {
                        "enabled": True,
                        "mode": voice_cloning_mode,
                        "exaggeration": float(exaggeration),
                        "cfg_weight": float(cfg_weight)
                    }
                    
                    if voice_cloning_mode == "direct_upload" and voice_prompt_file:
                        # Save uploaded voice file to temporary location
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        voice_prompt_path = os.path.join(temp_dir, f"voice_prompt_{job_id}.wav")
                        with open(voice_prompt_path, "wb") as f:
                            content = await voice_prompt_file.read()
                            f.write(content)
                        voice_cloning_opts["voice_prompt_path"] = voice_prompt_path
                        
                    elif voice_cloning_mode == "settings_sample" and voice_sample_id:
                        # Find the voice sample file
                        import glob
                        sample_pattern = os.path.join(VOICE_SAMPLES_DIR, f"{voice_sample_id}_*")
                        matching_files = glob.glob(sample_pattern)
                        if matching_files:
                            voice_cloning_opts["sample_path"] = matching_files[0]
                            logger.info(f"Using voice sample: {matching_files[0]}")
                        else:
                            error_msg = f"Voice sample not found for ID: {voice_sample_id} in {VOICE_SAMPLES_DIR}"
                            logger.error(error_msg)
                            return Result(success=False, error=error_msg)

                result = await pipeline_manager.process_book(book_data, update_progress, voice_cloning_opts)
                
                logger.info(f"[{job_id}] Pipeline processing completed with result: {result}")
                
                if result["success"]:
                    active_jobs[job_id] = {
                        "stage": "completed",
                        "progress": 100,
                        "message": result["result"],
                        "status": "completed",
                        "filename": filename,
                        "title": title,
                        "files_created": result.get("files_created", {})
                    }
                    logger.info(f"[{job_id}] Processing completed successfully")
                else:
                    active_jobs[job_id] = {
                        "stage": "failed",
                        "progress": -1,
                        "message": result["error"],
                        "status": "failed",
                        "filename": filename,
                        "title": title
                    }
                    logger.error(f"[{job_id}] Processing failed: {result['error']}")
                    
            except Exception as e:
                error_message = f"Processing failed for '{title}': {str(e)}"
                logger.error(f"[{job_id}] Exception in background processing: {error_message}", exc_info=True)
                active_jobs[job_id] = {
                    "stage": "failed",
                    "progress": -1,
                    "message": error_message,
                    "status": "failed",
                    "filename": filename,
                    "title": title
                }
        
        # Start background processing
        task = asyncio.create_task(process_with_progress())
        logger.info(f"[{job_id}] Background task created: {task}")
        
        # Store the task reference to prevent garbage collection
        active_jobs[job_id + "_task"] = task
        
        # Log the job creation for debugging
        logger.info(f"Created processing job {job_id} for '{title}' ({filename})")

        return Result(success=True, result={
                "job_id": job_id,
                "message": f"Started processing '{title}' ({filename})",
                "filename": filename,
                "title": title
            })
        
    except Exception as e:
        logger.error(f"Error starting book processing: {e}")
        # Provide a fallback filename for error logging
        error_filename = getattr(file, 'filename', 'unknown_file') if file else 'unknown_file'
        return Result(success=False, error=f"Failed to start processing {error_filename}: {str(e)}")

@app.get("/api/books/processing-status/{job_id}")
async def get_processing_status(job_id: str):
    """Get the processing status of a book by job ID"""
    try:
        logger.debug(f"[{job_id}] Status request received. Available jobs: {list(active_jobs.keys())}")
        
        if job_id not in active_jobs:
            logger.warning(f"Job ID '{job_id}' not found. Available jobs: {list(active_jobs.keys())}")
            return Result(success=False, error="Job not found")
        
        job_status = active_jobs[job_id]
        logger.debug(f"[{job_id}] Returning status: {job_status.get('progress', 0)}% - {job_status.get('stage', 'unknown')} - {job_status.get('message', 'No message')}")
        
        return Result(success=True, result=job_status)
    
    except Exception as e:
        logger.error(f"Error getting processing status for job {job_id}: {e}")
        return Result(success=False, error=str(e))

@app.get("/api/books/active-jobs")
async def get_active_jobs():
    """Get all currently active jobs for debugging"""
    try:
        # Filter out task references
        jobs = {k: v for k, v in active_jobs.items() if not k.endswith("_task")}
        logger.info(f"Active jobs requested. Current jobs: {list(jobs.keys())}")
        return Result(success=True, result={"active_jobs": jobs, "total_count": len(jobs)})
    
    except Exception as e:
        logger.error(f"Error getting active jobs: {e}")
        return Result(success=False, error=str(e))

@app.get("/api/books/validate-system")
async def validate_system_requirements():
    """Validate that all system components are properly configured"""
    try:
        validation_result = await pipeline_manager.validate_system_requirements()
        return Result(success=True, result=validation_result["results"])
    
    except Exception as e:
        logger.error(f"Error validating system: {e}")
        return Result(success=False, error=str(e))

# -------------------------
#  Voice Sample Management
# -------------------------

# Create voice samples directory if it doesn't exist
# VOICE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")
# os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

@app.get("/api/voice-samples/directory")
async def get_voice_samples_directory():
    """Get the current voice samples directory"""
    try:
        return Result(success=True, result={"directory": VOICE_SAMPLES_DIR})
    except Exception as e:
        logger.error(f"Error getting voice samples directory: {e}")
        return Result(success=False, error=str(e))

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"pong": True}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "voice_samples_dir": VOICE_SAMPLES_DIR,
            "voice_samples_dir_exists": os.path.exists(VOICE_SAMPLES_DIR),
            "voice_samples_dir_is_dir": os.path.isdir(VOICE_SAMPLES_DIR) if os.path.exists(VOICE_SAMPLES_DIR) else False,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/api/voice-samples")
async def list_voice_samples():
    """List all available voice samples"""
    try:
        logger.info(f"Listing voice samples from directory: {VOICE_SAMPLES_DIR}")
        
        # Check if the directory exists and is accessible
        if not os.path.exists(VOICE_SAMPLES_DIR):
            logger.info(f"Voice samples directory does not exist: {VOICE_SAMPLES_DIR}")
            return Result(success=True, result=[])
        
        if not os.path.isdir(VOICE_SAMPLES_DIR):
            logger.warning(f"Voice samples path is not a directory: {VOICE_SAMPLES_DIR}")
            return Result(success=True, result=[])
        
        try:
            files = os.listdir(VOICE_SAMPLES_DIR)
            logger.info(f"Found {len(files)} files in voice samples directory")
        except PermissionError:
            logger.error(f"Permission denied accessing voice samples directory: {VOICE_SAMPLES_DIR}")
            return Result(success=False, error="Permission denied accessing voice samples directory")
        except Exception as e:
            logger.error(f"Error listing voice samples directory: {e}")
            return Result(success=False, error=f"Error listing voice samples directory: {str(e)}")
        
        voice_samples = []
        for filename in files:
            if filename.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                file_path = os.path.join(VOICE_SAMPLES_DIR, filename)
                try:
                    # Extract sample_id and original filename from the stored filename
                    # Format: {unique_sample_id}_{original_filename}
                    # The unique_sample_id is like "sample_1755020669574" and contains the actual unique ID
                    if filename.startswith('sample_'):
                        # Handle the case where the filename format is "sample_timestamp_originalname.extension"
                        # We need to find where the timestamp ends and the original name begins
                        # The format is: sample_timestamp_originalname.extension
                        parts = filename.split('_', 2)  # Split into max 3 parts
                        if len(parts) >= 3:
                            # parts[0] = "sample"
                            # parts[1] = timestamp (e.g., "1755020669574")
                            # parts[2] = original filename with extension
                            unique_id = f"{parts[0]}_{parts[1]}"  # "sample_timestamp"
                            original_filename = parts[2]
                            name = original_filename.rsplit('.', 1)[0]  # Remove extension
                            
                            # Get file stats
                            stat = os.stat(file_path)
                            
                            voice_samples.append({
                                "id": unique_id,  # Use the full unique ID like "sample_1755020669574"
                                "name": name,  # Use the actual name entered by user
                                "fileName": filename,
                                "filePath": file_path,
                                "fileSize": stat.st_size,
                                "uploadTime": stat.st_mtime
                            })
                            logger.debug(f"Processed voice sample: {unique_id} -> {name}")
                        else:
                            logger.warning(f"Skipping voice sample file with invalid format: {filename}")
                    else:
                        # Handle other formats if they exist
                        parts = filename.split('_', 1)
                        if len(parts) == 2:
                            sample_id = parts[0]
                            original_filename = parts[1]
                            
                            # Get file stats
                            stat = os.stat(file_path)
                            
                            voice_samples.append({
                                "id": sample_id,
                                "name": original_filename.rsplit('.', 1)[0],  # Remove extension for display name
                                "fileName": filename,
                                "fileSize": stat.st_size,
                                "filePath": file_path,
                                "uploadTime": stat.st_mtime
                            })
                            logger.debug(f"Processed voice sample: {sample_id} -> {original_filename}")
                        else:
                            logger.warning(f"Skipping voice sample file with invalid format: {filename}")
                except Exception as e:
                    logger.warning(f"Error processing voice sample file {filename}: {e}")
                    continue
        
        # Sort by upload time (newest first)
        voice_samples.sort(key=lambda x: x.get("uploadTime", 0), reverse=True)
        
        logger.info(f"Successfully processed {len(voice_samples)} voice samples")
        return Result(success=True, result=voice_samples)
        
    except Exception as e:
        logger.error(f"Error listing voice samples: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/voice-samples/directory")
async def set_voice_samples_directory(directory: str = Form(...)):
    """Set the voice samples directory"""
    try:
        global VOICE_SAMPLES_DIR
        
        # Validate the directory
        if not os.path.exists(directory):
            # Try to create it
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                return Result(success=False, error=f"Cannot create directory: {str(e)}")
        
        # Check if it's writable
        test_file = os.path.join(directory, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.unlink(test_file)
        except Exception as e:
            return Result(success=False, error=f"Directory is not writable: {str(e)}")
        
        # Update the global variable and persist to config
        VOICE_SAMPLES_DIR = directory
        try:
            cfg = _load_config()
            cfg['voice_samples_dir'] = VOICE_SAMPLES_DIR
            _save_config(cfg)
        except Exception as e:
            logger.warning(f"Failed to persist voice samples dir: {e}")

        logger.info(f"Voice samples directory changed to: {VOICE_SAMPLES_DIR}")
        return Result(success=True, result={"directory": VOICE_SAMPLES_DIR})
    except Exception as e:
        logger.error(f"Error setting voice samples directory: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/voice-samples/upload")
async def upload_voice_sample(
    file: UploadFile = File(...),
    name: str = Form(...),
    sample_id: str = Form(...)
):
    """Upload and save a voice sample file"""
    try:
        # Validate audio file format
        content_type = file.content_type or ""
        if not content_type.startswith('audio/'):
            filename = file.filename or ""
            if not filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                return Result(success=False, error="Uploaded file must be an audio file")
        
        # Create safe filename with unique ID
        # Use the sample_id (which should be unique like "sample_1755020669574")
        # and append the original filename to preserve the extension
        safe_filename = f"{sample_id}_{file.filename}"
        file_path = os.path.join(VOICE_SAMPLES_DIR, safe_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Voice sample uploaded: {name} ({safe_filename})")
        
        return Result(success=True, result={
            "id": sample_id,
            "name": name,
            "fileName": safe_filename,
            "filePath": file_path,
            "fileSize": len(content)
        })
        
    except Exception as e:
        logger.error(f"Error uploading voice sample: {e}")
        return Result(success=False, error=str(e))

@app.delete("/api/voice-samples/{sample_id}")
async def delete_voice_sample(sample_id: str):
    """Delete a voice sample file"""
    try:
        # Find file with sample_id prefix
        for filename in os.listdir(VOICE_SAMPLES_DIR):
            if filename.startswith(f"{sample_id}_"):
                file_path = os.path.join(VOICE_SAMPLES_DIR, filename)
                os.unlink(file_path)
                logger.info(f"Voice sample deleted: {filename}")
                return Result(success=True, result=f"Voice sample {sample_id} deleted")
        
        return Result(success=False, error=f"Voice sample {sample_id} not found")
        
    except Exception as e:
        logger.error(f"Error deleting voice sample {sample_id}: {e}")
        return Result(success=False, error=str(e))

@app.get("/api/voice-samples/{sample_id}/file")
async def get_voice_sample_file(sample_id: str):
    """Get voice sample file for playback"""
    try:
        # Find file with sample_id prefix
        for filename in os.listdir(VOICE_SAMPLES_DIR):
            if filename.startswith(f"{sample_id}_"):
                file_path = os.path.join(VOICE_SAMPLES_DIR, filename)
                if os.path.exists(file_path):
                    return FileResponse(file_path)
        
        raise HTTPException(status_code=404, detail=f"Voice sample {sample_id} not found")
        
    except Exception as e:
        logger.error(f"Error getting voice sample file {sample_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Catch-all route to serve frontend files for client-side routing."""
    try:
        # Skip API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Try to serve the frontend index.html for any non-API route
        frontend_path = os.path.join(os.path.dirname(__file__), "..", "arbooks-desktop", "dist-electron", "index.html")
        if os.path.exists(frontend_path):
            with open(frontend_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            raise HTTPException(status_code=404, detail="Frontend not found")
    except HTTPException:
        raise
    except Exception as e:
        return {"message": "arBooks API Server", "status": "running", "error": str(e)}

def run_server():
    """Run the server with proper configuration for frozen environment"""
    try:
        # Configure uvicorn for frozen environment
        import uvicorn.logging
        try:
            uvicorn.logging.LOGGING_CONFIG = None
        except Exception as e:
            print(f"Warning: Could not configure uvicorn logging: {e}")
        
        # Run the server - use direct app import for frozen compatibility
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server() 