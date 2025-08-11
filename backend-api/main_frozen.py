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
from services.pdf_processor import PDFProcessor
from services.tts_service import TTSService
from services.chatterbox_service import ChatterboxTTSService
from services.whisperx_service import WhisperXService
from services.pipeline_manager import PipelineManager
from models.book_models import BookInfo, BookUploadData, Result

# Configure logging for frozen environment
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

    # Replace any existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
except Exception as e:
    # Fallback logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(f"Warning: Using fallback logging configuration due to: {e}")

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="arBooks API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
tts_service = TTSService()
whisperx_service = WhisperXService()
pipeline_manager = PipelineManager()

# Store for tracking active processing jobs
active_jobs = {}

# Create necessary directories
VOICE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "voice_samples")
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)

def create_safe_folder_name(title: str) -> str:
    """Create a safe folder name from a title."""
    import re
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Limit length
    return safe_name[:50]

# Basic health check endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "arBooks API Server", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "frozen": getattr(sys, 'frozen', False),
        "python_executable": sys.executable
    }

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"pong": time.time()}

# Service info endpoints
@app.get("/api/tts/chatterbox/info")
async def get_chatterbox_info():
    """Get Chatterbox TTS service information"""
    try:
        is_available = await tts_service.chatterbox_service.is_available()
        return {
            "available": is_available,
            "status": "Chatterbox TTS service info retrieved"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# Main API endpoints
@app.post("/api/tts/generate")
async def generate_tts_from_text(text: str = Form(...)):
    """Generate TTS from text"""
    try:
        # Create temporary output path
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            output_path = tmp.name
        
        result = await tts_service.generate_speech_from_text(text, output_path)
        
        # Clean up temp file
        try:
            os.unlink(output_path)
        except:
            pass
        
        return result
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/tts/voice-clone")
async def generate_voice_cloned_tts(
    text: str = Form(...),
    audio_prompt_file: UploadFile = File(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5)
):
    """Generate voice-cloned TTS"""
    try:
        # Save uploaded audio file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, audio_prompt_file.filename)
        
        with open(temp_audio_path, "wb") as f:
            content = await audio_prompt_file.read()
            f.write(content)
        
        # Generate voice-cloned TTS
        result = await tts_service.generate_voice_cloned_speech(
            text, temp_audio_path, exaggeration, cfg_weight
        )
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return result
    except Exception as e:
        logger.error(f"Error generating voice-cloned TTS: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/list")
async def list_books(directory_path: str = Form(...)):
    """List books in a directory"""
    try:
        if not os.path.exists(directory_path):
            return Result(success=False, error="Directory not found")
        
        books = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.pdf', '.md', '.txt')):
                file_path = os.path.join(directory_path, filename)
                books.append({
                    "filename": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                })
        
        return Result(success=True, result=books)
    except Exception as e:
        logger.error(f"Error listing books: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/read")
async def read_book(book_path: str = Form(...)):
    """Read a book file"""
    try:
        if not os.path.exists(book_path):
            return Result(success=False, error="Book file not found")
        
        with open(book_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Result(success=True, result={"content": content})
    except Exception as e:
        logger.error(f"Error reading book: {e}")
        return Result(success=False, error=str(e))

@app.get("/api/files/data")
async def get_file_data(file_path: str):
    """Get file data"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"content": content}
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/books/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(...),
    description: str = Form(...),
    book_path: str = Form(...)
):
    """Upload and process a PDF file"""
    try:
        # Create book directory
        safe_title = create_safe_folder_name(title)
        book_dir = os.path.join(book_path, safe_title)
        os.makedirs(book_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(book_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Convert PDF to markdown
        result = await pdf_processor.convert_pdf_to_markdown(file_path, book_dir)
        
        if result["success"]:
            return Result(success=True, result={
                "filePath": file_path,
                "bookDir": book_dir,
                "markdownContent": result.get("content", "")
            })
        else:
            return Result(success=False, error=result.get("error", "PDF conversion failed"))
            
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
        safe_title = create_safe_folder_name(title)
        book_dir = os.path.join(book_path, safe_title)
        file_path = os.path.join(book_dir, filename)
        
        if not os.path.exists(file_path):
            return Result(success=False, error="PDF file not found")
        
        result = await pdf_processor.convert_pdf_to_markdown(file_path, book_dir)
        return Result(success=result["success"], result=result)
        
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/books/generate-tts")
async def generate_tts(
    filename: str = Form(...),
    title: str = Form(...),
    book_path: str = Form(...)
):
    """Generate TTS for a book"""
    try:
        safe_title = create_safe_folder_name(title)
        book_dir = os.path.join(book_path, safe_title)
        markdown_path = os.path.join(book_dir, filename.replace('.pdf', '.md'))
        
        if not os.path.exists(markdown_path):
            return Result(success=False, error="Markdown file not found")
        
        result = await tts_service.generate_speech_from_file(markdown_path, book_dir)
        return Result(success=result["success"], result=result)
        
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
    """Generate voice-cloned TTS for a book"""
    try:
        safe_title = create_safe_folder_name(title)
        book_dir = os.path.join(book_path, safe_title)
        markdown_path = os.path.join(book_dir, filename.replace('.pdf', '.md'))
        
        if not os.path.exists(markdown_path):
            return Result(success=False, error="Markdown file not found")
        
        # Save voice prompt file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, voice_prompt_file.filename)
        
        with open(temp_audio_path, "wb") as f:
            content = await voice_prompt_file.read()
            f.write(content)
        
        result = await tts_service.generate_voice_cloned_speech_from_file(
            markdown_path, book_dir, exaggeration, cfg_weight, temp_audio_path
        )
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return Result(success=result["success"], result=result)
        
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
    """Transcribe audio file"""
    try:
        safe_title = create_safe_folder_name(title)
        book_dir = os.path.join(book_path, safe_title)
        audio_path = os.path.join(book_dir, filename)
        
        if not os.path.exists(audio_path):
            return Result(success=False, error="Audio file not found")
        
        result = await whisperx_service.transcribe_audio(audio_path, book_dir)
        return Result(success=result["success"], result=result)
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/transcribe/whisperx")
async def transcribe_with_whisperx(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
    output_format: str = Form("json")  # json, srt, vtt, tsv, txt
):
    """Transcribe audio using WhisperX"""
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, audio_file.filename)
        
        with open(temp_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Transcribe
        result = await whisperx_service.transcribe_audio(temp_audio_path, temp_dir)
        
        # Clean up
        try:
            os.unlink(temp_audio_path)
            os.rmdir(temp_dir)
        except:
            pass
        
        return Result(success=result["success"], result=result)
        
    except Exception as e:
        logger.error(f"Error transcribing with WhisperX: {e}")
        return Result(success=False, error=str(e))

@app.post("/api/whisperx/setup")
async def setup_whisperx():
    """Setup WhisperX environment"""
    try:
        result = await whisperx_service.setup_environment()
        return Result(success=result, result={"message": "WhisperX setup completed"})
    except Exception as e:
        logger.error(f"Error setting up WhisperX: {e}")
        return Result(success=False, error=str(e))

# Voice sample management
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
        
        # Create safe filename
        safe_filename = f"{sample_id}_{file.filename}"
        file_path = os.path.join(VOICE_SAMPLES_DIR, safe_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Voice sample uploaded: {name} ({safe_filename})")
        
        return Result(success=True, result={
            "filePath": file_path,
            "fileName": safe_filename,
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

# Catch-all route for frontend
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Catch-all route to serve frontend files for client-side routing."""
    try:
        # Skip API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Return basic API info for non-API routes
        return {"message": "arBooks API Server", "status": "running", "endpoint": full_path}
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
        
        # Run the server
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_server() 