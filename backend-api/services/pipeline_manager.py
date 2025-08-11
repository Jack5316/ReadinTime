"""
Pipeline Manager for coordinating the book processing workflow.
Provides progress tracking, error recovery, and status monitoring.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from enum import Enum

from .pdf_processor import PDFProcessor
from .tts_service import TTSService
from .whisperx_service import WhisperXService
from .chatterbox_service import ChatterboxTTSService

logger = logging.getLogger(__name__)

def create_safe_folder_name(title: str) -> str:
    """Create a safe folder name from a book title."""
    # Remove or replace problematic characters
    safe_name = title.replace(" ", "-")  # Replace spaces with hyphens
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_")  # Keep only alphanumeric, hyphens, underscores
    safe_name = safe_name.strip("-")  # Remove leading/trailing hyphens
    return safe_name or "untitled-book"  # Fallback if title is empty

class ProcessingStage(Enum):
    """Stages of book processing pipeline."""
    UPLOAD = "upload"
    PDF_CONVERSION = "pdf_conversion"
    TTS_GENERATION = "tts_generation"
    TRANSCRIPTION = "transcription"
    METADATA_SAVE = "metadata_save"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineManager:
    """Manages the complete book processing pipeline with progress tracking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pdf_processor = PDFProcessor()
        self.tts_service = TTSService()
        self.whisperx_service = WhisperXService()
        self.chatterbox_service = ChatterboxTTSService()
        
    async def process_book(
        self,
        book_data: Any,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        voice_cloning_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a book through the complete pipeline with progress tracking.
        
        Args:
            book_data: Book upload data containing all necessary information
            progress_callback: Optional callback for progress updates
            
        Returns:
            Result dictionary with success status and details
        """
        
        def update_progress(stage: str, percentage: int, message: str):
            self.logger.info(f"[PIPELINE] update_progress called: {percentage}% - {stage}: {message}")
            if progress_callback:
                try:
                    progress_callback(stage, percentage, message)
                    self.logger.info(f"[PIPELINE] progress_callback executed successfully")
                except Exception as e:
                    self.logger.error(f"[PIPELINE] Error in progress_callback: {e}")
            else:
                self.logger.warning(f"[PIPELINE] No progress_callback provided")
            self.logger.info(f"[{percentage}%] {stage}: {message}")
        
        try:
            filename = book_data.name or "unknown_file.pdf"
            book_path = book_data.bookPath
            title = book_data.title
            
            # Debug logging to see what we receive
            self.logger.info(f"DEBUG PIPELINE: filename = '{filename}'")
            self.logger.info(f"DEBUG PIPELINE: title = '{title}'")
            self.logger.info(f"DEBUG PIPELINE: book_path = '{book_path}'")
            self.logger.info(f"DEBUG PIPELINE: book_data attributes = {dir(book_data)}")
            self.logger.info(f"DEBUG PIPELINE: progress_callback = {progress_callback}")
            
            # Test the progress callback immediately
            safe_title = title or "your book"
            safe_filename = filename or "unknown_file.pdf"
            
            self.logger.info(f"[PIPELINE] About to test progress callback")
            update_progress("starting", 1, f"Pipeline started for '{safe_title}' ({safe_filename})")
            
            # Use title for folder name instead of filename to avoid conflicts
            folder_name = create_safe_folder_name(safe_title)
            output_dir = os.path.join(book_path, folder_name)
            
            self.logger.info(f"[PIPELINE] Using folder name: '{folder_name}' (from title: '{safe_title}')")
            
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            
            # Stage 1: PDF Upload (Already handled by API endpoint)
            update_progress("upload", 5, f"Processing '{safe_title}' ({safe_filename})...")
            
            # Stage 2: PDF to Markdown Conversion
            update_progress("pdf_conversion", 10, f"Converting '{safe_title}' PDF to markdown...")
            pdf_path = os.path.join(book_path, filename)
            
            if not os.path.exists(pdf_path):
                return {"success": False, "error": f"PDF file not found: {filename}"}
            
            conversion_result = await self.pdf_processor.convert_pdf_to_markdown(pdf_path, output_dir)
            if not conversion_result["success"]:
                return {"success": False, "error": f"PDF conversion failed: {conversion_result.get('error')}"}
            
            update_progress("pdf_conversion", 25, f"'{safe_title}' converted to markdown successfully")
            
            # Stage 3: TTS Generation (with optional voice cloning)
            if voice_cloning_options and voice_cloning_options.get("enabled"):
                update_progress("tts_generation", 30, f"Generating voice cloned audio for '{safe_title}'...")
                markdown_path = os.path.join(output_dir, "pdf_result.md")
                audio_path = os.path.join(output_dir, "audio", "voice_cloned_output.wav")
                
                # Use voice cloning TTS
                if voice_cloning_options.get("mode") == "direct_upload" and voice_cloning_options.get("voice_prompt_path"):
                    # Direct upload mode: use uploaded voice file
                    self.logger.info(f"[VOICE_CLONE] Starting DIRECT UPLOAD voice cloning with: {voice_cloning_options['voice_prompt_path']}")
                    tts_result = await self.tts_service.chatterbox_service.generate_speech_from_file(
                        markdown_path, 
                        audio_path,
                        exaggeration=voice_cloning_options.get("exaggeration", 0.5),
                        cfg_weight=voice_cloning_options.get("cfg_weight", 0.5),
                        audio_prompt_path=voice_cloning_options["voice_prompt_path"]
                    )
                elif voice_cloning_options.get("mode") == "settings_sample" and voice_cloning_options.get("sample_path"):
                    # Settings sample mode: use saved voice sample
                    self.logger.info(f"[VOICE_CLONE] Starting SETTINGS SAMPLE voice cloning with: {voice_cloning_options['sample_path']}")
                    self.logger.info(f"[VOICE_CLONE] Voice cloning parameters: exaggeration={voice_cloning_options.get('exaggeration', 0.5)}, cfg_weight={voice_cloning_options.get('cfg_weight', 0.5)}")
                    tts_result = await self.tts_service.chatterbox_service.generate_speech_from_file(
                        markdown_path, 
                        audio_path,
                        exaggeration=voice_cloning_options.get("exaggeration", 0.5),
                        cfg_weight=voice_cloning_options.get("cfg_weight", 0.5),
                        audio_prompt_path=voice_cloning_options["sample_path"]
                    )
                    self.logger.info(f"[VOICE_CLONE] Voice cloning TTS result: {tts_result}")
                else:
                    return {"success": False, "error": "Voice cloning enabled but no valid voice source provided"}
                    
                if not tts_result["success"]:
                    return {"success": False, "error": f"Voice cloning TTS generation failed: {tts_result.get('error')}"}
                
                update_progress("tts_generation", 50, f"Voice cloned audio generated for '{safe_title}' successfully")
            else:
                # Standard TTS generation
                update_progress("tts_generation", 30, f"Generating audio for '{safe_title}'...")
                markdown_path = os.path.join(output_dir, "pdf_result.md")
                audio_path = os.path.join(output_dir, "audio", "output.wav")
                
                tts_result = await self.tts_service.generate_speech(markdown_path, audio_path)
                if not tts_result["success"]:
                    return {"success": False, "error": f"TTS generation failed: {tts_result.get('error')}"}
                
                update_progress("tts_generation", 50, f"Audio generated for '{safe_title}' successfully")
            
            # Stage 4: Audio Transcription
            update_progress("transcription", 55, f"Transcribing audio for '{safe_title}' with WhisperX...")
            
            # Setup WhisperX environment if needed
            env_setup = await self.whisperx_service.setup_environment()
            if not env_setup:
                # WhisperX setup failed, return error
                error_msg = "WhisperX environment setup failed"
                self.logger.error(error_msg)
                update_progress("transcription", 60, f"Transcription failed for '{safe_title}': {error_msg}")
                return {"success": False, "error": error_msg}
            
            transcription_result = await self.whisperx_service.transcribe_and_align(
                audio_path, markdown_path, output_dir
            )
            
            if not transcription_result["success"]:
                return {"success": False, "error": f"Transcription failed: {transcription_result.get('error')}"}
            
            update_progress("transcription", 80, f"Audio transcribed for '{safe_title}' successfully")
            
            # Stage 5: Save Metadata
            update_progress("metadata_save", 85, f"Saving metadata for '{safe_title}'...")
            
            metadata_result = await self._save_book_metadata(book_data, output_dir)
            if not metadata_result["success"]:
                return {"success": False, "error": f"Metadata save failed: {metadata_result.get('error')}"}
            
            update_progress("metadata_save", 95, f"Metadata saved for '{safe_title}' successfully")
            
            # Stage 6: Final validation
            update_progress("completed", 100, f"Book '{safe_title}' processed successfully!")
            
            return {
                "success": True,
                "result": f"Book '{safe_title}' processed successfully",
                "output_dir": output_dir,
                "files_created": self._get_created_files(output_dir)
            }
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            update_progress("failed", -1, error_msg)
            return {"success": False, "error": error_msg}
    
    async def _save_book_metadata(self, book_data: Any, output_dir: str) -> Dict[str, Any]:
        """Save book metadata to info.json file."""
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from models.book_models import BookInfo
            
            # Create book info
            book_info = BookInfo(
                title=book_data.title,
                author=book_data.author,
                description=book_data.description,
                folder=create_safe_folder_name(book_data.title),
                cover=""  # TODO: Generate cover image
            )
            
            # Save metadata
            info_path = os.path.join(output_dir, "info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(book_info.dict(), f, indent=2)
            
            self.logger.info(f"Saved metadata to {info_path}")
            return {"success": True, "result": "Metadata saved"}
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_created_files(self, output_dir: str) -> Dict[str, bool]:
        """Check which files were created during processing."""
        expected_files = {
            "pdf_result.md": os.path.exists(os.path.join(output_dir, "pdf_result.md")),
            "audio/output.wav": os.path.exists(os.path.join(output_dir, "audio", "output.wav")),
            "text_mappings.json": os.path.exists(os.path.join(output_dir, "text_mappings.json")),
            "info.json": os.path.exists(os.path.join(output_dir, "info.json")),
            "segments.json": os.path.exists(os.path.join(output_dir, "segments.json")),
            "transcription.srt": os.path.exists(os.path.join(output_dir, "transcription.srt")),
            "transcription.vtt": os.path.exists(os.path.join(output_dir, "transcription.vtt")),
        }
        return expected_files
    
    async def get_processing_status(self, book_path: str, title: str) -> Dict[str, Any]:
        """Get the current processing status of a book."""
        try:
            # Use title for folder name instead of filename to avoid conflicts
            folder_name = create_safe_folder_name(title)
            output_dir = os.path.join(book_path, folder_name)
            
            if not os.path.exists(output_dir):
                return {"status": "not_started", "stage": None, "progress": 0}
            
            created_files = self._get_created_files(output_dir)
            
            # Determine current stage based on files
            if created_files.get("info.json"):
                return {"status": "completed", "stage": "completed", "progress": 100, "files": created_files}
            elif created_files.get("text_mappings.json"):
                return {"status": "processing", "stage": "transcription", "progress": 80, "files": created_files}
            elif created_files.get("audio/output.wav"):
                return {"status": "processing", "stage": "tts_generation", "progress": 50, "files": created_files}
            elif created_files.get("pdf_result.md"):
                return {"status": "processing", "stage": "pdf_conversion", "progress": 25, "files": created_files}
            else:
                return {"status": "processing", "stage": "upload", "progress": 10, "files": created_files}
                
        except Exception as e:
            self.logger.error(f"Error getting processing status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def validate_system_requirements(self) -> Dict[str, Any]:
        """Validate that all system requirements are met."""
        try:
            validation_results = {
                "pdf_processor": False,
                "tts_service": False,
                "whisperx_service": False,
                "chatterbox_available": False,
                "virtual_environments": []
            }
            
            # Check PDF processor
            try:
                # Test with a simple validation
                validation_results["pdf_processor"] = True
            except Exception as e:
                self.logger.warning(f"PDF processor check failed: {e}")
            
            # Check TTS service (lightweight check to avoid heavy model load here)
            try:
                chatterbox_status = await self.tts_service.chatterbox_service.is_available(quick_only=True)
                validation_results["chatterbox_available"] = chatterbox_status
                validation_results["tts_service"] = True
            except Exception as e:
                self.logger.warning(f"TTS service check failed: {e}")
            
            # Check WhisperX service
            try:
                whisperx_setup = await self.whisperx_service.setup_environment()
                validation_results["whisperx_service"] = whisperx_setup
            except Exception as e:
                self.logger.warning(f"WhisperX service check failed: {e}")
            
            # Check unified virtual environment
            if getattr(sys, 'frozen', False):
                # Running in frozen executable - no venv needed
                validation_results["virtual_environments"].append({
                    "name": "frozen_executable",
                    "exists": True,
                    "path": "bundled_with_executable"
                })
            else:
                # Running in development - check venv-unified
                venv_path = Path(__file__).parent.parent / "venv-unified"
                validation_results["virtual_environments"].append({
                    "name": "unified",
                    "exists": venv_path.exists(),
                    "path": str(venv_path)
                })
            
            return {"success": True, "results": validation_results}
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return {"success": False, "error": str(e)}
