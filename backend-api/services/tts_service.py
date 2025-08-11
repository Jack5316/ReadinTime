import os
import logging
import asyncio
import sys
import tempfile
import re
from typing import Dict, Any
from pathlib import Path
from .chatterbox_service import ChatterboxTTSService

class TTSService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venv_path = None
        self.python_path = None
        self.chatterbox_service = ChatterboxTTSService()
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup paths for unified virtual environment."""
        # Check if running in frozen executable (PyInstaller bundle)
        if getattr(sys, 'frozen', False):
            # Use bundled Python in frozen executable
            self.python_path = Path(sys.executable)
            self.venv_path = Path(sys.executable).parent
            self.logger.info(f"Running in frozen executable, using bundled Python: {self.python_path}")
        else:
            # Use unified virtual environment in development
            current_dir = Path(__file__).parent.parent
            self.venv_path = current_dir / "venv-unified"
            
            # Platform-specific paths
            if sys.platform == "win32":
                self.python_path = self.venv_path / "Scripts" / "python.exe"
            else:
                self.python_path = self.venv_path / "bin" / "python"
            
            self.logger.info(f"Unified venv path: {self.venv_path}")
            self.logger.info(f"Python path: {self.python_path}")
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean and prepare text for TTS generation."""
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove headers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code blocks
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.strip()
        
        # No character limit - let chunking handle large texts
        self.logger.info(f"Cleaned text ready for TTS: {len(text)} characters")
        
        return text

    async def generate_speech_from_text(self, text: str, output_path: str) -> Dict[str, Any]:
        """Helper to generate speech directly from raw text."""
        try:
            # Create temp markdown file so we can reuse existing logic
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            try:
                result = await self.generate_speech(tmp_path, output_path)
            finally:
                # Clean up the temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return result
        except Exception as e:
            self.logger.error(f"Error generating speech from text: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_speech(self, markdown_path: str, output_path: str) -> Dict[str, Any]:
        """Main method to generate speech from markdown text using Chatterbox TTS."""
        try:
            self.logger.info(f"Generating TTS for: {markdown_path}")
            
            # Read the markdown file
            with open(markdown_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean text for TTS
            cleaned_text = self._clean_text_for_tts(text)
            
            if not cleaned_text.strip():
                return {"success": False, "error": "No text content found for TTS generation"}
            
            self.logger.info(f"Processing {len(cleaned_text)} characters for TTS")
            
            # Use Chatterbox TTS service
            self.logger.info("Using Chatterbox TTS")
            
            # Delegate availability handling to the chatterbox service to avoid double-checks
            # The service performs a lightweight availability check and loads only once when needed
            result = await self.chatterbox_service.generate_speech(
                text=cleaned_text,
                output_path=output_path
            )
            
            if result["success"]:
                self.logger.info(f"Chatterbox TTS completed successfully")
                return result
            else:
                self.logger.error(f"Chatterbox TTS failed: {result.get('error', 'Unknown error')}")
                return {"success": False, "error": f"Chatterbox TTS failed: {result.get('error', 'Unknown error')}"}
            
        except Exception as e:
            self.logger.error(f"Error in TTS generation: {e}")
            return {"success": False, "error": f"TTS generation failed: {str(e)}"}