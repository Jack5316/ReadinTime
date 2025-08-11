import os
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile

# Setup frozen imports for bundled application
from services.frozen_imports import setup_frozen_imports
setup_frozen_imports()

# Try to import WhisperX directly
WHISPERX_AVAILABLE = False
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    print("WhisperX successfully imported directly")
except ImportError as e:
    print(f"WhisperX direct import failed: {e}")


class WhisperXService:
    """Service for transcription and alignment using WhisperX."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venv_path = None
        self.whisperx_source_path = None
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup paths for WhisperX environment and source code."""
        # Check if running in frozen executable (PyInstaller bundle)
        if getattr(sys, 'frozen', False):
            # Use bundled Python in frozen executable
            self.python_path = Path(sys.executable)
            self.venv_path = Path(sys.executable).parent
            
            # In frozen environment, WhisperX should be available via direct import
            # or through the bundled source code
            if hasattr(sys, '_MEIPASS'):
                base_path = Path(sys._MEIPASS)
                self.whisperx_source_path = base_path / 'whisperX-3.4.2'
            else:
                self.whisperx_source_path = None
            
            self.logger.info(f"Running in frozen executable, using bundled Python: {self.python_path}")
            if self.whisperx_source_path and self.whisperx_source_path.exists():
                self.logger.info(f"WhisperX source found at: {self.whisperx_source_path}")
        else:
            # Use unified virtual environment in development
            current_dir = Path(__file__).parent.parent
            self.venv_path = current_dir / "venv-unified"
            
            # Platform-specific paths
            if sys.platform == "win32":
                self.python_path = self.venv_path / "Scripts" / "python.exe"
                self.pip_path = self.venv_path / "Scripts" / "pip.exe"
            else:
                self.python_path = self.venv_path / "bin" / "python"
                self.pip_path = self.venv_path / "bin" / "pip"
            
            # Check for source code in development
            self.whisperx_source_path = current_dir / "whisperX-3.4.2"
            if not self.whisperx_source_path.exists():
                self.whisperx_source_path = None
            
            self.logger.info(f"Using venv-unified Python: {self.python_path}")
            self.logger.info(f"Venv-unified path: {self.venv_path}")
            if self.whisperx_source_path:
                self.logger.info(f"WhisperX source path: {self.whisperx_source_path}")
    
    async def setup_environment(self):
        """Setup WhisperX environment if not already configured."""
        try:
            # If WhisperX is already available via direct import, we're good
            if WHISPERX_AVAILABLE:
                self.logger.info("WhisperX available via direct import")
                return True
            
            # In frozen environment, try to set up the import paths
            if getattr(sys, 'frozen', False):
                self.logger.info("Attempting to setup WhisperX in frozen environment")
                
                # Try to import from the bundled source
                if self.whisperx_source_path and self.whisperx_source_path.exists():
                    whisperx_module_path = self.whisperx_source_path / 'whisperx'
                    if whisperx_module_path.exists():
                        # Add the source directory to Python path
                        if str(self.whisperx_source_path) not in sys.path:
                            sys.path.insert(0, str(self.whisperx_source_path))
                            self.logger.info(f"Added WhisperX source to Python path: {self.whisperx_source_path}")
                        
                        # Try importing again
                        try:
                            import whisperx
                            # Update the module-level variable
                            import services.whisperx_service
                            services.whisperx_service.WHISPERX_AVAILABLE = True
                            self.logger.info("WhisperX successfully imported from bundled source")
                            return True
                        except ImportError as e:
                            self.logger.warning(f"Failed to import WhisperX from bundled source: {e}")
                
                # If still not available, fall back to subprocess approach
                self.logger.info("WhisperX not available via direct import, will use subprocess approach")
                return True
            
            # In development environment, set up virtual environment
            # Create virtual environment if it doesn't exist
            if not self.venv_path.exists():
                self.logger.info("Creating WhisperX virtual environment...")
                await self._run_command([
                    sys.executable, "-m", "venv", str(self.venv_path)
                ])
            
            # Check if WhisperX is already installed and working
            try:
                result = await self._run_command([
                    str(self.python_path), "-c", 
                    "import whisperx; print('WhisperX available')"
                ])
                if "WhisperX available" in result.stdout:
                    self.logger.info("WhisperX already available")
                    return True
            except Exception as e:
                self.logger.info(f"WhisperX not available, setting up: {e}")
            
            # Install dependencies
            self.logger.info("Installing WhisperX dependencies...")
            
            # Upgrade pip first
            await self._run_command([
                str(self.pip_path), "install", "--upgrade", "pip"
            ])
            
            # Install PyTorch (CPU version for compatibility)
            await self._run_command([
                str(self.pip_path), "install", "torch", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
            
            # Install WhisperX dependencies from the source
            requirements = [
                "openai-whisper",
                "faster-whisper>=0.10.0",
                "transformers>=4.26.0",
                "pyannote.audio",
                "torch-audio-embeddings", 
                "torchaudio",
                "librosa",
                "soundfile",
                "scipy",
                "scikit-learn",
                "ctranslate2>=3.17.0"
            ]
            
            for req in requirements:
                try:
                    await self._run_command([
                        str(self.pip_path), "install", req
                    ])
                except Exception as e:
                    self.logger.warning(f"Failed to install {req}: {e}")
            
            # Install WhisperX from source
            if self.whisperx_source_path and self.whisperx_source_path.exists():
                self.logger.info("Installing WhisperX from source...")
                await self._run_command([
                    str(self.pip_path), "install", "-e", str(self.whisperx_source_path)
                ])
            else:
                self.logger.info("WhisperX source path not available, using installed version")
            
            self.logger.info("WhisperX environment setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up WhisperX environment: {e}")
            return False
    
    async def _run_command(self, command, cwd=None):
        """Run a command asynchronously."""
        self.logger.debug(f"Running command: {' '.join(command)}")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        stdout, stderr = await process.communicate()
        
        result = type('Result', (), {
            'stdout': stdout.decode('utf-8', errors='ignore'),
            'stderr': stderr.decode('utf-8', errors='ignore'),
            'returncode': process.returncode
        })()
        
        # Log stderr as warning instead of error for non-zero return codes
        if result.stderr:
            self.logger.warning(f"Command stderr: {result.stderr}")
        
        # Only raise exception for actual command failures (non-zero return code)
        # Don't treat stderr output alone as failure
        if result.returncode != 0:
            self.logger.error(f"Command failed with return code {result.returncode}: {' '.join(command)}")
            self.logger.error(f"Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        
        return result
    
    async def transcribe_audio(self, audio_path: str, output_dir: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio using WhisperX with word-level timestamps."""
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # If WhisperX is available via direct import, use it directly
            if WHISPERX_AVAILABLE:
                return await self._transcribe_direct(audio_path, output_dir, language)
            else:
                # Fall back to subprocess approach
                return await self._transcribe_subprocess(audio_path, output_dir, language)
            
        except Exception as e:
            self.logger.error(f"Error in WhisperX transcription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _transcribe_direct(self, audio_path: str, output_dir: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe using direct WhisperX import."""
        try:
            self.logger.info(f"Using direct WhisperX import for transcription")
            
            # Import whisperx functions
            from whisperx import load_audio, load_model, load_align_model, align
            
            # Load audio
            self.logger.info("Loading audio...")
            audio = load_audio(audio_path)
            self.logger.info(f"Audio loaded, shape: {audio.shape if hasattr(audio, 'shape') else len(audio)}")
            
            # Transcribe
            self.logger.info("Loading model...")
            model = load_model("small", device="cpu", compute_type="int8")
            self.logger.info("Model loaded, starting transcription...")
            
            # Handle language parameter - use None for auto-detection instead of "auto"
            transcribe_language = None if language == "auto" else language
            if transcribe_language:
                result = model.transcribe(audio, batch_size=4, language=transcribe_language)
            else:
                result = model.transcribe(audio, batch_size=4)
            self.logger.info("Transcription completed")
            
            # Align if needed
            if result.get("language") in ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]:
                self.logger.info("Loading alignment model...")
                model_a, metadata = load_align_model(language_code=result["language"], device="cpu")
                self.logger.info("Aligning transcription...")
                result = align(result["segments"], model_a, metadata, audio, device="cpu", return_char_alignments=False)
                self.logger.info("Alignment completed")
            
            # Prepare result in expected format
            output_result = {
                "success": True,
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "words": []
            }
            
            # Extract words from segments
            for segment in result.get("segments", []):
                if "words" in segment:
                    for word in segment["words"]:
                        output_result["words"].append({
                            "text": word.get("word", ""),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0),
                            "confidence": word.get("score", 0)
                        })
            
            self.logger.info(f"Extracted {len(output_result['words'])} words from {len(output_result['segments'])} segments")
            
            # Create additional output formats
            await self._create_output_formats(output_result, output_dir)
            
            return {
                "success": True,
                "engine": "whisperx",
                "words": output_result.get("words", []),
                "segments": output_result.get("segments", []),
                "language": output_result.get("language", "unknown"),
                "output_dir": output_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error in direct WhisperX transcription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _transcribe_subprocess(self, audio_path: str, output_dir: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe using subprocess approach (fallback)."""
        try:
            # Create transcription script
            from services.frozen_imports import create_frozen_script_wrapper
            
            script_content = f"""
import whisperx
import json
import sys
import traceback

try:
    # Load audio
    print("Loading audio...")
    audio = whisperx.load_audio(r'{audio_path}')
    print(f"Audio loaded, shape: {{audio.shape if hasattr(audio, 'shape') else len(audio)}}")
    
    # Transcribe
    print("Loading model...")
    model = whisperx.load_model("small", device="cpu", compute_type="int8")
    print("Model loaded, starting transcription...")
    
    # Handle language parameter - use None for auto-detection instead of "auto"
    transcribe_language = None if "{language}" == "auto" else "{language}"
    if transcribe_language:
        result = model.transcribe(audio, batch_size=4, language=transcribe_language)
    else:
        result = model.transcribe(audio, batch_size=4)
    print("Transcription completed")
    
    # Align if needed
    if result.get("language") in ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]:
        print("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
        print("Aligning transcription...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu", return_char_alignments=False)
        print("Alignment completed")
    
    # Prepare result in expected format
    output_result = {{
        "success": True,
        "language": result.get("language", "unknown"),
        "segments": result.get("segments", []),
        "words": []
    }}
    
    # Extract words from segments
    for segment in result.get("segments", []):
        if "words" in segment:
            for word in segment["words"]:
                output_result["words"].append({{
                    "text": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "confidence": word.get("score", 0)
                }})
    
    print(f"Extracted {{len(output_result['words'])}} words from {{len(output_result['segments'])}} segments")
    print("RESULT:", json.dumps(output_result))

except Exception as e:
    print(f"Exception occurred: {{str(e)}}")
    traceback.print_exc()
    error_result = {{
        "success": False,
        "error": str(e)
    }}
    print("RESULT:", json.dumps(error_result))
"""
            
            # Create temporary script file with frozen environment support
            wrapped_script = create_frozen_script_wrapper(script_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(wrapped_script)
                script_path = f.name
            
            try:
                # Run the transcription script
                self.logger.info(f"Running WhisperX transcription on {audio_path}")
                result = await self._run_command([
                    str(self.python_path), script_path
                ])
                
                # Extract the result from stdout
                output_lines = result.stdout.strip().split('\n')
                result_line = None
                for line in output_lines:
                    if line.startswith("RESULT:"):
                        result_line = line[7:].strip()  # Remove "RESULT:" prefix
                        break
                
                if result_line:
                    transcription_result = json.loads(result_line)
                    
                    if transcription_result.get("success"):
                        self.logger.info(f"WhisperX transcription successful, found {len(transcription_result.get('words', []))} words")
                        
                        # Create additional output formats
                        await self._create_output_formats(transcription_result, output_dir)
                        
                        return {
                            "success": True,
                            "engine": "whisperx",
                            "words": transcription_result.get("words", []),
                            "segments": transcription_result.get("segments", []),
                            "language": transcription_result.get("language", "unknown"),
                            "output_dir": output_dir
                        }
                    else:
                        return {"success": False, "error": transcription_result.get("error", "Unknown error")}
                else:
                    self.logger.error("Could not parse transcription result from script output")
                    self.logger.error(f"Script stdout: {result.stdout}")
                    self.logger.error(f"Script stderr: {result.stderr}")
                    return {"success": False, "error": "Could not parse transcription result"}
                
            finally:
                # Clean up temporary script
                try:
                    os.unlink(script_path)
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error in subprocess WhisperX transcription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_output_formats(self, result: Dict[str, Any], output_dir: str):
        """Create additional output formats (SRT, VTT, TSV, TXT)."""
        try:
            words = result["words"]
            
            # Create SRT file
            srt_content = self._create_srt_from_words(words)
            with open(os.path.join(output_dir, "transcription.srt"), 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # Create VTT file
            vtt_content = self._create_vtt_from_words(words)
            with open(os.path.join(output_dir, "transcription.vtt"), 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            
            # Create TSV file
            tsv_content = self._create_tsv_from_words(words)
            with open(os.path.join(output_dir, "transcription.tsv"), 'w', encoding='utf-8') as f:
                f.write(tsv_content)
            
            # Create plain text file
            txt_content = " ".join([word["text"] for word in words if word["text"].strip()])
            with open(os.path.join(output_dir, "transcription.txt"), 'w', encoding='utf-8') as f:
                f.write(txt_content)
                
            self.logger.info("Created additional output formats")
            
        except Exception as e:
            self.logger.warning(f"Error creating output formats: {e}")
    
    def _create_srt_from_words(self, words: List[Dict]) -> str:
        """Create SRT subtitle format from word timestamps."""
        srt_content = []
        current_line = []
        line_start = None
        line_end = None
        subtitle_index = 1
        
        for i, word in enumerate(words):
            if not word["text"].strip():
                continue
                
            if line_start is None:
                line_start = word["start"]
            
            current_line.append(word["text"])
            line_end = word["end"]
            
            # End line on punctuation or every 10 words
            if (word["text"].strip().endswith(('.', '!', '?')) or 
                len(current_line) >= 10 or 
                i == len(words) - 1):
                
                if current_line:
                    start_time = self._format_time_srt(line_start)
                    end_time = self._format_time_srt(line_end)
                    text = " ".join(current_line).strip()
                    
                    srt_content.append(f"{subtitle_index}")
                    srt_content.append(f"{start_time} --> {end_time}")
                    srt_content.append(text)
                    srt_content.append("")
                    
                    subtitle_index += 1
                    current_line = []
                    line_start = None
        
        return "\n".join(srt_content)
    
    def _create_vtt_from_words(self, words: List[Dict]) -> str:
        """Create VTT subtitle format from word timestamps."""
        srt_content = self._create_srt_from_words(words)
        vtt_content = "WEBVTT\n\n" + srt_content.replace(" --> ", " --> ")
        return vtt_content
    
    def _create_tsv_from_words(self, words: List[Dict]) -> str:
        """Create TSV format from word timestamps."""
        tsv_lines = ["start\tend\ttext"]
        for word in words:
            if word["text"].strip():
                tsv_lines.append(f"{word['start']:.3f}\t{word['end']:.3f}\t{word['text']}")
        return "\n".join(tsv_lines)
    
    def _format_time_srt(self, seconds: float) -> str:
        """Format time in SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    
    def group_sentences(self, words: List[Dict], min_words: int = 8) -> List[Dict]:
        """Group words into sentence-like segments for image alignment."""
        segments = []
        current_segment = []
        current_word_count = 0
        
        for word in words:
            if not word["text"].strip():
                continue
                
            current_segment.append(word)
            current_word_count += 1
            
            # End sentence on punctuation or minimum word count
            should_end = (word["text"].strip().endswith((".", "!", "?")) and current_word_count >= min_words) or current_word_count >= min_words * 2
            
            if should_end:
                if current_segment:
                    new_segment = {
                        "text": " ".join([w["text"] for w in current_segment]).strip(),
                        "start": current_segment[0]["start"],
                        "end": current_segment[-1]["end"]
                    }
                    segments.append(new_segment)
                    current_segment = []
                    current_word_count = 0
        
        # Handle remaining words
        if current_segment:
            new_segment = {
                "text": " ".join([w["text"] for w in current_segment]).strip(),
                "start": current_segment[0]["start"],
                "end": current_segment[-1]["end"]
            }
            segments.append(new_segment)
        
        return segments
    
    async def transcribe_and_align(self, audio_path: str, markdown_path: str, output_dir: str) -> Dict[str, Any]:
        """Main method to transcribe audio using WhisperX and create alignments."""
        try:
            # Ensure environment is set up
            env_setup = await self.setup_environment()
            if not env_setup:
                return {"success": False, "error": "Failed to setup WhisperX environment"}
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                return {"success": False, "error": "Audio file not found"}
            
            # Transcribe with WhisperX
            self.logger.info("Starting WhisperX transcription")
            result = await self.transcribe_audio(audio_path, output_dir)
            
            if not result["success"]:
                return result
            
            # Generate image mappings if transcription was successful
            try:
                words = result["words"]
                
                # Group into sentences for image generation
                sentences = self.group_sentences(words)
                
                # Create image mappings
                image_mappings = []
                for i, sentence in enumerate(sentences):
                    image_mappings.append({
                        **sentence,
                        "image": os.path.join(output_dir, "images", f"image_{i}.png")
                    })
                
                # Save text mappings (for frontend compatibility)
                text_mappings_path = os.path.join(output_dir, "text_mappings.json")
                with open(text_mappings_path, 'w', encoding='utf-8') as f:
                    json.dump(image_mappings, f, indent=2)
                
                # Save image mappings (for future use)
                image_mappings_path = os.path.join(output_dir, "image_mappings.json")
                with open(image_mappings_path, 'w', encoding='utf-8') as f:
                    json.dump(image_mappings, f, indent=2)
                
                self.logger.info(f"Created {len(image_mappings)} text/image mappings")
                
                result["image_mappings"] = image_mappings
                
            except Exception as e:
                self.logger.warning(f"Failed to create image mappings: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in WhisperX transcription and alignment: {e}")
            return {"success": False, "error": str(e)}
