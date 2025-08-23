import os
import logging
import asyncio
import subprocess
import sys
import tempfile
import re
from typing import Dict, Any, Optional
from pathlib import Path
import json
import gc
import weakref
import time
import textwrap

# Note: torch/torchaudio imports are handled within subprocess execution
# to avoid dependency conflicts with the main environment


class ChatterboxTTSService:
    """Service for generating speech using Chatterbox TTS model via unified venv."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venv_path = None
        self.python_path = None
        self._cached_model = None
        self._model_lock = asyncio.Lock()
        # Availability check throttling to avoid repeated heavy loads
        self._last_check_time: float = 0.0
        self._last_check_result: Optional[bool] = None
        # If a previous availability check failed, do not attempt another heavy load
        # for this many seconds (default 5 minutes). This prevents repeated crashes
        # from health probes that call this endpoint frequently.
        self._availability_cooldown_seconds: int = 300
        # Inference/device controls (GPU-first by default, fallback to CPU)
        self._inference_device = os.getenv('CHB_TTS_DEVICE', 'cuda').lower()
        if self._inference_device not in ('cpu', 'cuda'):
            self._inference_device = 'cuda'
        
        # Check if CUDA is actually available, fallback to CPU if not
        if self._inference_device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA requested but not available, falling back to CPU")
                    self._inference_device = 'cpu'
                else:
                    self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            except ImportError:
                self.logger.warning("PyTorch not available, falling back to CPU")
                self._inference_device = 'cpu'
        
        # If using CPU, hide GPUs from all child processes to avoid CUDA init errors
        if self._inference_device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.logger.info("Using CPU for TTS generation")
        # Optionally lower audio sample rate to reduce memory/disk usage (defaults to 22050 on CPU)
        env_sr = int(os.getenv('CHB_TTS_SR', '0')) if os.getenv('CHB_TTS_SR') else 0
        self._target_sample_rate = env_sr or (22050 if self._inference_device == 'cpu' else None)
        # Release strategy: process-per-chunk (dev) is memory-safe; cached (frozen) for speed
        self._release_strategy = os.getenv('CHB_TTS_RELEASE', 'process_per_chunk')
        self._setup_paths()
    
    def _check_system_memory(self) -> bool:
        """Check if system has sufficient memory for TTS processing"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent
            
            self.logger.info(f"System memory: {available_gb:.1f} GB available, {used_percent:.1f}% used")
            
            # Require at least 2GB available and less than 90% used
            if available_gb < 2.0 or used_percent > 90:
                self.logger.warning(f"Insufficient memory: {available_gb:.1f} GB available, {used_percent:.1f}% used")
                return False
            return True
        except Exception as e:
            self.logger.warning(f"Could not check system memory: {e}")
            return True  # Assume OK if we can't check
    
    def _setup_paths(self):
        """Setup paths for unified virtual environment."""
        # Check if running in frozen executable (PyInstaller bundle)
        if getattr(sys, 'frozen', False):
            # Use bundled Python in frozen executable
            self.python_path = Path(sys.executable)
            self.venv_path = Path(sys.executable).parent
            self.logger.info(f"Running in frozen executable, using bundled Python: {self.python_path}")
            self.is_frozen = True
            try:        
                import chatterbox.tts  # type: ignore[reportMissingImports]
                self.logger.info("Chatterbox TTS imported successfully in frozen environment")
            except ImportError as e:
                self.logger.error("PyInstaller bundle is missing chatterbox.tts")
                raise
        else:
            # Use unified virtual environment in development
            current_dir = Path(__file__).parent.parent
            self.venv_path = current_dir / "venv-unified"
            self.is_frozen = False
            
            # Platform-specific paths
            if sys.platform == "win32":
                self.python_path = self.venv_path / "Scripts" / "python.exe"
            else:
                self.python_path = self.venv_path / "bin" / "python"
            
            # Verify the paths exist
            if not self.venv_path.exists():
                self.logger.error(f"Unified venv path does not exist: {self.venv_path}")
                raise FileNotFoundError(f"venv-unified directory not found at {self.venv_path}")
            
            if not self.python_path.exists():
                self.logger.error(f"Python executable not found: {self.python_path}")
                raise FileNotFoundError(f"Python executable not found at {self.python_path}")
            
            self.logger.info(f"Using venv-unified Python: {self.python_path}")
            self.logger.info(f"Unified venv path: {self.venv_path}")
    
    async def _get_cached_model(self):
        """Get or create a cached model instance to prevent repeated loading."""
        async with self._model_lock:
            if self._cached_model is None:
                self.logger.info("Loading Chatterbox TTS model for caching...")
                
                if self.is_frozen:
                    # Load model directly in frozen environment
                    try:
                        import warnings
                        import os
                        
                        warnings.filterwarnings("ignore")
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                        
                        from chatterbox.tts import ChatterboxTTS  # type: ignore[reportMissingImports]
                        
                        # Force garbage collection before loading
                        gc.collect()
                        
                        self._cached_model = ChatterboxTTS.from_pretrained(device=self._inference_device)
                        self.logger.info("Chatterbox TTS model cached successfully")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load model for caching: {e}")
                        self._cached_model = None
                        raise
                else:
                    # For development environment, we'll still use subprocess but with better caching
                    self.logger.info("Model caching not available in development mode, using subprocess approach")
                    self._cached_model = "subprocess_mode"
            
            return self._cached_model
    
    async def _clear_cached_model(self):
        """Clear the cached model to free memory."""
        async with self._model_lock:
            if self._cached_model is not None and self._cached_model != "subprocess_mode":
                try:
                    # Clear any model-specific caches
                    if hasattr(self._cached_model, 'clear_cache'):
                        self._cached_model.clear_cache()
                    
                    # Delete the model
                    del self._cached_model
                    self._cached_model = None
                    
                    # Force garbage collection
                    gc.collect()
                    
                    self.logger.info("Cached model cleared successfully")
                except Exception as e:
                    self.logger.warning(f"Error clearing cached model: {e}")
            elif self._cached_model == "subprocess_mode":
                self._cached_model = None
    
    async def _run_command(self, command, cwd=None, timeout=120):
        """Run a command asynchronously with timeout."""
        try:
            if cwd:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            
            # Add timeout to prevent hanging
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.error(f"Command timed out after {timeout} seconds")
                process.kill()
                await process.wait()
                return type('CommandResult', (), {
                    'returncode': 124,  # Standard timeout exit code
                    'stdout': '',
                    'stderr': f'Command timed out after {timeout} seconds'
                })()
            
            # Handle Windows encoding issues
            def safe_decode(data):
                if not data:
                    return ''
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        return data.decode('cp1252')  # Windows-1252
                    except UnicodeDecodeError:
                        return data.decode('utf-8', errors='replace')
            
            return type('CommandResult', (), {
                'returncode': process.returncode,
                'stdout': safe_decode(stdout),
                'stderr': safe_decode(stderr)
            })()
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return type('CommandResult', (), {
                'returncode': 1,
                'stdout': '',
                'stderr': str(e)
            })()
    
    async def is_available(self, force: bool = False, quick_only: bool = True) -> bool:
        """Check if Chatterbox TTS is available.

        Args:
            force: If True, bypass cooldown/backoff and perform a fresh check.
            quick_only: If True, perform only a lightweight import check without loading models.
                This avoids heavy memory usage and double-loading when generation will load the model anyway.
        """
        try:
            # If we already have a cached model (frozen mode), it's available
            if self._cached_model is not None and self._cached_model != "subprocess_mode":
                self._last_check_time = time.time()
                self._last_check_result = True
                return True

            # Cooldown after failures to avoid repeated heavy loading from health checks
            now = time.time()
            if (
                not force
                and self._last_check_result is False
                and (now - self._last_check_time) < self._availability_cooldown_seconds
            ):
                self.logger.warning(
                    "Skipping Chatterbox availability check due to recent failure (cooldown active)"
                )
                return False

            if self.is_frozen:
                # In frozen environment, test imports directly
                try:
                    from chatterbox.tts import ChatterboxTTS  # type: ignore[reportMissingImports]
                    self.logger.info("ChatterboxTTS imported successfully in frozen environment")
                    
                    # For quick checks, do not attempt to load the model
                    if quick_only:
                        self._last_check_time = time.time()
                        self._last_check_result = True
                        return True

                    # Test model loading in frozen environment (full check)
                    import warnings
                    import gc
                    import os
                    
                    warnings.filterwarnings("ignore")
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    
                    gc.collect()
                    
                    try:
                        self.logger.info("Testing model loading in frozen environment...")
                        
                        # Use cached model if available, otherwise test loading
                        cached_model = await self._get_cached_model()
                        if cached_model and cached_model != "subprocess_mode":
                            self.logger.info("Chatterbox TTS model available (cached)")
                            self._last_check_time = time.time()
                            self._last_check_result = True
                            return True
                        else:
                            # Test loading without caching
                            model = ChatterboxTTS.from_pretrained(device=self._inference_device)
                            del model
                            gc.collect()
                            self.logger.info("Chatterbox TTS model loaded successfully in frozen environment")
                            self._last_check_time = time.time()
                            self._last_check_result = True
                            return True
                    except MemoryError as e:
                        self.logger.warning(f"Chatterbox TTS failed: Insufficient memory to load model: {e}")
                        self._last_check_time = time.time()
                        self._last_check_result = False
                        return False
                    except Exception as e:
                        self.logger.warning(f"Chatterbox TTS model loading failed in frozen environment: {e}")
                        import traceback
                        self.logger.warning(f"Full traceback: {traceback.format_exc()}")
                        self._last_check_time = time.time()
                        self._last_check_result = False
                        return False
                        
                except ImportError as e:
                    self.logger.warning(f"Chatterbox imports failed in frozen environment: {e}")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
            else:
                # Use subprocess approach for development environment
                # First check if basic imports work
                # Set working directory to backend-api to ensure chatterbox/src is in Python path
                current_dir = Path(__file__).parent.parent
                result = await self._run_command([
                    str(self.python_path), "-c", 
                    "import chatterbox.tts; from chatterbox.tts import ChatterboxTTS; print('CHATTERBOX_IMPORTS_OK')"
                ], cwd=str(current_dir))
                
                if result.returncode != 0 or "CHATTERBOX_IMPORTS_OK" not in result.stdout:
                    self.logger.warning("Chatterbox imports failed")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                
                # For quick checks, do NOT attempt to load the model. Let generation load it once.
                if quick_only:
                    self._last_check_time = time.time()
                    self._last_check_result = True
                    return True

                # Then check if we can actually load the model (this might fail due to memory) — full check
                model_test = await self._run_command([
                    str(self.python_path), "-c", 
                    """
import warnings
import gc
import os
import sys
warnings.filterwarnings("ignore")

# Set environment variables to help with memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Try to free up memory first
gc.collect()

try:
    print("Starting model loading test...")
    from chatterbox.tts import ChatterboxTTS
    print("ChatterboxTTS imported successfully")
    
    # Try loading with minimal memory footprint
    print("Attempting to load model with CPU device and minimal memory...")
    
    # Force garbage collection before loading
    gc.collect()
    
    # Load model with CPU device (should use less memory)
            model = ChatterboxTTS.from_pretrained(device=self._inference_device)
    print('CHATTERBOX_MODEL_OK')
    
    # Clean up immediately
    del model
    gc.collect()
    
except MemoryError as e:
    print(f'CHATTERBOX_MEMORY_ERROR: {e}')
except OSError as e:
    error_str = str(e).lower()
    if 'os error 1455' in error_str or '页面文件太小' in error_str or 'paging file' in error_str:
        print('CHATTERBOX_MEMORY_ERROR: Windows virtual memory insufficient')
    elif 'access' in error_str or 'permission' in error_str:
        print(f'CHATTERBOX_ACCESS_ERROR: {e}')
    else:
        print(f'CHATTERBOX_MODEL_ERROR: {e}')
except RuntimeError as e:
    if 'out of memory' in str(e).lower() or 'memory' in str(e).lower():
        print(f'CHATTERBOX_MEMORY_ERROR: {e}')
    else:
        print(f'CHATTERBOX_RUNTIME_ERROR: {e}')
except Exception as e:
    print(f'CHATTERBOX_MODEL_ERROR: {e}')
finally:
    # Force cleanup
    gc.collect()
"""
                ], cwd=str(current_dir), timeout=120)
                
                if "CHATTERBOX_MODEL_OK" in model_test.stdout:
                    self.logger.info("Chatterbox TTS model loaded successfully")
                    self._last_check_time = time.time()
                    self._last_check_result = True
                    return True
                elif "CHATTERBOX_MEMORY_ERROR" in model_test.stdout:
                    self.logger.warning("Chatterbox TTS failed: Insufficient memory to load model")
                    self.logger.warning("Consider increasing Windows virtual memory (pagefile) size")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                elif "CHATTERBOX_ACCESS_ERROR" in model_test.stdout:
                    self.logger.warning("Chatterbox TTS failed: Access violation or permission error")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                elif "CHATTERBOX_RUNTIME_ERROR" in model_test.stdout:
                    self.logger.warning("Chatterbox TTS failed: Runtime error during model loading")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                elif model_test.returncode != 0:
                    self.logger.warning(f"Chatterbox model test failed with exit code {model_test.returncode}")
                    self.logger.warning(f"Process crashed or was terminated. This usually indicates insufficient memory.")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                else:
                    self.logger.warning(f"Chatterbox model test failed: {model_test.stdout}")
                    self._last_check_time = time.time()
                    self._last_check_result = False
                    return False
                    
        except Exception as e:
            self.logger.warning(f"Chatterbox availability test failed: {e}")
            self._last_check_time = time.time()
            self._last_check_result = False
            return False
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean and prepare text for TTS generation."""
        if not text.strip():
            return "You need to add some text for me to talk."
        
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove headers
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code blocks
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.strip()
        
        return text
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 200) -> list:
        """
        Split text into smaller chunks suitable for TTS processing.
        Chatterbox TTS works better with shorter text segments.
        Reduced default chunk size to 200 for better stability and lower memory usage.
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit, save current chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            
            # If a single sentence is too long, split it by words
            if len(current_chunk) > max_chunk_size:
                words = current_chunk.split()
                if len(chunks) > 0 or len(words) > 1:  # Don't split if it's the only content
                    chunks.append(" ".join(words[:len(words)//2]))
                    current_chunk = " ".join(words[len(words)//2:])
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks and merge them
        filtered_chunks = []
        temp_chunk = ""
        
        for chunk in chunks:
            if len(chunk) < 20 and temp_chunk:  # Merge very short chunks (reduced from 30 to 20)
                temp_chunk += " " + chunk
            elif len(chunk) < 20:
                temp_chunk = chunk
            else:
                if temp_chunk:
                    filtered_chunks.append(temp_chunk)
                    temp_chunk = ""
                filtered_chunks.append(chunk)
        
        if temp_chunk:
            filtered_chunks.append(temp_chunk)
        
        return filtered_chunks if filtered_chunks else [text]
    
    async def generate_speech(
        self, 
        text: str, 
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate speech from text using Chatterbox TTS.
        
        Args:
            text: Text to convert to speech
            output_path: Path where to save the audio file
            exaggeration: Emotion exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            audio_prompt_path: Optional path to audio file for voice cloning (if None, uses default voice)
            
        Returns:
            Dictionary with success status and details
        """
        # Use lightweight availability check first to avoid repeated heavy loads
        if not await self.is_available(quick_only=True):
            return {
                "success": False, 
                "error": "Chatterbox TTS model not available"
            }
        
        # Check system memory before processing
        if not self._check_system_memory():
            return {
                "success": False,
                "error": "Insufficient system memory for TTS processing. Please close other applications and try again."
            }
        
        try:
            self.logger.info(f"[VOICE_CLONE] Starting voice cloning generation")
            self.logger.info(f"[VOICE_CLONE] Text length: {len(text)} chars")
            self.logger.info(f"[VOICE_CLONE] Output path: {output_path}")
            self.logger.info(f"[VOICE_CLONE] Audio prompt: {audio_prompt_path}")
            self.logger.info(f"[VOICE_CLONE] Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")
            
            # Clean and prepare text
            self.logger.info(f"[VOICE_CLONE] Cleaning text...")
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text.strip():
                self.logger.error(f"[VOICE_CLONE] No text after cleaning!")
                return {
                    "success": False, 
                    "error": "No usable text after cleaning"
                }
            
            # Split text into manageable chunks
            # Use smaller chunks for voice cloning to reduce memory pressure; allow env override
            try:
                if audio_prompt_path:
                    max_chunk_size = int(os.getenv('CHB_TTS_CHUNK_VC', '160'))
                else:
                    max_chunk_size = int(os.getenv('CHB_TTS_CHUNK', '300'))
            except Exception:
                max_chunk_size = 160 if audio_prompt_path else 300
            text_chunks = self._split_text_into_chunks(cleaned_text, max_chunk_size=max_chunk_size)
            self.logger.info(f"Processing {len(text_chunks)} text chunks (max {max_chunk_size}) for {len(cleaned_text)} characters")
            
            # If only one chunk, process normally
            if len(text_chunks) == 1:
                return await self._generate_single_chunk(
                    text_chunks[0], output_path, exaggeration, cfg_weight, audio_prompt_path
                )
            
            # If multiple chunks, use the multi-chunk generation approach
            else:
                # Calculate dynamic timeout based on book size
                # Voice cloning takes much longer than regular TTS
                if audio_prompt_path:
                    # Voice cloning: ~5 minutes per chunk
                    estimated_minutes = len(text_chunks) * 5
                    timeout_seconds = max(1800, estimated_minutes * 60)  # Minimum 30 minutes for voice cloning
                else:
                    # Regular TTS: ~2 minutes per chunk
                    estimated_minutes = len(text_chunks) * 2
                    timeout_seconds = max(600, estimated_minutes * 60)  # Minimum 10 minutes for regular TTS
                
                self.logger.info(f"Large book detected: {len(text_chunks)} chunks, estimated {estimated_minutes} minutes")
                self.logger.info(f"Using dynamic timeout: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
                
                return await self._generate_and_combine_chunks(
                    text_chunks, output_path, exaggeration, cfg_weight, audio_prompt_path, timeout_seconds
                )
                
        except Exception as e:
            self.logger.error(f"Error generating speech with Chatterbox TTS: {e}")
            return {
                "success": False,
                "error": f"Chatterbox TTS generation failed: {str(e)}"
            }
    
    async def _generate_single_chunk(
        self, 
        text: str, 
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate speech for a single text chunk."""
        try:
            self.logger.info(f"Generating speech for single chunk: {len(text)} characters")
            
            # Check memory before processing
            if not self._check_system_memory():
                return {
                    "success": False,
                    "error": "Insufficient system memory for TTS processing. Please close other applications and try again."
                }
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if self.is_frozen:
                # Use direct import approach for frozen environment
                return await self._generate_single_chunk_frozen(
                    text, output_path, exaggeration, cfg_weight, audio_prompt_path
                )
            else:
                # Use subprocess approach for development environment
                # Set working directory to backend-api to ensure chatterbox/src is in Python path
                current_dir = Path(__file__).parent.parent
            
            # Write text to temporary file to avoid command line length limits
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as text_file:
                text_file.write(text)
                text_file_path = text_file.name
            
            # Prefer calling the CLI which already honors Data/models.json and local/bundled models
            try:
                timeout_env = os.getenv('CHB_TTS_TIMEOUT_SECONDS')
                if timeout_env and timeout_env.isdigit():
                    timeout = int(timeout_env)
                else:
                    timeout = 1800 if (audio_prompt_path is not None) else 900

                cli_path = Path(__file__).parent.parent / "main_cli.py"
                # Ensure HF caches go to Data/chatterbox_models as well
                try:
                    data_dir = (current_dir.parent / "Data").resolve()
                    ch_models = (data_dir / "chatterbox_models").resolve()
                    ch_models.mkdir(parents=True, exist_ok=True)
                    os.environ.setdefault("HF_HOME", str(ch_models))
                    os.environ.setdefault("TRANSFORMERS_CACHE", str(ch_models))
                    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ch_models))
                except Exception:
                    pass

                cmd = [
                    str(self.python_path),
                    str(cli_path),
                    "--text-file", text_file_path,
                    "--out", output_path,
                    "--device", "cpu",
                    "--exaggeration", str(exaggeration),
                    "--cfg-weight", str(cfg_weight),
                ]
                if audio_prompt_path:
                    cmd.extend(["--prompt", str(audio_prompt_path)])

                result_cli = await self._run_command(cmd, cwd=str(current_dir), timeout=int(timeout))

                if result_cli.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    self.logger.info(f"Chatterbox TTS (CLI) completed successfully: {file_size} bytes")
                    return {
                        "success": True,
                        "engine": "Chatterbox TTS (CLI)",
                        "output_path": output_path,
                        "file_size": file_size,
                    }
                else:
                    self.logger.info("CLI path did not succeed; falling back to embedded script execution")
            except Exception as _cli_exc:
                self.logger.info(f"CLI execution path failed, falling back to embedded script: {_cli_exc}")

            # Create TTS generation script that reads from the text file
            # Force CPU device in dev mode to avoid CUDA issues
            device = 'cpu'
            # Ensure subprocess cannot see any CUDA devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            target_sr = self._target_sample_rate or 0

            tts_script_raw = f'''
import os
# Set CPU/GPU env BEFORE importing torch/torchaudio for stability
if {json.dumps(device)} == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Reduce verbose output and disable progress bars; force UTF-8 IO
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Aggressive memory management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import warnings
import gc
import json
import tempfile
import psutil
import torch
import wave
import numpy as np
HAS_TORCHAUDIO = False
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception as e:
    print("torchaudio import failed; continuing without it: " + str(e))

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    # Reconfigure stdio to handle non-ASCII safely
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
try:
    import tqdm as _tqdm
    class _NoTQDM:
        def __init__(self, it=None, *a, **k):
            self.it = it
        def __iter__(self):
            return iter(self.it) if self.it is not None else iter(())
        def update(self, *a, **k):
            return None
        def close(self):
            return None
    _tqdm.tqdm = _NoTQDM  # type: ignore
except Exception:
    pass

def log_memory_usage(stage):
    """Log current memory usage with more detail"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        print(f"[{{stage}}] Memory: {{memory_info.rss / 1024 / 1024:.1f}} MB RSS, {{memory_info.vms / 1024 / 1024:.1f}} MB VMS, {{memory_percent:.1f}}%")
        
        # Log system memory if available
        try:
            system_memory = psutil.virtual_memory()
            print(f"[{{stage}}] System: {{system_memory.available / 1024 / 1024 / 1024:.1f}} GB available, {{system_memory.percent:.1f}}% used")
            
            # Check if we're running out of memory
            if system_memory.percent > 95 or system_memory.available < 1.0 * (1024**3):  # Less than 1GB available
                print(f"WARNING: System memory critically low! {{system_memory.percent:.1f}}% used, {{system_memory.available / (1024**3):.1f}} GB available")
                return False
        except:
            pass
        return True
    except Exception as e:
        print(f"[{{stage}}] Memory logging failed: {{e}}")
        return True

def check_memory_limit():
    """Check if current memory usage is within safe limits"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        
        # Limit process memory to 4GB to prevent crashes
        if memory_gb > 4.0:
            print(f"ERROR: Process memory usage too high: {{memory_gb:.1f}} GB. Stopping to prevent crash.")
            return False
        return True
    except Exception as e:
        print(f"Memory limit check failed: {{e}}")
        return True

def force_memory_cleanup():
    """Aggressively clean up memory"""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear any other caches
        if hasattr(torch, 'jit'):
            torch.jit._state._python_cu.clear_cache()
        
        print("Memory cleanup completed")
    except Exception as e:
        print(f"Memory cleanup failed: {{e}}")

try:
    log_memory_usage("Start")
    
    # Check initial memory limits
    if not check_memory_limit():
        print("ERROR: Initial memory usage too high, aborting")
        sys.exit(1)
    
    # Force initial cleanup
    force_memory_cleanup()
    
    from chatterbox.tts import ChatterboxTTS

    # Force CPU usage for stability and lower memory usage
    device = {json.dumps(device)}
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Constrain math library threads on CPU for stability
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print(f"Using device: {{device}}")

    # Clear any existing CUDA cache and force aggressive cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Configure PyTorch backend threads for stability
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
    except Exception:
        pass

    # Force garbage collection before loading
    force_memory_cleanup()

    log_memory_usage("Before model load")
    
    # Check memory before loading model
    if not check_memory_limit():
        print("ERROR: Memory usage too high before model load, aborting")
        sys.exit(1)
    
    # Load the model with explicit memory management
    print("Loading Chatterbox TTS model...")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded successfully")
    except Exception as exc_e:
        print(f"Model loading failed: {exc_e}")
        # Try to free memory and retry once
        force_memory_cleanup()
        if not check_memory_limit():
            print("ERROR: Memory still too high after cleanup, aborting")
            sys.exit(1)
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded successfully on retry")
    
    log_memory_usage("After model load")
    
    # Check memory after model load
    if not check_memory_limit():
        print("ERROR: Memory usage too high after model load, aborting")
        sys.exit(1)
    
    # Read text from file to avoid command line length limits
    text_file_path = r"{text_file_path}"
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    audio_prompt_path = {json.dumps(audio_prompt_path) if audio_prompt_path else "None"}
    output_path = r"{output_path}"
    exaggeration = {exaggeration}
    cfg_weight = {cfg_weight}
    
    print(f"Generating speech for: {{text[:50]}}...")
    print(f"Parameters: exaggeration={{exaggeration}}, cfg_weight={{cfg_weight}}")
    
    log_memory_usage("Before generation")

    # Check memory before generation
    if not check_memory_limit():
        print("ERROR: Memory usage too high before generation, aborting")
        sys.exit(1)

    # Preprocess voice prompt to reduce memory usage (mono, 16kHz, <=3s)
    if audio_prompt_path and audio_prompt_path != "None" and os.path.exists(audio_prompt_path):
        try:
            if not HAS_TORCHAUDIO:
                raise RuntimeError("torchaudio unavailable for prompt preprocessing; skipping")
            prompt_audio, prompt_sr = torchaudio.load(audio_prompt_path)
            # Convert to mono
            if prompt_audio.shape[0] > 1:
                prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
            target_prompt_sr = 16000
            if prompt_sr != target_prompt_sr:
                prompt_audio = torchaudio.functional.resample(prompt_audio, prompt_sr, target_prompt_sr)
                prompt_sr = target_prompt_sr
            # Trim to first 3 seconds
            max_samples = int(3.0 * prompt_sr)
            if prompt_audio.shape[1] > max_samples:
                prompt_audio = prompt_audio[:, :max_samples]
            # Normalize
            max_val = prompt_audio.abs().max().item()
            if max_val > 0:
                prompt_audio = prompt_audio / max_val
            # Save to temp file
            prompt_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            prompt_tmp_path = prompt_tmp.name
            prompt_tmp.close()
            torchaudio.save(prompt_tmp_path, prompt_audio, prompt_sr)
            audio_prompt_path = prompt_tmp_path
            print(f"Preprocessed voice prompt saved: {{audio_prompt_path}}")
            
            # Clean up prompt tensor
            del prompt_audio
            force_memory_cleanup()
            
        except Exception as e:
            print("Prompt preprocessing failed, using original prompt: " + str(e))
    
    # Generate speech with parameters
    if audio_prompt_path != "None":
        print(f"Using voice cloning with prompt: {{audio_prompt_path}}")
        with torch.inference_mode():
            wav = model.generate(
                text, 
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
    else:
        print("Using default voice")
        with torch.inference_mode():
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
    
    print(f"Speech generated successfully")
    print(f"Audio tensor shape: {{wav.shape}}")
    print(f"Sample rate: {{model.sr}}")
    
    log_memory_usage("After generation")
    
    # Check memory after generation
    if not check_memory_limit():
        print("ERROR: Memory usage too high after generation, aborting")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save audio
    # Optional resample to reduce memory/disk size
    TARGET_SR = {int(target_sr)}
    sr_to_use = model.sr
    if TARGET_SR and TARGET_SR > 0 and TARGET_SR != model.sr and HAS_TORCHAUDIO:
        print(f"Resampling from {{model.sr}} to {{TARGET_SR}}")
        try:
            wav = torchaudio.functional.resample(wav, model.sr, TARGET_SR)
            sr_to_use = TARGET_SR
        except Exception as e:
            print("Resample failed, keeping original SR " + str(model.sr) + ": " + str(e))
            sr_to_use = model.sr
    if HAS_TORCHAUDIO:
        try:
            torchaudio.save(output_path, wav, sr_to_use)
            print("Audio saved successfully with torchaudio")
        except Exception as e:
            print("torchaudio.save failed, falling back to wave module: " + str(e))
            HAS_TORCHAUDIO = False
    if not HAS_TORCHAUDIO:
        wav_np = wav.squeeze(0).detach().cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype(np.int16)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr_to_use))
            wf.writeframes(wav_int16.tobytes())
        print("Audio saved successfully with wave module")
    
    log_memory_usage("After save")
    
    # Check memory before cleanup
    if not check_memory_limit():
        print("ERROR: Memory usage too high before cleanup, aborting")
        sys.exit(1)
    
    # Clean up audio tensor but keep model for potential reuse
    del wav
    if 'wav_np' in locals():
        del wav_np
    if 'wav_int16' in locals():
        del wav_int16
    
    # Aggressive cleanup after generation
    force_memory_cleanup()
    
    log_memory_usage("After cleanup")
    
    # Clean up text file
    try:
        os.unlink(text_file_path)
    except:
        pass
    
    # Verify file was created
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        file_size = os.path.getsize(output_path)
        print(f"SUCCESS: Audio file created {{file_size}} bytes")
        print("CHATTERBOX_SUCCESS")
    else:
        print("ERROR: Audio file not created")
        sys.exit(1)
        
except Exception as e:
    print(f"Chatterbox TTS failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
            # The script is already properly formatted, just wrap it under a top-level guard
            tts_script = "if True:\n" + textwrap.indent(tts_script_raw, "    ")
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as script_file:
                script_file.write(tts_script)
                script_path = script_file.name
            
            try:
                # Execute the TTS script with better error handling and monitoring
                self.logger.info(f"Running Chatterbox TTS script for {len(text)} characters")
                
                # Determine timeout for single-chunk generation
                # Priority: environment override -> voice cloning default -> regular default
                timeout_env = os.getenv('CHB_TTS_TIMEOUT_SECONDS')
                if timeout_env and timeout_env.isdigit():
                    timeout = int(timeout_env)
                else:
                    # Voice cloning typically needs more time even for short texts
                    timeout = 1800 if (audio_prompt_path is not None) else 900
                
                self.logger.info(f"Using single-chunk timeout: {timeout} seconds")
                
                result = await self._run_command([str(self.python_path), script_path], cwd=str(current_dir), timeout=int(timeout))
                
                # Sanitize and truncate outputs to avoid noisy progress bars and huge logs
                def _sanitize_output(text: str) -> str:
                    if not text:
                        return ""
                    import re
                    # Drop tqdm-like "Sampling:" lines and carriage-return updates
                    lines = text.splitlines()
                    filtered = []
                    for ln in lines:
                        if re.match(r"^\s*Sampling:\s", ln):
                            continue
                        filtered.append(ln)
                    text2 = "\n".join(filtered)
                    # Truncate to last 2000 chars
                    return text2[-2000:] if len(text2) > 2000 else text2

                safe_stdout = _sanitize_output(result.stdout)
                safe_stderr = _sanitize_output(result.stderr)

                if safe_stdout:
                    self.logger.info(f"TTS stdout (sanitized): {safe_stdout}")
                if safe_stderr:
                    self.logger.info(f"TTS stderr (sanitized): {safe_stderr}")
                
                if result.returncode == 0 and "CHATTERBOX_SUCCESS" in result.stdout:
                    # Check if output file was created
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        file_size = os.path.getsize(output_path)
                        self.logger.info(f"Chatterbox TTS completed successfully: {file_size} bytes")
                        return {
                            "success": True, 
                            "engine": "Chatterbox TTS",
                            "output_path": output_path,
                            "file_size": file_size
                        }
                    else:
                        self.logger.error("Chatterbox TTS script reported success but no audio file created")
                        # Fall through to failure handling below

                # If script did not report success, check if output file was still created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    self.logger.warning(
                        f"Chatterbox TTS script returned code {result.returncode} but audio was created: {file_size} bytes. Accepting as success."
                    )
                    return {
                        "success": True,
                        "engine": "Chatterbox TTS",
                        "output_path": output_path,
                        "file_size": file_size
                    }

                else:
                    error_msg = safe_stderr or safe_stdout or "Unknown error"
                    
                    # Check for specific error types
                    if "3221225477" in str(result.returncode) or "access violation" in error_msg.lower():
                        error_msg = "Memory access violation - text may be too long or system resources insufficient"
                    elif "memory" in error_msg.lower() or "oom" in error_msg.lower():
                        error_msg = "Out of memory error - try reducing text length or freeing system memory"
                    # Do not treat CUDA messages as fatal; they are often harmless on CPU-only systems
                    # Leave error_msg unchanged for any CUDA mentions so we don't misclassify warnings as errors
                    
                    self.logger.error(f"Chatterbox TTS script failed (exit code {result.returncode}): {error_msg}")
                    return {"success": False, "error": f"Chatterbox TTS error: {error_msg}"}
                    
            finally:
                # Clean up script and text files
                try:
                    os.unlink(script_path)
                except:
                    pass
                try:
                    os.unlink(text_file_path)
                except:
                    pass
                
        except Exception as exc:
            # Avoid scoping issues with exception variables on some Python versions
            try:
                import traceback
                tb = traceback.format_exc()
            except Exception:
                tb = ""
            self.logger.error(f"Error generating single chunk: {exc}\n{tb}")
            return {"success": False, "error": f"Single chunk generation failed: {exc}"}
    
    async def _generate_single_chunk_frozen(
        self, 
        text: str, 
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate speech for a single text chunk in frozen environment using cached model."""
        try:
            self.logger.info(f"Generating speech for single chunk in frozen environment: {len(text)} characters")
            
            # Check memory before processing in frozen mode
            if not self._check_system_memory():
                return {
                    "success": False,
                    "error": "Insufficient system memory for frozen TTS processing. Please close other applications and try again."
                }
            
            import warnings
            import gc
            import os
            import torch
            import torchaudio
            import psutil
            
            # Suppress warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            def log_memory_usage(stage):
                """Log current memory usage"""
                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.info(f"[{stage}] Memory: {memory_info.rss / 1024 / 1024:.1f} MB RSS, {memory_info.vms / 1024 / 1024:.1f} MB VMS")
            
            try:
                log_memory_usage("Start")
                
                # Get cached model instead of loading it repeatedly
                model = await self._get_cached_model()
                if model is None or model == "subprocess_mode":
                    self.logger.error("Failed to get cached model")
                    return {"success": False, "error": "Model not available"}
                
                self.logger.info("Using cached Chatterbox TTS model")
                
                log_memory_usage("Before generation")
                
                self.logger.info(f"Generating speech for: {text[:50]}...")
                self.logger.info(f"Parameters: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
                
                # Generate speech with parameters
                if audio_prompt_path:
                    self.logger.info(f"Using voice cloning with prompt: {audio_prompt_path}")
                    wav = model.generate(
                        text, 
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
                else:
                    self.logger.info("Using default voice")
                    wav = model.generate(
                        text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
                
                self.logger.info(f"Speech generated successfully")
                self.logger.info(f"Audio tensor shape: {wav.shape}")
                self.logger.info(f"Sample rate: {model.sr}")
                
                log_memory_usage("After generation")
                
                # Save audio
                torchaudio.save(output_path, wav, model.sr)
                
                log_memory_usage("After save")
                
                # Clean up audio tensor but keep model cached
                del wav
                gc.collect()
                
                log_memory_usage("After cleanup")
                
                # Verify file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    self.logger.info(f"Chatterbox TTS completed successfully: {file_size} bytes")
                    return {
                        "success": True, 
                        "engine": "Chatterbox TTS (frozen, cached)",
                        "output_path": output_path,
                        "file_size": file_size
                    }
                else:
                    self.logger.error("Chatterbox TTS succeeded but no audio file created")
                    return {"success": False, "error": "No audio file generated"}
                    
            except MemoryError as e:
                self.logger.error(f"Memory error in frozen TTS generation: {e}")
                # Clear cached model on memory error
                await self._clear_cached_model()
                return {"success": False, "error": f"Memory error: {str(e)}"}
            except Exception as e:
                self.logger.error(f"Error in frozen TTS generation: {e}")
                return {"success": False, "error": f"Frozen TTS generation failed: {str(e)}"}
                
        except Exception as e:
            self.logger.error(f"Error in frozen single chunk generation: {e}")
            return {"success": False, "error": f"Frozen single chunk generation failed: {str(e)}"}
    
    async def _generate_and_combine_chunks(
        self, 
        text_chunks: list, 
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None,
        timeout_seconds: int = 1800  # Increased from 600 to 1800 seconds (30 minutes) for voice cloning
    ) -> Dict[str, Any]:
        """Generate speech for multiple chunks and combine them into a single audio file with proper memory management."""
        try:
            self.logger.info(f"Generating speech for {len(text_chunks)} chunks with memory management")
            
            # Check memory before processing multiple chunks
            if not self._check_system_memory():
                return {
                    "success": False,
                    "error": "Insufficient system memory for multi-chunk TTS processing. Please close other applications and try again."
                }
            
            # For very large books, log progress estimates
            if len(text_chunks) > 50:
                self.logger.info(f"Large book processing: {len(text_chunks)} chunks detected")
                self.logger.info(f"Estimated processing time: {len(text_chunks) * 2} minutes")
                self.logger.info(f"Timeout set to: {timeout_seconds/60:.1f} minutes")
            
            # For very large books (100+ chunks), we might want to process in smaller batches
            # to avoid memory issues, but for now we'll process all at once with better timeout
            if len(text_chunks) > 100:
                self.logger.warning(f"Very large book: {len(text_chunks)} chunks. Consider splitting into smaller books for optimal performance.")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Process chunks individually to manage memory properly
            temp_audio_files = []
            
            for i, chunk in enumerate(text_chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
                
                # Check memory before processing each chunk
                if not self._check_system_memory():
                    self.logger.error(f"Insufficient memory before processing chunk {i+1}")
                    # Clean up temp files
                    for temp_file in temp_audio_files:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                    return {
                        "success": False,
                        "error": f"Insufficient system memory while processing chunk {i+1}. Please close other applications and try again."
                    }
                
                # Create temporary file for this chunk
                temp_audio_file = os.path.join(output_dir, f"temp_chunk_{i:04d}.wav")
                temp_audio_files.append(temp_audio_file)
                
                # Generate speech for this chunk (with retry-on-split fallback)
                chunk_result = await self._generate_single_chunk(
                    chunk, 
                    temp_audio_file,
                    exaggeration,
                    cfg_weight,
                    audio_prompt_path
                )
                if not chunk_result.get("success"):
                    err_msg = str(chunk_result.get("error", ""))
                    # If we hit memory/access violation, try to split this chunk smaller and retry
                    if ("memory" in err_msg.lower()) or ("access violation" in err_msg.lower()):
                        self.logger.warning(f"Chunk {i+1} failed due to memory; retrying with smaller subchunks")
                        subchunks = self._split_text_into_chunks(chunk, max_chunk_size=max(80, len(chunk)//2))
                        sub_temp_files = []
                        retry_failed = False
                        for j, sub in enumerate(subchunks):
                            sub_temp = os.path.join(output_dir, f"temp_chunk_{i:04d}_{j:02d}.wav")
                            sub_temp_files.append(sub_temp)
                            sub_res = await self._generate_single_chunk(
                                sub,
                                sub_temp,
                                exaggeration,
                                cfg_weight,
                                audio_prompt_path
                            )
                            if not sub_res.get("success"):
                                retry_failed = True
                                self.logger.error(f"Subchunk {j+1}/{len(subchunks)} for chunk {i+1} failed: {sub_res.get('error')}")
                                break
                        if not retry_failed:
                            # Combine subchunks into the original temp file path
                            combine_sub = await self._combine_audio_files(sub_temp_files, temp_audio_file)
                            # Cleanup sub temp files
                            for fp in sub_temp_files:
                                try:
                                    if os.path.exists(fp):
                                        os.remove(fp)
                                except:
                                    pass
                            if not combine_sub.get("success"):
                                self.logger.error(f"Failed to combine subchunks for chunk {i+1}: {combine_sub.get('error')}")
                                chunk_result = combine_sub
                            else:
                                chunk_result = {"success": True, "output_path": temp_audio_file}
                        else:
                            # Cleanup partial sub temp files
                            for fp in sub_temp_files:
                                try:
                                    if os.path.exists(fp):
                                        os.remove(fp)
                                except:
                                    pass
                
                if not chunk_result["success"]:
                    self.logger.error(f"Failed to generate chunk {i+1}: {chunk_result['error']}")
                    # Clean up temp files
                    for temp_file in temp_audio_files:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                    return chunk_result
                
                # Aggressive memory cleanup between chunks
                import gc
                import psutil
                
                # Log memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.info(f"Memory after chunk {i+1}: {memory_info.rss / 1024 / 1024:.1f} MB RSS")
                
                # Force multiple garbage collection passes
                for gc_pass in range(3):
                    collected = gc.collect()
                    if collected > 0:
                        self.logger.debug(f"GC pass {gc_pass+1}: collected {collected} objects")
                
                # Small delay to allow system to clean up
                await asyncio.sleep(0.1)
                
                # Log progress
                if (i + 1) % 10 == 0 or i == len(text_chunks) - 1:
                    self.logger.info(f"Completed {i+1}/{len(text_chunks)} chunks")
            
            # Combine all audio files
            self.logger.info("Combining audio chunks...")
            combined_result = await self._combine_audio_files(temp_audio_files, output_path)
            
            # Clean up temporary files
            for temp_file in temp_audio_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
            
            if combined_result["success"]:
                file_size = os.path.getsize(output_path)
                self.logger.info(f"Successfully combined {len(text_chunks)} chunks into {file_size} bytes")
                return {"success": True, "engine": "Chatterbox TTS (chunked)", "output_path": output_path, "file_size": file_size}
            else:
                return combined_result
                
        except Exception as e:
            self.logger.error(f"Error in chunked generation: {e}")
            return {"success": False, "error": f"Chunked generation failed: {str(e)}"}
    
    async def _combine_audio_files(self, audio_files: list, output_path: str):
        """Combine multiple audio files into a single file."""
        try:
            self.logger.info(f"Combining {len(audio_files)} audio files into {output_path}")
            
            if self.is_frozen:
                # Use direct import approach for frozen environment
                return await self._combine_audio_files_frozen(audio_files, output_path)
            else:
                # Use subprocess approach for development environment
                # Set working directory to backend-api to ensure chatterbox/src is in Python path
                current_dir = Path(__file__).parent.parent
                
            # Create a script to combine audio files
            combine_script_raw = f'''
import os
import sys
import torch
import torchaudio
import json

try:
    # Load all audio files
    audio_files = {json.dumps(audio_files)}
    output_path = r"{output_path}"
    
    print(f"Loading {{len(audio_files)}} audio files...")
    
    # Load and concatenate audio files
    combined_audio = None
    sample_rate = None
    
    for i, audio_file in enumerate(audio_files):
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {{audio_file}} not found, skipping...")
            continue
            
        print(f"Loading audio file {{i+1}}/{{len(audio_files)}}: {{audio_file}}")
        audio, sr = torchaudio.load(audio_file)
        
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            print(f"Warning: Sample rate mismatch ({{sr}} vs {{sample_rate}}), resampling...")
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        
        if combined_audio is None:
            combined_audio = audio
        else:
            combined_audio = torch.cat([combined_audio, audio], dim=1)
    
    if combined_audio is not None:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save combined audio
        torchaudio.save(output_path, combined_audio, sample_rate)
        print(f"Successfully combined {{len(audio_files)}} files into {{output_path}}")
        print("COMBINE_SUCCESS")
    else:
        print("No valid audio files to combine")
        sys.exit(1)
        
except Exception as e:
    print(f"Error combining audio files: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
            # Preserve relative indentation of the try: block when wrapping
            combine_script_norm = textwrap.dedent(combine_script_raw).strip("\n")
            combine_script = "if True:\n" + textwrap.indent(combine_script_norm, "    ")
            
            # Write script to temporary file
            script_path = self.venv_path / "combine_audio.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(combine_script)
            
            # Determine timeout for audio combination step
            timeout_env = os.getenv('CHB_TTS_COMBINE_TIMEOUT_SECONDS') or os.getenv('CHB_TTS_TIMEOUT_SECONDS')
            try:
                timeout_seconds = int(timeout_env) if timeout_env else 600
            except Exception:
                timeout_seconds = 600

            self.logger.info(f"Using combine timeout: {timeout_seconds} seconds")

            # Run the script
            result = await self._run_command([str(self.python_path), str(script_path)], cwd=str(current_dir), timeout=timeout_seconds)
            
            # Clean up script
            try:
                script_path.unlink()
            except:
                pass
            
            if result.returncode == 0 and "COMBINE_SUCCESS" in result.stdout:
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    self.logger.info(f"Audio combination successful: {file_size} bytes")
                    return {"success": True, "output_path": output_path, "file_size": file_size}
                else:
                    self.logger.error("Audio combination completed but no output file was generated")
                    return {"success": False, "error": "No output file generated"}
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                self.logger.error(f"Audio combination failed: {error_msg}")
                return {"success": False, "error": f"Audio combination failed: {error_msg}"}
                
        except Exception as e:
            self.logger.error(f"Error in audio combination: {e}")
            return {"success": False, "error": f"Audio combination error: {str(e)}"}
    
    async def _combine_audio_files_frozen(self, audio_files: list, output_path: str):
        """Combine multiple audio files into a single file in frozen environment using direct imports."""
        try:
            self.logger.info(f"Combining {len(audio_files)} audio files in frozen environment")
            
            import os
            import torch
            import torchaudio
            
            # Load and concatenate audio files
            combined_audio = None
            sample_rate = None
            
            for i, audio_file in enumerate(audio_files):
                if not os.path.exists(audio_file):
                    self.logger.warning(f"Audio file {audio_file} not found, skipping...")
                    continue
                    
                self.logger.info(f"Loading audio file {i+1}/{len(audio_files)}: {audio_file}")
                audio, sr = torchaudio.load(audio_file)
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    self.logger.warning(f"Sample rate mismatch ({sr} vs {sample_rate}), resampling...")
                    audio = torchaudio.functional.resample(audio, sr, sample_rate)
                
                if combined_audio is None:
                    combined_audio = audio
                else:
                    combined_audio = torch.cat([combined_audio, audio], dim=1)
            
            if combined_audio is not None:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save combined audio
                torchaudio.save(output_path, combined_audio, sample_rate)
                self.logger.info(f"Successfully combined {len(audio_files)} files into {output_path}")
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path)
                    self.logger.info(f"Audio combination successful: {file_size} bytes")
                    return {"success": True, "output_path": output_path, "file_size": file_size}
                else:
                    self.logger.error("Audio combination completed but no output file was generated")
                    return {"success": False, "error": "No output file generated"}
            else:
                self.logger.error("No valid audio files to combine")
                return {"success": False, "error": "No valid audio files to combine"}
                
        except Exception as e:
            self.logger.error(f"Error in frozen audio combination: {e}")
            return {"success": False, "error": f"Frozen audio combination error: {str(e)}"}
    
    async def generate_speech_from_file(
        self, 
        markdown_path: str, 
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        audio_prompt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate speech from a markdown file.
        
        Args:
            markdown_path: Path to the markdown file
            output_path: Path where to save the audio file
            exaggeration: Emotion exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            audio_prompt_path: Optional path to audio file for voice cloning
            
        Returns:
            Dictionary with success status and details
        """
        try:
            # Read the markdown content
            if not os.path.exists(markdown_path):
                return {"success": False, "error": "Markdown file not found"}
            
            with open(markdown_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                return {"success": False, "error": "No text content found"}
            
            return await self.generate_speech(
                text=text,
                output_path=output_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                audio_prompt_path=audio_prompt_path
            )
            
        except Exception as e:
            self.logger.error(f"Error reading markdown file: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not await self.is_available():
            return {"available": False}
        
        if self.is_frozen:
            return {
                "available": True,
                "venv_path": str(self.venv_path),
                "python_path": str(self.python_path),
                "model_type": "Chatterbox TTS (frozen, cached)" if self._cached_model and self._cached_model != "subprocess_mode" else "Chatterbox TTS (frozen)"
            }
        else:
            return {
                "available": True,
                "venv_path": str(self.venv_path),
                "python_path": str(self.python_path),
                "model_type": "Chatterbox TTS (venv)"
            }
    
    async def cleanup(self):
        """Clean up resources and clear cached model."""
        await self._clear_cached_model()
        self.logger.info("Chatterbox TTS service cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if self._cached_model is not None and self._cached_model != "subprocess_mode":
                # Force cleanup in destructor
                if hasattr(self._cached_model, 'clear_cache'):
                    self._cached_model.clear_cache()
                del self._cached_model
                gc.collect()
        except:
            pass 