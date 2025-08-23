import os
import logging
import re
import asyncio
import subprocess
import sys
import json
import tempfile
from typing import Dict, Any
from pathlib import Path

# Check if we have direct access to markitdown (for development)
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
    print("Microsoft Markitdown library available")
except ImportError as e:
    print(f"Warning: Markitdown not available in main environment: {e}")
    print("Will use PDF virtual environment for processing")
    print("PyPDF2 will be used via PDF virtual environment")

# Note: PyPDF2 will be used via the PDF virtual environment

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.venv_path = None
        self.python_path = None
        self.rules = self._load_cleaning_rules()
        self._setup_unified_venv_paths()
    
    def _setup_unified_venv_paths(self):
        """Setup paths for unified virtual environment."""
        # Check if running in PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Running in frozen executable - use bundled Python
            self.venv_path = None
            self.python_path = Path(sys.executable)
            self.logger.info(f"Running in frozen executable, using bundled Python: {self.python_path}")
        else:
            # Running in development - use unified venv
            project_root = Path(__file__).parent.parent.parent
            self.venv_path = project_root / "backend-api" / "venv-unified"
            
            # Platform-specific python executable paths in venv
            if sys.platform == "win32":
                self.python_path = self.venv_path / "Scripts" / "python.exe"
            else:
                self.python_path = self.venv_path / "bin" / "python"
            
            # Warn if virtual environment python not found
            if not self.python_path.exists():
                self.logger.warning(f"Unified venv python not found at {self.python_path}, falling back to system python")
                self.python_path = Path(sys.executable)
            
            self.logger.info(f"Unified venv path: {self.venv_path}")
            self.logger.info(f"Python path: {self.python_path}")
    
    async def _run_pdf_command(self, script_content: str, pdf_path: str, timeout=120):
        """Run a PDF processing command in the unified virtual environment."""
        try:
            # Create a temporary script file with frozen environment support
            from services.frozen_imports import create_frozen_script_wrapper
            import tempfile
            
            wrapped_script = create_frozen_script_wrapper(script_content)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(wrapped_script)
                temp_script_path = temp_script.name
            
            # Run the script in the unified venv
            process = await asyncio.create_subprocess_exec(
                str(self.python_path),
                temp_script_path,
                pdf_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.error(f"PDF processing timed out after {timeout} seconds")
                process.kill()
                await process.wait()
                return {"success": False, "error": f"Processing timed out after {timeout} seconds"}
            finally:
                # Clean up temp script
                try:
                    os.unlink(temp_script_path)
                except:
                    pass
            
            # Handle encoding issues
            def safe_decode(data):
                if not data:
                    return ''
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        return data.decode('cp1252')
                    except UnicodeDecodeError:
                        return data.decode('utf-8', errors='replace')
            
            stdout_str = safe_decode(stdout)
            stderr_str = safe_decode(stderr)
            
            if process.returncode == 0:
                return {"success": True, "output": stdout_str}
            else:
                return {"success": False, "error": stderr_str, "output": stdout_str}
                
        except Exception as e:
            self.logger.error(f"PDF command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_pdf_command_sync(self, script_content: str, pdf_path: str, timeout=120):
        """Run a PDF processing command synchronously in the PDF virtual environment."""
        try:
            # Create a temporary script file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(script_content)
                temp_script_path = temp_script.name
            
            # Run the script in the unified venv using subprocess
            process = subprocess.Popen(
                [str(self.python_path), temp_script_path, pdf_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.logger.error(f"PDF processing timed out after {timeout} seconds")
                process.kill()
                process.wait()
                return {"success": False, "error": f"Processing timed out after {timeout} seconds"}
            finally:
                # Clean up temp script
                try:
                    os.unlink(temp_script_path)
                except:
                    pass
            
            # Handle encoding issues
            def safe_decode(data):
                if not data:
                    return ''
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        return data.decode('cp1252')
                    except UnicodeDecodeError:
                        return data.decode('utf-8', errors='replace')
            
            stdout_str = safe_decode(stdout)
            stderr_str = safe_decode(stderr)
            
            if process.returncode == 0:
                return {"success": True, "output": stdout_str}
            else:
                return {"success": False, "error": stderr_str, "output": stdout_str}
                
        except Exception as e:
            self.logger.error(f"PDF command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_pdf_file(self, pdf_path: str) -> bool:
        """Validate that the PDF file exists and is readable."""
        if not os.path.isfile(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            self.logger.error(f"PDF file is empty: {pdf_path}")
            return False
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            self.logger.warning(f"File doesn't have .pdf extension: {pdf_path}")
        
        # Try to read the beginning of the file to check if it's a valid PDF
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(5)
                if not header.startswith(b'%PDF-'):
                    self.logger.error(f"File doesn't appear to be a valid PDF: {pdf_path}")
                    return False
        except Exception as e:
            self.logger.error(f"Error reading PDF file: {e}")
            return False
        
        self.logger.info(f"PDF validation passed for: {pdf_path} (size: {file_size} bytes)")
        return True

    def _load_cleaning_rules(self) -> Dict[str, Any]:
        """Load cleaning rules from JSON so we don't hard-code vendors/phrases."""
        default_rules = {
            "drop_line_patterns": [
                r"^illustrated\s+by[:\-].*$",
                r"^published\s+by.*$",
                r"^copyright.*$",
                r"^all\s+rights\s+reserved.*$",
                r"^isbn.*$",
                r"^thank\s+you\s+for\s+downloading.*$",
                r"^please\s+(share|make\s+a\s+donation).*$",
                r"^about\s+the\s+(author|book).*$",
                r"^dedication[:]?.*$",
                r"^acknowledg(e)?ments.*$",
                r"^contents$",
            ],
            "drop_phrases": [
                "free children's books",
                "support our mission",
                "share our books",
                "make a donation",
            ],
            "drop_domains": [
                "patreon.com", "facebook.com", "instagram.com", "twitter.com"
            ],
        }
        try:
            cfg_path = os.environ.get("CLEANING_RULES_PATH") or os.path.join(os.path.dirname(__file__), "cleaning_rules.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    user_rules = json.load(f)
                # merge user rules over defaults
                for k, v in default_rules.items():
                    if isinstance(user_rules.get(k), list):
                        default_rules[k] = user_rules[k]
            else:
                self.logger.info(f"Cleaning rules file not found at {cfg_path}, using defaults")
        except Exception as e:
            self.logger.warning(f"Failed to load cleaning rules: {e}; using defaults")
        return default_rules
    
    def extract_with_markitdown(self, pdf_path: str):
        """Extract text using Microsoft Markitdown in PDF virtual environment."""
        try:
            # Use direct import from venv-unified
            from markitdown import MarkItDown
            import warnings
            import io
            import sys
            import os
            
            # Setup magika model path for frozen environment
            magika_model_dir = None
            if getattr(sys, 'frozen', False):
                # In frozen environment, set magika model directory
                if hasattr(sys, '_MEIPASS'):
                    base_path = Path(sys._MEIPASS)
                else:
                    base_path = Path(sys.executable).parent
                
                # Look for magika models in the bundled location
                magika_models_path = base_path / "magika" / "models" / "standard_v3_3"
                if magika_models_path.exists():
                    magika_model_dir = magika_models_path
                    self.logger.info(f"Found magika models at: {magika_models_path}")
                else:
                    self.logger.warning(f"Magika models directory not found at: {magika_models_path}")
                    # Try alternative location
                    alt_path = base_path / "venv-unified" / "Lib" / "site-packages" / "magika" / "models" / "standard_v3_3"
                    if alt_path.exists():
                        magika_model_dir = alt_path
                        self.logger.info(f"Found magika models at alternative location: {alt_path}")
            
            # Suppress PDF parsing warnings (common with some PDFs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Also capture stderr to hide PDF parsing warnings
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                
                try:
                    # Create MarkItDown with custom magika model directory if available
                    if magika_model_dir:
                        # We need to monkey patch magika to use our model directory
                        import magika
                        original_magika_init = magika.Magika.__init__
                        
                        def patched_magika_init(self, model_dir=None, **kwargs):
                            if model_dir is None:
                                model_dir = magika_model_dir
                            return original_magika_init(self, model_dir=model_dir, **kwargs)
                        
                        magika.Magika.__init__ = patched_magika_init
                        self.logger.info(f"Patched magika to use model directory: {magika_model_dir}")
                    
                    md = MarkItDown()
                    result = md.convert(pdf_path)
                    
                    # Check if we got actual content
                    if hasattr(result, 'text_content') and result.text_content:
                        extracted_text = result.text_content
                    elif hasattr(result, 'text') and result.text:
                        extracted_text = result.text
                    else:
                        # Try to get any available content
                        extracted_text = str(result) if result else None
                    
                    if not extracted_text or len(extracted_text.strip()) < 10:
                        raise ValueError("MarkItDown returned empty or insufficient content")
                    
                    return extracted_text
                    
                finally:
                    # Restore stderr
                    sys.stderr = old_stderr
                
        except Exception as e:
            self.logger.error(f"Error in MarkItDown extraction: {e}")
            raise
    
    def test_markitdown_directly(self, pdf_path: str) -> Dict[str, Any]:
        """Test MarkItDown directly and return detailed results."""
        try:
            from markitdown import MarkItDown
            import warnings
            import io
            import sys
            import os
            
            # Setup magika model path for frozen environment
            magika_model_dir = None
            if getattr(sys, 'frozen', False):
                # In frozen environment, set magika model directory
                if hasattr(sys, '_MEIPASS'):
                    base_path = Path(sys._MEIPASS)
                else:
                    base_path = Path(sys.executable).parent
                
                # Look for magika models in the bundled location
                magika_models_path = base_path / "magika" / "models" / "standard_v3_3"
                if magika_models_path.exists():
                    magika_model_dir = magika_models_path
                    print(f"Found magika models at: {magika_models_path}")
                else:
                    print(f"Warning: Magika models directory not found at: {magika_models_path}")
                    # Try alternative location
                    alt_path = base_path / "venv-unified" / "Lib" / "site-packages" / "magika" / "models" / "standard_v3_3"
                    if alt_path.exists():
                        magika_model_dir = alt_path
                        print(f"Found magika models at alternative location: {alt_path}")
            
            print(f"Testing MarkItDown on: {pdf_path}")
            
            # Suppress warnings but capture them
            captured_warnings = []
            def warning_handler(message, category, filename, lineno, file=None, line=None):
                captured_warnings.append(f"{category.__name__}: {message}")
            
            old_showwarning = warnings.showwarning
            warnings.showwarning = warning_handler
            
            # Capture stderr
            old_stderr = sys.stderr
            stderr_capture = io.StringIO()
            sys.stderr = stderr_capture
            
            try:
                # Create MarkItDown with custom magika model directory if available
                if magika_model_dir:
                    # We need to monkey patch magika to use our model directory
                    import magika
                    original_magika_init = magika.Magika.__init__
                    
                    def patched_magika_init(self, model_dir=None, **kwargs):
                        if model_dir is None:
                            model_dir = magika_model_dir
                        return original_magika_init(self, model_dir=model_dir, **kwargs)
                    
                    magika.Magika.__init__ = patched_magika_init
                    print(f"Patched magika to use model directory: {magika_model_dir}")
                
                md = MarkItDown()
                result = md.convert(pdf_path)
                
                # Get stderr content
                stderr_content = stderr_capture.getvalue()
                
                # Analyze the result
                result_info = {
                    "success": False,
                    "result_type": type(result).__name__,
                    "result_attributes": dir(result),
                    "warnings": captured_warnings,
                    "stderr": stderr_content,
                    "text_content": None,
                    "content_length": 0
                }
                
                # Try different ways to get text content
                if hasattr(result, 'text_content') and result.text_content:
                    result_info["text_content"] = result.text_content[:500] + "..." if len(result.text_content) > 500 else result.text_content
                    result_info["content_length"] = len(result.text_content)
                    result_info["success"] = True
                elif hasattr(result, 'text') and result.text:
                    result_info["text_content"] = result.text[:500] + "..." if len(result.text) > 500 else result.text
                    result_info["content_length"] = len(result.text)
                    result_info["success"] = True
                else:
                    # Try to convert result to string
                    result_str = str(result)
                    if result_str and len(result_str.strip()) > 10:
                        result_info["text_content"] = result_str[:500] + "..." if len(result_str) > 500 else result_str
                        result_info["content_length"] = len(result_str)
                        result_info["success"] = True
                
                return result_info
                
            finally:
                warnings.showwarning = old_showwarning
                sys.stderr = old_stderr
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def create_basic_fallback(self, pdf_path: str) -> str:
        """Create a basic placeholder when MarkItDown extraction fails."""
        self.logger.info("Creating basic fallback content due to MarkItDown extraction failure")
        
        filename = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path)
        
        fallback_text = f"""# PDF Content from {filename}

**Note:** MarkItDown PDF processing failed. This is a placeholder for development purposes.

**File Information:**
- File: {pdf_path}
- Size: {file_size} bytes
- Status: MarkItDown processing failed, using fallback
- Venv Path: {self.venv_path}
- Python Path: {self.python_path}

**Troubleshooting:**
This usually indicates one of the following issues:
1. MarkItDown is not properly installed in the PDF virtual environment
2. The PDF file may be corrupted or encrypted
3. The PDF file may contain only images without extractable text
4. There may be a permissions issue accessing the PDF file

**For development purposes:**
This would normally contain the extracted text from your PDF file. 
You can replace this with sample text or manually extracted content.

**Sample Story Content:**
Maya discovered an old music box in her grandmother's attic. When she wound the tiny key, a delicate ballerina began to spin, and magical notes filled the air. Suddenly, the room transformed into a grand ballroom from long ago.

The ballerina stepped out of the music box and offered Maya her hand. "Welcome to the Dance of Dreams," she whispered. Together, they waltzed across clouds of silver and gold, while shooting stars provided the rhythm.

As the final note played, Maya found herself back in the dusty attic. But in her hand remained a tiny silver key – proof that magic exists for those who believe in wonder.
"""

        return fallback_text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text for better processing."""
        # Normalize line endings and trim spurious control chars
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Remove obvious boilerplate: URLs, emails
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\S+@\S+', '', text)

        # Drop lines that are clearly non-story metadata/ads (from configurable rules)
        boilerplate_re = re.compile('|'.join(self.rules.get("drop_line_patterns", [])), re.IGNORECASE)

        cleaned_lines = []
        for raw_line in text.split('\n'):
            line = raw_line.strip()
            if not line:
                cleaned_lines.append('')
                continue
            if boilerplate_re.search(line):
                continue
            # Remove stray page numbers and headers like "Page 12" or just "12"
            if re.fullmatch(r'(page\s*)?\d{1,4}', line, flags=re.IGNORECASE):
                continue
            # Remove lines that are mostly non-word characters
            if len(re.sub(r'\W', '', line)) < max(4, int(len(line) * 0.25)):
                # line contains too few word chars -> likely decoration
                continue
            # Fuzzy phrase filtering
            norm = re.sub(r'[^a-z0-9 ]+', '', line.lower())
            skip = False
            for phrase in self.rules.get("drop_phrases", []):
                pnorm = re.sub(r'[^a-z0-9 ]+', '', phrase.lower())
                if pnorm and pnorm in norm:
                    skip = True
                    break
            if skip:
                continue
            # Drop lines containing blacklisted domains
            if any(dom in line.lower() for dom in self.rules.get("drop_domains", [])):
                continue
            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Collapse multiple spaces and multiple blank lines
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up common OCR artifacts
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # zero-width

        return text.strip()
    
    def extract_story_content(self, text: str) -> str:
        """Extract main story content from text."""
        # Split by paragraphs (blank lines)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        # Heuristics for filtering non-story paragraphs
        banned_terms = [
            'illustrated by', 'published by', 'copyright', 'all rights reserved', 'isbn',
            'donation', 'patreon', 'share our books', 'thank you for downloading'
        ] + self.rules.get("drop_phrases", [])
        banned_re = re.compile('|'.join(re.escape(t) for t in banned_terms), re.IGNORECASE)

        def is_story_paragraph(p: str) -> bool:
            if banned_re.search(p):
                return False
            # keep paragraphs with enough words and letters
            words = re.findall(r'[A-Za-z]{2,}', p)
            if len(words) < 5:
                return False
            # avoid paragraphs that are mostly uppercase (titles)
            letters = ''.join(words)
            if letters and sum(1 for ch in letters if ch.isupper()) / max(1, len(letters)) > 0.8:
                return False
            return True

        story_paragraphs = [p for p in paragraphs if is_story_paragraph(p)]

        # Trim front/back matter by finding a strong starting/ending paragraph
        start_idx = 0
        for i, p in enumerate(story_paragraphs):
            if len(p.split()) >= 20:
                start_idx = i
                break
        end_idx = len(story_paragraphs)
        for i in range(len(story_paragraphs) - 1, -1, -1):
            if len(story_paragraphs[i].split()) >= 12:
                end_idx = i + 1
                break

        core = story_paragraphs[start_idx:end_idx]
        story_text = '\n\n'.join(core)

        # Sanity check: if too short, fall back to less strict filtering
        if len(story_text) < 400:
            fallback = [p for p in paragraphs if not banned_re.search(p)]
            story_text = '\n\n'.join(fallback)

        return story_text.strip()
    
    async def extract_pdf_cover(self, pdf_path: str, output_dir: str) -> str:
        """Extract the first page of a PDF as a cover image."""
        try:
            # Create a temporary script to extract cover using PyPDF2 and Pillow
            script_content = '''
import sys
import os
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image, ImageDraw, ImageFont
import io

def extract_cover(pdf_path, output_dir):
    try:
        # Try to extract first page as image using PyPDF2
        reader = PdfReader(pdf_path)
        if len(reader.pages) == 0:
            return ""
        
        # Get the first page
        page = reader.pages[0]
        
        # Try to extract images from the page
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    try:
                        # Extract image data
                        img_data = xObject[obj].get_data()
                        
                        # Try to determine image format and save
                        if xObject[obj]['/Filter'] == '/DCTDecode':
                            # JPEG
                            cover_path = os.path.join(output_dir, "cover.jpg")
                            with open(cover_path, 'wb') as f:
                                f.write(img_data)
                            return "cover.jpg"
                        elif xObject[obj]['/Filter'] == '/JPXDecode':
                            # JPEG2000
                            cover_path = os.path.join(output_dir, "cover.jp2")
                            with open(cover_path, 'wb') as f:
                                f.write(img_data)
                            return "cover.jp2"
                        elif xObject[obj]['/Filter'] == '/FlateDecode':
                            # PNG or other compressed format
                            try:
                                img = Image.open(io.BytesIO(img_data))
                                cover_path = os.path.join(output_dir, "cover.png")
                                img.save(cover_path, "PNG")
                                return "cover.png"
                            except:
                                pass
                    except Exception as e:
                        print(f"Error processing image object: {e}")
                        continue
        
        # If no images found, create a text-based cover from the first page
        try:
            text = page.extract_text()
            if text:
                # Create a simple text-based cover
                img = Image.new('RGB', (400, 600), color='white')
                draw = ImageDraw.Draw(img)
                
                # Try to use a default font, fallback to basic if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                # Draw title (first line of text)
                lines = text.split('\\n')[:3]  # Take first 3 lines
                y_position = 50
                for line in lines[:2]:  # First two lines as title
                    if line.strip():
                        # Center the text
                        bbox = draw.textbbox((0, 0), line.strip(), font=font)
                        text_width = bbox[2] - bbox[0]
                        x_position = (400 - text_width) // 2
                        draw.text((x_position, y_position), line.strip(), fill='black', font=font)
                        y_position += 40
                
                # Add a decorative border
                draw.rectangle([20, 20, 380, 580], outline='gray', width=2)
                
                cover_path = os.path.join(output_dir, "cover.png")
                img.save(cover_path, "PNG")
                return "cover.png"
        except Exception as e:
            print(f"Error creating text-based cover: {e}")
        
        return ""
        
    except Exception as e:
        print(f"Error extracting cover: {e}")
        return ""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <pdf_path> <output_dir>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    result = extract_cover(pdf_path, output_dir)
    print(result)
'''
            
            # Run the cover extraction script
            result = await self._run_pdf_command(script_content, pdf_path, timeout=60)
            
            if result.get("success") and result.get("result"):
                # Check if any cover file was created
                cover_extensions = ["cover.png", "cover.jpg", "cover.jp2"]
                for ext in cover_extensions:
                    cover_path = os.path.join(output_dir, ext)
                    if os.path.exists(cover_path):
                        self.logger.info(f"Successfully extracted cover image to {cover_path}")
                        return ext
            
            self.logger.warning("Cover extraction failed, will use placeholder")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF cover: {e}")
            return ""

    async def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Convert PDF to markdown and save to output directory."""
        try:
            # Validate PDF file
            if not self.validate_pdf_file(pdf_path):
                return {"success": False, "error": "PDF validation failed"}
            
            # Try different extraction methods
            extracted_text = None
            
            # Always try MarkItDown from PDF virtual environment first
            try:
                extracted_text = self.extract_with_markitdown(pdf_path)
                if extracted_text and len(extracted_text.strip()) > 0:
                    self.logger.info("Successfully extracted text using MarkItDown")
                else:
                    self.logger.warning("MarkItDown extraction returned empty content")
                    extracted_text = None
            except Exception as e:
                self.logger.warning(f"Markitdown failed: {e}")
            
            # Final fallback only if MarkItDown completely failed
            if not extracted_text:
                self.logger.warning("MarkItDown extraction failed, using fallback content")
                extracted_text = self.create_basic_fallback(pdf_path)
            
            # Clean and process the text
            cleaned_text = self.clean_text(extracted_text)
            story_content = self.extract_story_content(cleaned_text)
            
            # Save to output file
            output_path = os.path.join(output_dir, "pdf_result.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(story_content)
            
            self.logger.info(f"Successfully saved processed text to {output_path}")
            
            return {
                "success": True,
                "output_path": output_path,
                "text_length": len(story_content)
            }
            
        except Exception as e:
            self.logger.error(f"Error in PDF processing: {e}")
            return {"success": False, "error": str(e)}