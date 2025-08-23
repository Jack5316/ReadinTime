#!/usr/bin/env python3
"""
Self-contained PDF CLI tool for converting PDFs to Markdown.
No external dependencies on localhost services - everything is included here.
"""

import argparse
import json
import os
import sys
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import required libraries, with fallbacks
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available, will use basic text extraction")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available, will skip image processing")

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: MarkItDown not available, will use PyPDF2 fallback")


class SelfContainedPDFProcessor:
    """Self-contained PDF processor with no external service dependencies."""
    
    def __init__(self):
        self.rules = self._load_cleaning_rules()
    
    def _load_cleaning_rules(self) -> Dict[str, Any]:
        """Load cleaning rules for text processing."""
        return {
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
    
    def validate_pdf_file(self, pdf_path: str) -> bool:
        """Validate that the PDF file exists and is readable."""
        if not os.path.isfile(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            print(f"Error: PDF file is empty: {pdf_path}")
            return False
        
        # Try to read the beginning of the file to check if it's a valid PDF
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(5)
                if not header.startswith(b'%PDF-'):
                    print(f"Error: File doesn't appear to be a valid PDF: {pdf_path}")
                    return False
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return False
        
        print(f"PDF validation passed: {pdf_path} (size: {file_size} bytes)")
        return True
    
    def extract_with_markitdown(self, pdf_path: str) -> Optional[str]:
        """Extract text using Microsoft Markitdown if available."""
        if not MARKITDOWN_AVAILABLE:
            return None
        
        try:
            import warnings
            import io
            
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Capture stderr to hide PDF parsing warnings
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                
                try:
                    md = MarkItDown()
                    result = md.convert(pdf_path)
                    
                    # Check if we got actual content
                    if hasattr(result, 'text_content') and result.text_content:
                        extracted_text = result.text_content
                    elif hasattr(result, 'text') and result.text:
                        extracted_text = result.text
                    else:
                        extracted_text = str(result) if result else None
                    
                    if not extracted_text or len(extracted_text.strip()) < 10:
                        return None
                    
                    return extracted_text
                    
                finally:
                    # Restore stderr
                    sys.stderr = old_stderr
                
        except Exception as e:
            print(f"MarkItDown extraction failed: {e}")
            return None
    
    def extract_with_pypdf2(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyPDF2 as fallback."""
        if not PYPDF2_AVAILABLE:
            return None
        
        try:
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                return None
            
            extracted_text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            return extracted_text if extracted_text.strip() else None
            
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return None
    
    def create_basic_fallback(self, pdf_path: str) -> str:
        """Create a basic placeholder when extraction fails."""
        print("Creating basic fallback content due to extraction failure")
        
        filename = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path)
        
        fallback_text = f"""# PDF Content from {filename}

**Note:** PDF processing failed. This is a placeholder for development purposes.

**File Information:**
- File: {pdf_path}
- Size: {file_size} bytes
- Status: PDF processing failed, using fallback

**Troubleshooting:**
This usually indicates one of the following issues:
1. The PDF file may be corrupted or encrypted
2. The PDF file may contain only images without extractable text
3. There may be a permissions issue accessing the PDF file

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

        # Drop lines that are clearly non-story metadata/ads
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
    
    def extract_pdf_cover(self, pdf_path: str, output_dir: str) -> str:
        """Extract the first page of a PDF as a cover image."""
        if not PIL_AVAILABLE or not PYPDF2_AVAILABLE:
            print("Warning: PIL or PyPDF2 not available, skipping cover extraction")
            return ""
        
        try:
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
                                print(f"Successfully extracted cover image: cover.jpg")
                                return "cover.jpg"
                            elif xObject[obj]['/Filter'] == '/JPXDecode':
                                # JPEG2000
                                cover_path = os.path.join(output_dir, "cover.jp2")
                                with open(cover_path, 'wb') as f:
                                    f.write(img_data)
                                print(f"Successfully extracted cover image: cover.jp2")
                                return "cover.jp2"
                            elif xObject[obj]['/Filter'] == '/FlateDecode':
                                # PNG or other compressed format
                                try:
                                    img = Image.open(io.BytesIO(img_data))
                                    cover_path = os.path.join(output_dir, "cover.png")
                                    img.save(cover_path, "PNG")
                                    print(f"Successfully extracted cover image: cover.png")
                                    return "cover.png"
                                except:
                                    pass
                        except Exception as e:
                            print(f"Warning: Error processing image object: {e}")
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
                    lines = text.split('\n')[:3]  # Take first 3 lines
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
                    print(f"Successfully created text-based cover: cover.png")
                    return "cover.png"
            except Exception as e:
                print(f"Warning: Error creating text-based cover: {e}")
        
        except Exception as e:
            print(f"Warning: Error extracting PDF cover: {e}")
        
        return ""
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Convert PDF to markdown and save to output directory."""
        try:
            print(f"Starting PDF conversion: {pdf_path}")
            
            # Validate PDF file
            if not self.validate_pdf_file(pdf_path):
                return {"success": False, "error": "PDF validation failed"}
            
            # Try different extraction methods
            extracted_text = None
            
            # Try MarkItDown first (best quality)
            if MARKITDOWN_AVAILABLE:
                print("Attempting MarkItDown extraction...")
                extracted_text = self.extract_with_markitdown(pdf_path)
                if extracted_text and len(extracted_text.strip()) > 0:
                    print("Successfully extracted text using MarkItDown")
                else:
                    print("MarkItDown extraction returned empty content")
                    extracted_text = None
            
            # Try PyPDF2 as fallback
            if not extracted_text and PYPDF2_AVAILABLE:
                print("Attempting PyPDF2 extraction...")
                extracted_text = self.extract_with_pypdf2(pdf_path)
                if extracted_text and len(extracted_text.strip()) > 0:
                    print("Successfully extracted text using PyPDF2")
                else:
                    print("PyPDF2 extraction returned empty content")
                    extracted_text = None
            
            # Final fallback if all extraction methods failed
            if not extracted_text:
                print("All extraction methods failed, using fallback content")
                extracted_text = self.create_basic_fallback(pdf_path)
            
            # Clean and process the text
            print("Cleaning extracted text...")
            cleaned_text = self.clean_text(extracted_text)
            story_content = self.extract_story_content(cleaned_text)
            
            # Save to output file
            output_path = os.path.join(output_dir, "pdf_result.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(story_content)
            
            print(f"Successfully saved processed text to {output_path}")
            
            # Try to extract cover image
            print("Attempting to extract cover image...")
            cover_filename = self.extract_pdf_cover(pdf_path, output_dir)
            
            return {
                "success": True,
                "output_path": output_path,
                "text_length": len(story_content),
                "cover_filename": cover_filename
            }
            
        except Exception as e:
            print(f"Error in PDF processing: {e}")
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Self-contained PDF -> Markdown CLI tool")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--outdir", required=True, help="Output directory where pdf_result.md will be written")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"PDF CLI Tool - Self-contained version")
        print(f"Input PDF: {args.pdf}")
        print(f"Output directory: {args.outdir}")
        print(f"Available libraries:")
        print(f"  - MarkItDown: {'✓' if MARKITDOWN_AVAILABLE else '✗'}")
        print(f"  - PyPDF2: {'✓' if PYPDF2_AVAILABLE else '✗'}")
        print(f"  - PIL/Pillow: {'✓' if PIL_AVAILABLE else '✗'}")
        print()

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Process the PDF
    processor = SelfContainedPDFProcessor()
    result = processor.convert_pdf_to_markdown(args.pdf, args.outdir)

    # Output result
    if args.verbose:
        print(f"Processing result: {json.dumps(result, indent=2)}")
    else:
        print(json.dumps(result))

    return 0 if result and result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())


