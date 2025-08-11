# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReadInTime (arBooks) is an Electron-based desktop application that converts PDF books into accessible audiobooks with synchronized text-to-speech and visual reading aids. The project features a complete book processing pipeline with voice cloning capabilities.

**Main Components:**
- **Frontend**: Electron + React + TypeScript desktop application (`arbooks-desktop/`)
- **Backend API**: FastAPI-based service (`backend-api/`) with integrated processing pipeline
- **Legacy Backend**: Standalone Python processors (`backend/`) - legacy individual components

## Development Commands

### Frontend Development (arbooks-desktop/)
```bash
cd arbooks-desktop
npm install                    # Install dependencies
npm run dev                    # Start development server + Electron app
npm run build                  # Build frontend for production
npm run build:electron         # Build complete Electron distribution
npm run start                  # Production mode (build frontend + start Electron)
npm run start:backend          # Start backend API server manually
```

### Backend API Development (backend-api/)
```bash
cd backend-api
python main.py                 # Start FastAPI server on localhost:8000
python validate_system.py      # Check system requirements and setup
python test_api_comprehensive.py  # Run comprehensive API tests
```

### Legacy Backend Components (backend/)
Each component has build scripts for standalone executables:
```bash
# Navigate to specific component (pdf_to_markdown, transcriptions, etc.)
./build.sh    # Linux/Mac - builds standalone executable
./build.ps1   # Windows - builds standalone executable
```

## Architecture

### Current System (Integrated Pipeline)
The application now uses a **unified FastAPI backend** (`backend-api/`) that provides:
- **Single-step processing**: Complete PDF-to-audiobook pipeline in one API call
- **Progress tracking**: Real-time status updates during processing
- **Voice cloning**: Advanced TTS with custom voice synthesis using Chatterbox
- **Multiple transcription engines**: WhisperX and legacy transcription support
- **System validation**: Automatic health checks and setup validation

### Frontend Architecture
- **Main Process**: `electron/main.ts` - IPC handlers, file operations, API communication
- **Renderer Process**: React SPA with HashRouter, communicates with main via IPC
- **State Management**: Zustand store with persistence (`src/store/useStore.ts`)
- **Pages**: Home, Library, Book (reading interface), Settings
- **Components**: Modular React components in `src/components/`
- **Global Types**: `src/global.ts` defines the Electron IPC API surface

### Backend Pipeline Architecture
**Enhanced Pipeline** (recommended - backend-api/):
1. **PDF Upload & Validation**: File upload with format validation
2. **PDF Processing**: Markitdown + PyPDF2 fallback for text extraction  
3. **TTS Generation**: Chatterbox TTS with optional voice cloning
4. **Audio Transcription**: WhisperX for word-level timestamp alignment
5. **Metadata Creation**: Complete book package with synchronized content

**Legacy Pipeline** (backend/ - individual components):
- Standalone executables for each processing step
- Manual step-by-step processing workflow
- Used for debugging or when integrated pipeline fails

### Key Data Types
Core interfaces defined in `src/types/book/index.ts`:
- `BookInfo`: Book metadata (title, author, description, folder, cover)
- `BookData`: Text mappings with timing data (`TextSegment[]`) 
- `BookUploadData`: Complete upload request structure
- `TextSegment`: Time-aligned text segments (`text`, `start`, `end`)
- `VoiceCloningOptions`: Voice cloning configuration

## Technology Stack

### Frontend
- **Framework**: Electron 33 + React 18 + TypeScript
- **Styling**: Tailwind CSS + DaisyUI components
- **State Management**: Zustand with persistence middleware
- **Routing**: React Router with HashRouter (Electron-compatible)
- **Audio Processing**: Howler.js + react-use-audio-player for playback
- **Book Reading**: react-pageflip for page turning animations
- **Build Tool**: Vite with Electron plugin

### Backend API (Primary)
- **Framework**: FastAPI with uvicorn server
- **Language**: Python 3.11+ with virtual environments
- **PDF Processing**: Markitdown (Microsoft) + PyPDF2 fallback
- **TTS Engine**: Chatterbox (S3-GEN model) with voice cloning
- **Transcription**: WhisperX for word-level alignment
- **Audio Processing**: librosa, soundfile for audio manipulation
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

### Legacy Backend Components
- **Compilation**: PyInstaller for standalone executables
- **Virtual Environments**: Separate venvs for each component (pdf, tts, whisperx)
- **Output**: Executables bundled in `arbooks-desktop/bin/`

## File Structure Patterns

### Frontend Conventions
- **Components**: Organized by feature (`book/`, `nav/`, `settings/`, `voice/`)
- **Hooks**: Custom React hooks in `src/hooks/` for reusable logic
- **Types**: Centralized TypeScript interfaces in `src/types/`
- **Global State**: Zustand store with persistence for app settings
- **IPC Communication**: Type-safe Electron API defined in `src/global.ts`

### Backend Structure
- **Services**: Modular services in `backend-api/services/` (PDF, TTS, transcription)
- **Models**: Pydantic models in `backend-api/models/` for request/response types
- **Multiple Environments**: Separate venvs for different processing components
- **Configuration**: Environment-specific settings and validation

## API Integration

### Key Endpoints
- `POST /api/books/process-complete` - Complete pipeline processing with progress tracking
- `GET /api/books/processing-status/{job_id}` - Real-time processing status
- `POST /api/tts/voice-clone` - Voice cloning TTS generation
- `POST /api/transcribe/whisperx` - Direct WhisperX transcription
- `GET /health` - System health with service availability

### IPC Bridge Pattern
The Electron main process acts as a bridge between the React frontend and the FastAPI backend:
1. React components call `window.electron.methodName()`
2. Main process receives IPC call and makes HTTP request to FastAPI
3. Response is returned through the IPC channel back to React

## Development Notes

### Environment Setup
- **Frontend**: Runs on port 5173 (Vite dev server)
- **Backend API**: Runs on port 8000 (FastAPI/uvicorn)
- **Electron**: Main process automatically starts backend API on app launch
- **Multiple Python Environments**: Each processing component uses separate venv

### Build Process
- **Development**: `npm run dev` starts both Vite and Electron
- **Production**: Vite builds to `dist-electron/`, executables copied from `bin/`
- **Distribution**: Electron-builder creates portable Windows executable
- **Asset Management**: Custom Vite plugin copies backend executables to dist

### Accessibility Features
- **Fonts**: OpenDyslexic font family bundled for dyslexic readers
- **Voice Settings**: Configurable TTS with voice cloning options
- **Visual Options**: Multiple color themes and font size controls
- **Reading Modes**: Speech-only, text-only, or combined modes

## Linting and Type Checking

```bash
cd arbooks-desktop
npx eslint .                   # Run ESLint with TypeScript rules
npx tsc --noEmit              # Type checking without compilation
npm run build:ts:electron     # Type check Electron main process
```

**ESLint Configuration**: TypeScript ESLint with React hooks and React refresh plugins

## System Requirements and Setup

### Virtual Environment Setup
The application requires multiple Python virtual environments:
- `backend-api/venvs/pdf/` - PDF processing (markitdown, PyPDF2)
- `backend-api/venvs/tts/` - TTS generation (chatterbox, torch)
- `backend-api/venvs/whisperx/` - Audio transcription (whisperx, transformers)

### Validation Commands
```bash
cd backend-api
python validate_system.py     # Check all system requirements
python test_chatterbox.py     # Test TTS functionality  
python test_whisperx.py       # Test transcription
```