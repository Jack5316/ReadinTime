import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { get as getAppRootDir } from "app-root-dir";
import path from 'path';
import dotenv from "dotenv";
import { readFileSync } from "fs";
import { BookInfo, BookUploadData } from "../src/types/book"
import { Result } from "../src/types/result"
import { spawn, ChildProcess } from "child_process";
import http from 'http';

dotenv.config();

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  console.log("Another instance is already running. Exiting...");
  app.quit();
} else {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    // Someone tried to run a second instance, we should focus our window instead
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

const DEV_VERBOSE = process.env.DEV_VERBOSE === '1';
function devLog(...args: any[]) { if (DEV_VERBOSE) console.log('[DEV]', ...args); }
devLog("DIRNAME:", __dirname);
devLog("ROOT DIR:", getAppRootDir());
devLog("PLATFORM:", process.platform);

const VITE_PORT = 5173;
const API_PORT = 8000;
const API_BASE_URL = `http://127.0.0.1:${API_PORT}`;

let mainWindow: BrowserWindow | null = null;
let apiServerProcess: ChildProcess | null = null;
let apiServerStarted = false;
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// Check if Vite dev server is running
async function isViteServerRunning(): Promise<boolean> {
  try {
    return new Promise((resolve) => {
      const req = http.get(`http://localhost:${VITE_PORT}`, {
        timeout: 1000,
      }, (res) => {
        resolve(res.statusCode === 200);
      });
      
      req.on('error', () => {
        resolve(false);
      });
      
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
    });
  } catch (error) {
    return false;
  }
}

// Check if a port is in use
async function isPortInUse(port: number): Promise<boolean> {
  const net = await import('net');
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(port, '127.0.0.1', () => {
      server.once('close', () => resolve(false));
      server.close();
    });
    server.on('error', (err: any) => {
      if (err.code === 'EADDRINUSE') {
        resolve(true);
      } else {
        resolve(false);
      }
    });
  });
}

// Health check function
async function checkApiServerHealth(silent: boolean = false): Promise<boolean> {
  try {
    // Skip health check if backend is disabled
    if (process.env.SKIP_BACKEND === 'true') {
      if (!silent) console.log('Skipping health check - backend disabled');
      return false;
    }

    const endpoints = [
      'http://127.0.0.1:8000/ping',
      'http://127.0.0.1:8000/health',
      'http://127.0.0.1:8000/'
    ];

    const tryEndpoint = (url: string) => new Promise<boolean>((resolve) => {
      const req = http.get(url, { timeout: 3000, headers: { 'Accept': 'application/json' } }, (res) => {
        const isOk = res.statusCode === 200;
        if (!silent) console.log(`[Backend] Health check ${url}: ${isOk ? 'OK' : 'Failed'} (${res.statusCode})`);
        resolve(isOk);
      });
      req.on('error', () => resolve(false));
      req.on('timeout', () => { req.destroy(); resolve(false); });
    });

    for (const url of endpoints) {
      if (await tryEndpoint(url)) return true;
    }
    return false;
  } catch (error) {
    if (!silent) {
      console.log(`Health check exception: ${error}`);
    }
    return false;
  }
}

async function createWindow() {
  try {
    mainWindow = new BrowserWindow({
      width: 800,
      height: 600,
      backgroundColor: '#f8f4e6',
      useContentSize: true,
      webPreferences: {
        preload: path.join(__dirname, "preload.js"),
        nodeIntegration: false,
        contextIsolation: true,
      },
    });

    mainWindow.webContents.on('before-input-event', (event, input) => {
    if (!mainWindow) return;
    
    if (input.control || input.meta) {
      switch (input.key.toLowerCase()) {
        case 'i':
          if (input.shift) mainWindow.webContents.toggleDevTools()
          break
        case 'r':
          mainWindow.webContents.reload()
          break
        case '+':
        case '=':
          mainWindow.webContents.setZoomLevel(mainWindow.webContents.getZoomLevel() + 0.5)
          break
        case '-':
          mainWindow.webContents.setZoomLevel(mainWindow.webContents.getZoomLevel() - 0.5)
          break
        case '0':
          mainWindow.webContents.setZoomLevel(0)
          break
      }
    } else if (input.key.toLowerCase() === 'f11') {
      mainWindow.setFullScreen(!mainWindow.isFullScreen())
    } else if (input.key.toLowerCase() === 'escape') {
      // ESC key exits fullscreen mode
      if (mainWindow.isFullScreen()) {
        mainWindow.setFullScreen(false)
      }
    }
    // Ctrl+Alt+F toggles true borderless fullscreen (kiosk)
    if ((input.control || input.meta) && input.alt && input.key.toLowerCase() === 'f') {
      const isKiosk = mainWindow.isKiosk();
      mainWindow.setKiosk(!isKiosk);
      if (!isKiosk) {
        mainWindow.setMenuBarVisibility(false);
        mainWindow.setAutoHideMenuBar(true);
      }
    }
  })

    // Check if Vite dev server is running and load accordingly
    const viteRunning = await isViteServerRunning();
    if (viteRunning) {
      const devUrl = `http://localhost:${VITE_PORT}`;
      console.log("Loading from Vite dev server:", devUrl);
      mainWindow.loadURL(devUrl);
    } else {
      const indexPath = path.join(__dirname, 'index.html');
      console.log("Loading from built files:", indexPath);
      mainWindow.loadFile(indexPath);
    }
    
    // Open dev tools in development
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }

    mainWindow.setMenu(null);
    
    mainWindow.on('closed', () => {
      mainWindow = null;
    });
    
    console.log("Created window!");
  } catch (error) {
    console.error("Error creating window:", error);
  }
}

async function startApiServer() {
  try {
    // Skip backend startup if environment variable is set
    if (process.env.SKIP_BACKEND === 'true') {
      console.log("Skipping backend startup due to SKIP_BACKEND environment variable");
      return;
    }
    
    if (apiServerStarted) {
      console.log("API server startup already initiated, skipping...");
      return;
    }
    
    devLog("Attempting to start API server...");
    
    // First, try to check if an API server is already running and healthy
    const isHealthy = await checkApiServerHealth(true);
    if (isHealthy) {
      console.log("[Backend] Using existing server (healthy)");
      apiServerStarted = true;
      return;
    }
    
    const portInUse = await isPortInUse(API_PORT);
    devLog(`Port ${API_PORT} in use check:`, portInUse);
    if (portInUse) {
      console.log("[Backend] Port in use; assuming server is running. Skipping start.");
      apiServerStarted = true;
      return;
    }
    
    console.log("[Backend] Starting dev server...");
    apiServerStarted = true;
    
    const backendPath = isDev
      ? path.join(getAppRootDir(), "..", "backend-api")
      : path.join(process.resourcesPath, "backend-api");

    // Prefer Python venv in development; compiled exe only in production or when forced
    const fs = await import("fs/promises");
    const exists = async (p: string) => !!(await fs.access(p).then(() => true).catch(() => false));

    const preferPythonDev = isDev && process.env.FORCE_COMPILED_BACKEND !== 'true';

    const mainScript = path.join(backendPath, "main.py");
    const pythonPath = process.platform === "win32"
      ? path.join(backendPath, "venv-unified", "Scripts", "python.exe")
      : path.join(backendPath, "venv-unified", "bin", "python");

    let resolvedPythonPath = pythonPath;
    let resolvedMainScript = mainScript;

    if (preferPythonDev) {
      // Try dev python first
      if (!(await exists(resolvedPythonPath)) || !(await exists(resolvedMainScript))) {
        const devBackend = path.join(getAppRootDir(), "..", "backend-api");
        const devPython = process.platform === "win32"
          ? path.join(devBackend, "venv-unified", "Scripts", "python.exe")
          : path.join(devBackend, "venv-unified", "bin", "python");
        const devScript = path.join(devBackend, "main.py");
        if (await exists(devPython) && await exists(devScript)) {
          console.warn("Falling back to dev backend path", devBackend);
          resolvedPythonPath = devPython;
          resolvedMainScript = devScript;
        }
      }

      if (await exists(resolvedPythonPath) && await exists(resolvedMainScript)) {
        devLog("Using Python backend (dev mode):", resolvedPythonPath, resolvedMainScript);
        apiServerProcess = spawn(resolvedPythonPath, [resolvedMainScript], {
          cwd: path.dirname(resolvedMainScript),
          stdio: ['ignore', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUNBUFFERED: '1', LOG_LEVEL: process.env.BACKEND_LOG_LEVEL || 'WARNING' }
        });
      } else {
        console.warn("Python backend not found; attempting to use compiled backend instead.");
      }
    }

    if (!apiServerProcess) {
      // Use compiled backend executable
      let backendExecutableUnified: string;
      let backendExecutableClassic: string;
      if (process.platform === "win32") {
        backendExecutableUnified = path.join(backendPath, "dist-unified", "arBooks_Backend_Unified.exe");
        backendExecutableClassic = path.join(backendPath, "dist", "arBooks_Backend.exe");
      } else {
        backendExecutableUnified = path.join(backendPath, "dist-unified", "arBooks_Backend_Unified");
        backendExecutableClassic = path.join(backendPath, "dist", "arBooks_Backend");
      }

      const backendExecutable = (await exists(backendExecutableUnified)) ? backendExecutableUnified : backendExecutableClassic;

      if (await exists(backendExecutable)) {
        devLog("Using compiled backend executable:", backendExecutable);
        apiServerProcess = spawn(backendExecutable, [], {
          cwd: backendPath,
          stdio: ['ignore', 'pipe', 'pipe']
        });
      } else if (!preferPythonDev) {
        // As a last resort, try python even in prod if compiled not found (dev unpacked run)
        if (await exists(resolvedPythonPath) && await exists(resolvedMainScript)) {
          devLog("Compiled backend not found, falling back to Python mode:", resolvedPythonPath);
          apiServerProcess = spawn(resolvedPythonPath, [resolvedMainScript], {
            cwd: path.dirname(resolvedMainScript),
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, PYTHONUNBUFFERED: '1', LOG_LEVEL: process.env.BACKEND_LOG_LEVEL || 'WARNING' }
          });
        } else {
          throw new Error(`Backend not found. Looked for compiled exe and Python at ${resolvedPythonPath}`);
        }
      }
    }

    const printLimited = (prefix: string, buf: Buffer) => {
      const msg = buf.toString('utf8').trim();
      if (!msg) return;
      const lines = msg.split(/\r?\n/);
      const head = lines.slice(0, 3).join('\n');
      const extra = lines.length > 3 ? `\n... (${lines.length - 3} more)` : '';
      console.log(`${prefix} ${head}${extra}`);
    };
    if (apiServerProcess) {
      apiServerProcess.stdout?.on('data', (d) => printLimited('[Backend]', d as Buffer));
      apiServerProcess.stderr?.on('data', (d) => printLimited('[Backend:ERR]', d as Buffer));
      apiServerProcess.on('close', (code) => {
        console.log(`API server exited with code ${code}`);
        apiServerStarted = false;
      });
    }

    // Wait for the server to actually start and become responsive
    devLog("Waiting for API server to become responsive...");
    let retries = 6; // Reduced from 10 to 6
    let serverReady = false;
    
    while (retries > 0 && !serverReady) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Increased from 1000ms to 2000ms
      serverReady = await checkApiServerHealth(true);
      if (!serverReady) {
        retries--;
        devLog(`API server not ready yet, ${retries} retries remaining...`);
      }
    }
    
    if (serverReady) {
      console.log("[Backend] Ready");
    } else {
      console.log("[Backend] Started (not fully responsive yet)");
      // Don't treat this as a failure - the server might still be initializing services
    }

  } catch (error) {
    console.error("Error starting API server:", error);
    apiServerStarted = false;
  }
}

// API helper functions
async function apiRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request to ${endpoint} failed:`, error);
    throw error;
  }
}

async function apiFormRequest(endpoint: string, formData: FormData): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request to ${endpoint} failed:`, error);
    throw error;
  }
}

app.whenReady().then(async () => {
  // Start the backend API server
  await startApiServer();

  // Do not block the app on backend health in development; monitor in background
  if (isDev) {
    (async () => {
      let retries = 15;
      while (retries > 0) {
        const ok = await checkApiServerHealth(true);
        if (ok) { console.log('[Backend] Ready'); break; }
        await new Promise(r => setTimeout(r, 2000));
        retries -= 1;
      }
      if (retries === 0) console.warn('[Backend] Still not healthy after background retries; app will continue.');
    })();
  } else {
    // In production, wait a bit and warn instead of quitting immediately
    let healthy = await checkApiServerHealth();
    let retries = 8;
    while (!healthy && retries > 0) {
      await new Promise(r => setTimeout(r, 2000));
      healthy = await checkApiServerHealth();
      retries -= 1;
    }
    if (!healthy) {
      console.error('API server appears unavailable, continuing to load UI. Some features may not work until backend becomes ready.');
    }
  }

  ipcMain.handle('select-directory', async () => {
    try {
      if (!mainWindow) {
        return { success: false, error: 'Main window not available' };
      }
      
      if (!mainWindow.isFocused()) {
        mainWindow.focus();
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Select Books Directory'
      });

      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, error: 'No directory selected' };
      }

      return { success: true, result: result.filePaths[0] };
    } catch (error) {
      console.error("Error selecting directory:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  // Toggle kiosk (true borderless fullscreen) from renderer
  ipcMain.handle('toggle-kiosk', async () => {
    if (!mainWindow) return { success: false };
    const isKiosk = mainWindow.isKiosk();
    mainWindow.setKiosk(!isKiosk);
    if (!isKiosk) {
      mainWindow.setMenuBarVisibility(false);
      mainWindow.setAutoHideMenuBar(true);
    }
    return { success: true, kiosk: !isKiosk };
  });

  ipcMain.handle("list-books", async (event: any, directoryPath: string) => {
    try {
      const formData = new FormData();
      formData.append('directory_path', directoryPath);

      const result = await apiFormRequest('/api/books/list', formData);
      return result;
    } catch (error) {
      console.error("Error listing books:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("read-book", async (event: any, bookPath: string) => {
    try {
      const formData = new FormData();
      formData.append('book_path', bookPath);

      const result = await apiFormRequest('/api/books/read', formData);
      return result;
    } catch (error) {
      console.error("Error reading book:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("get-file-data", async (event: any, filePath: string) => {
    try {
      if (!filePath || filePath.trim() === '') {
        console.log("Skipping empty file path request");
        return { success: false, error: "File path is empty" };
      }

      console.log(`Requesting file data for: ${filePath}`);
      const encodedPath = encodeURIComponent(filePath);
      const requestUrl = `${API_BASE_URL}/api/files/data?file_path=${encodedPath}`;
      console.log(`Request URL: ${requestUrl}`);

      const response = await fetch(requestUrl);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API Error (${response.status}): ${errorText}`);
        throw new Error(`Failed to get file data: ${response.statusText} - ${errorText}`);
      }

      const buffer = await response.arrayBuffer();
      console.log(`Successfully loaded file data: ${buffer.byteLength} bytes`);
      return { success: true, result: Buffer.from(buffer) };
    } catch (error) {
      console.error("Error getting file data:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("validate-system", async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/books/validate-system`);
      
      if (!response.ok) {
        throw new Error(`System validation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result; // Return the full Result object from the API
    } catch (error) {
      console.error("Error validating system:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("get-processing-status", async (event: any, jobId: string) => {
    try {
      console.log(`[ELECTRON] Getting processing status for job: ${jobId}`);
      const url = `${API_BASE_URL}/api/books/processing-status/${jobId}`;
      console.log(`[ELECTRON] Fetching from: ${url}`);
      
      const response = await fetch(url);
      console.log(`[ELECTRON] Response status: ${response.status}`);
      
      if (!response.ok) {
        throw new Error(`Failed to get processing status: ${response.statusText}`);
      }

      const result = await response.json();
      console.log(`[ELECTRON] Processing status result:`, result);
      return result; // Return the full Result object from the API
    } catch (error) {
      console.error("[ELECTRON] Error getting processing status:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("process-book-complete", async (event: any, bookData: any) => {
    try {
      console.log("[ELECTRON] Starting enhanced book processing...");
      console.log("[ELECTRON] Received bookData:", JSON.stringify(bookData, null, 2));
      
      // The frontend is sending:
      // - name: filename
      // - pdfData: file content (ArrayBuffer or similar)
      // - title, author, description, bookPath
      
      const fileName = bookData.name;
      const pdfData = bookData.pdfData;
      const title = bookData.title;
      const author = bookData.author;
      const description = bookData.description || '';
      const bookPath = bookData.bookPath;
      const voiceCloning = bookData.voiceCloning;
      
      console.log("Extracted values:", { fileName, title, author, bookPath, hasPdfData: !!pdfData, voiceCloning });
      
      if (!fileName || !pdfData || !title || !author || !bookPath) {
        const missing = [];
        if (!fileName) missing.push('name (filename)');
        if (!pdfData) missing.push('pdfData');
        if (!title) missing.push('title');
        if (!author) missing.push('author');
        if (!bookPath) missing.push('bookPath');
        
        console.error("Missing required data:", missing);
        console.error("Received bookData keys:", Object.keys(bookData));
        console.error("BookData values:", {
          name: bookData.name,
          title: bookData.title, 
          author: bookData.author,
          bookPath: bookData.bookPath,
          hasPdfData: !!bookData.pdfData
        });
        
        throw new Error(`Missing required book data: ${missing.join(', ')}. Received keys: ${Object.keys(bookData).join(', ')}`);
      }

      const formData = new FormData();
      
      // Convert pdfData to a Blob and add it to form data
      let fileBlob;
      if (pdfData instanceof ArrayBuffer) {
        fileBlob = new Blob([pdfData], { type: 'application/pdf' });
      } else if (pdfData instanceof Uint8Array) {
        fileBlob = new Blob([pdfData], { type: 'application/pdf' });
      } else {
        // If it's already a Blob or File, use it directly
        fileBlob = pdfData;
      }
      
      // Ensure we have a valid filename, use fallback if needed
      const safeFileName = fileName || 'unknown_file.pdf';
      
      formData.append('file', fileBlob, safeFileName);
      formData.append('title', title);
      formData.append('author', author);
      formData.append('description', description);
      formData.append('book_path', bookPath);
      
      // Add voice cloning options if available
      if (voiceCloning && voiceCloning.enabled) {
        console.log("[ELECTRON] Adding voice cloning options to request:", voiceCloning);
        formData.append('voice_cloning_enabled', 'true');
        formData.append('voice_cloning_mode', voiceCloning.mode || 'none');
        formData.append('exaggeration', voiceCloning.exaggeration?.toString() || '0.5');
        formData.append('cfg_weight', voiceCloning.cfgWeight?.toString() || '0.5');
        
        if (voiceCloning.mode === 'settings_sample' && voiceCloning.selectedSampleId) {
          formData.append('voice_sample_id', voiceCloning.selectedSampleId);
        } else if (voiceCloning.mode === 'direct_upload' && voiceCloning.voicePromptFile) {
          // Add the voice prompt file for direct upload
          let voiceFileBlob;
          if (voiceCloning.voicePromptFile instanceof ArrayBuffer) {
            voiceFileBlob = new Blob([voiceCloning.voicePromptFile], { type: 'audio/wav' });
          } else if (voiceCloning.voicePromptFile instanceof File) {
            voiceFileBlob = voiceCloning.voicePromptFile;
          } else {
            voiceFileBlob = voiceCloning.voicePromptFile;
          }
          formData.append('voice_prompt_file', voiceFileBlob, 'voice_prompt.wav');
        }
      } else {
        formData.append('voice_cloning_enabled', 'false');
        formData.append('voice_cloning_mode', 'none');
      }

      console.log("Sending request to API with file:", safeFileName);

      const response = await fetch(`${API_BASE_URL}/api/books/process-complete`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error Response:", errorText);
        throw new Error(`Book processing failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] API response:", result);
      console.log("[ELECTRON] API response keys:", Object.keys(result));
      console.log("[ELECTRON] API response.result:", result.result);
      console.log("[ELECTRON] API response.result keys:", Object.keys(result.result || {}));
      console.log("[ELECTRON] API response.result.job_id:", result.result?.job_id);
      
      console.log("[ELECTRON] Returning to frontend:", { success: true, result });
      return { success: true, result };
    } catch (error) {
      console.error("[ELECTRON] Error processing book:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  // Voice cloning IPC handler
  ipcMain.handle("generate-voice-cloned-speech", async (event: any, text: string, voicePromptFile: ArrayBuffer, exaggeration?: number, cfgWeight?: number) => {
    try {
      console.log("[ELECTRON] Starting voice cloned speech generation...");
      console.log("[ELECTRON] Text length:", text.length);
      console.log("[ELECTRON] Voice prompt file size:", voicePromptFile.byteLength);
      console.log("[ELECTRON] Exaggeration:", exaggeration);
      console.log("[ELECTRON] CFG Weight:", cfgWeight);

      if (!text || !voicePromptFile) {
        throw new Error("Missing required parameters: text and voicePromptFile are required");
      }

      const formData = new FormData();
      
      // Convert ArrayBuffer to Blob for the audio file
      const audioBlob = new Blob([voicePromptFile], { type: 'audio/wav' });
      
      formData.append('text', text);
      formData.append('audio_prompt_file', audioBlob, 'voice_prompt.wav');
      formData.append('exaggeration', (exaggeration ?? 0.5).toString());
      formData.append('cfg_weight', (cfgWeight ?? 0.5).toString());

      console.log("[ELECTRON] Sending voice cloning request to API...");

      const response = await fetch(`${API_BASE_URL}/api/tts/voice-clone`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] Voice cloning API Error Response:", errorText);
        throw new Error(`Voice cloning failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] Voice cloning API response:", result);
      
      return result;
    } catch (error) {
      console.error("[ELECTRON] Error generating voice cloned speech:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  // Regular TTS handler - uses voice cloning when enabled and voice sample is available
  ipcMain.handle("generate-tts", async (event: any, bookData: BookUploadData, voiceCloningOptions?: { enabled: boolean; selectedSampleId: string | null; exaggeration: number; cfgWeight: number }) => {
    try {
      console.log("[ELECTRON] Starting TTS generation for book...");
      console.log("[ELECTRON] Book data:", bookData);
      console.log("[ELECTRON] Voice cloning options:", voiceCloningOptions);

      const formData = new FormData();
      formData.append('filename', bookData.name);
      formData.append('book_path', bookData.bookPath);

      let endpoint = '/api/books/generate-tts';
      
      // Check if voice cloning should be used
      if (voiceCloningOptions?.enabled && voiceCloningOptions.selectedSampleId) {
        console.log("[ELECTRON] Using voice cloning for TTS generation");
        endpoint = '/api/books/generate-voice-cloned-tts';
        
        // Get the voice sample file and add it to the form data
        try {
          const voiceSampleResponse = await fetch(`${API_BASE_URL}/api/voice-samples/${voiceCloningOptions.selectedSampleId}/file`);
          if (voiceSampleResponse.ok) {
            const voiceSampleBlob = await voiceSampleResponse.blob();
            formData.append('voice_prompt_file', voiceSampleBlob, 'voice_prompt.wav');
            formData.append('exaggeration', voiceCloningOptions.exaggeration.toString());
            formData.append('cfg_weight', voiceCloningOptions.cfgWeight.toString());
          } else {
            console.warn("[ELECTRON] Failed to fetch voice sample, falling back to regular TTS");
            endpoint = '/api/books/generate-tts';
          }
        } catch (error) {
          console.warn("[ELECTRON] Error fetching voice sample, falling back to regular TTS:", error);
          endpoint = '/api/books/generate-tts';
        }
      } else {
        console.log("[ELECTRON] Using regular TTS generation");
      }

      console.log("[ELECTRON] Sending TTS request to API endpoint:", endpoint);

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] TTS API Error Response:", errorText);
        throw new Error(`TTS generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] TTS API response:", result);
      
      return result;
    } catch (error) {
      console.error("[ELECTRON] Error generating TTS:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  // Voice sample management IPC handlers
  ipcMain.handle("upload-voice-sample", async (event: any, file: ArrayBuffer, name: string, sampleId: string, fileName: string) => {
    try {
      console.log("[ELECTRON] Uploading voice sample:", name, fileName);

      const formData = new FormData();
      const fileBlob = new Blob([file], { type: 'audio/wav' });
      
      formData.append('file', fileBlob, fileName);
      formData.append('name', name);
      formData.append('sample_id', sampleId);

      const response = await fetch(`${API_BASE_URL}/api/voice-samples/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] Voice sample upload error:", errorText);
        throw new Error(`Voice sample upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] Voice sample uploaded:", result);
      
      return result;
    } catch (error) {
      console.error("[ELECTRON] Error uploading voice sample:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });



  ipcMain.handle("get-voice-samples-directory", async (event: any) => {
    try {
      console.log("[ELECTRON] Getting voice samples directory");

      const response = await fetch(`${API_BASE_URL}/api/voice-samples/directory`, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] Get voice samples directory error:", errorText);
        throw new Error(`Failed to get voice samples directory: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] Voice samples directory:", result);
      
      // Extract directory string from response
      const directory = result.directory || (typeof result === 'string' ? result : null);
      
      if (directory) {
        return { success: true, result: directory };
      } else {
        throw new Error('Invalid directory format in API response');
      }
    } catch (error) {
      console.error("[ELECTRON] Error getting voice samples directory:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("set-voice-samples-directory", async (event: any, directory: string) => {
    try {
      console.log("[ELECTRON] Setting voice samples directory:", directory);

      const formData = new FormData();
      formData.append('directory', directory);

      const response = await fetch(`${API_BASE_URL}/api/voice-samples/directory`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] Set voice samples directory error:", errorText);
        throw new Error(`Failed to set voice samples directory: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] Voice samples directory set:", result);
      
      return result;
    } catch (error) {
      console.error("[ELECTRON] Error setting voice samples directory:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("delete-voice-sample", async (event: any, sampleId: string) => {
    try {
      console.log("[ELECTRON] Deleting voice sample:", sampleId);

      const response = await fetch(`${API_BASE_URL}/api/voice-samples/${sampleId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[ELECTRON] Voice sample delete error:", errorText);
        throw new Error(`Voice sample delete failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[ELECTRON] Voice sample deleted:", result);
      
      return result;
    } catch (error) {
      console.error("[ELECTRON] Error deleting voice sample:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("get-voice-sample-url", async (event: any, sampleId: string) => {
    try {
      // Return URL for voice sample file access
      const url = `${API_BASE_URL}/api/voice-samples/${sampleId}/file`;
      return { success: true, result: url };
    } catch (error) {
      console.error("[ELECTRON] Error getting voice sample URL:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  await createWindow();

  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (apiServerProcess) {
    console.log("Terminating API server process...");
    apiServerProcess.kill('SIGTERM');
    // Force kill after 5 seconds if it doesn't exit gracefully
    setTimeout(() => {
      if (apiServerProcess && !apiServerProcess.killed) {
        console.log("Force killing API server process...");
        apiServerProcess.kill('SIGKILL');
      }
    }, 5000);
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (apiServerProcess) {
    console.log("Terminating API server process before quit...");
    apiServerProcess.kill('SIGTERM');
  }
});
