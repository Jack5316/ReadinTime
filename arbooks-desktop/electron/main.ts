import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { get as getAppRootDir } from "app-root-dir";
import path from 'path';
import dotenv from "dotenv";
import { readFileSync } from "fs";
import type { BookInfo, BookUploadData } from "../src/types/book"
import type { Result } from "../src/types/result"
import { spawn } from "child_process";
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
const FORCE_OFFLINE = process.env.ELECTRON_OFFLINE === '1';
// Offline-only mode: all operations run locally via IPC/CLI; no localhost HTTP

// Voice samples directory state (persisted to userData)
let voiceSamplesDir: string | null = null;
function getDefaultVoiceSamplesDir() {
  return path.join(app.getPath('userData'), 'voice-samples');
}
function getVoiceSamplesConfigPath() {
  return path.join(app.getPath('userData'), 'voice-samples-config.json');
}
async function loadVoiceSamplesDirFromConfig(): Promise<string> {
  const fs = await import('fs/promises');
  try {
    const cfgPath = getVoiceSamplesConfigPath();
    const buf = await fs.readFile(cfgPath, { encoding: 'utf-8' });
    const cfg = JSON.parse(buf || '{}');
    const dir = typeof cfg.directory === 'string' && cfg.directory.trim() ? cfg.directory : getDefaultVoiceSamplesDir();
    await fs.mkdir(dir, { recursive: true });
    voiceSamplesDir = dir;
    return dir;
  } catch {
    const dir = getDefaultVoiceSamplesDir();
    try { await fs.mkdir(dir, { recursive: true }); } catch {}
    voiceSamplesDir = dir;
    return dir;
  }
}
async function saveVoiceSamplesDirToConfig(dir: string): Promise<void> {
  const fs = await import('fs/promises');
  const cfgPath = getVoiceSamplesConfigPath();
  const cfg = { directory: dir } as any;
  await fs.mkdir(path.dirname(cfgPath), { recursive: true });
  await fs.writeFile(cfgPath, JSON.stringify(cfg, null, 2), { encoding: 'utf-8' });
}

let mainWindow: BrowserWindow | null = null;
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// Prevent rapid F11 toggling
let lastF11Toggle = 0;
const F11_DEBOUNCE_MS = 300; // Minimum time between F11 toggles

// Offline job tracking (NO_LOCALHOST): mirror backend active_jobs for progress polling
type JobStatus = {
  stage: string;
  progress: number;
  message: string;
  status: 'processing' | 'completed' | 'failed';
  filename?: string;
  title?: string;
  files_created?: Record<string, boolean>;
};
const activeJobs: Record<string, JobStatus> = {};

// Split text into manageable chunks for TTS (sentence-based, approx maxChars per chunk)
function splitTextIntoChunks(text: string, maxChars: number = 300): string[] {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks: string[] = [];
  let current = '';
  for (const s of sentences) {
    if ((current + (current ? ' ' : '') + s).length > maxChars && current) {
      chunks.push(current.trim());
      current = s;
    } else {
      current = current ? current + ' ' + s : s;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks.length ? chunks : [text];
}

// Minimal WAV concatenation (PCM/IEEE float) assuming consistent format across inputs
async function combineWavs(inputFiles: string[], outputFile: string): Promise<void> {
  const fs = await import('fs/promises');

  type WavInfo = {
    format: number; // 1=PCM, 3=IEEE float
    channels: number;
    sampleRate: number;
    bitsPerSample: number;
    dataOffset: number;
    dataLength: number;
    buffer: Buffer;
  };

  function readUInt32LE(buf: Buffer, off: number) { return buf.readUInt32LE(off); }
  function readUInt16LE(buf: Buffer, off: number) { return buf.readUInt16LE(off); }

  function parseWav(buf: Buffer): WavInfo {
    if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error('Not a RIFF/WAVE file');
    }
    let pos = 12;
    let fmtFound = false;
    let dataFound = false;
    let format = 1;
    let channels = 1;
    let sampleRate = 16000;
    let bitsPerSample = 16;
    let dataOffset = 0;
    let dataLength = 0;
    while (pos + 8 <= buf.length) {
      const id = buf.toString('ascii', pos, pos + 4);
      const size = readUInt32LE(buf, pos + 4);
      const chunkStart = pos + 8;
      if (id === 'fmt ') {
        format = readUInt16LE(buf, chunkStart + 0);
        channels = readUInt16LE(buf, chunkStart + 2);
        sampleRate = readUInt32LE(buf, chunkStart + 4);
        bitsPerSample = readUInt16LE(buf, chunkStart + 14);
        fmtFound = true;
      } else if (id === 'data') {
        dataOffset = chunkStart;
        dataLength = size;
        dataFound = true;
      }
      pos = chunkStart + size + (size % 2); // pad to even
    }
    if (!fmtFound || !dataFound) throw new Error('Invalid WAV: missing fmt or data');
    return { format, channels, sampleRate, bitsPerSample, dataOffset, dataLength, buffer: buf };
  }

  const infos: WavInfo[] = [];
  for (const f of inputFiles) {
    const buf = Buffer.from(await fs.readFile(f));
    const info = parseWav(buf);
    infos.push(info);
  }

  const first = infos[0];
  for (const i of infos.slice(1)) {
    if (i.format !== first.format || i.channels !== first.channels || i.sampleRate !== first.sampleRate || i.bitsPerSample !== first.bitsPerSample) {
      throw new Error('WAV format mismatch between chunks');
    }
  }

  const totalData = infos.reduce((acc, i) => acc + i.dataLength, 0);
  const fmtChunkSize = 16; // PCM/IEEE float
  const riffChunkSize = 4 + (8 + fmtChunkSize) + (8 + totalData); // 'WAVE' + fmt + data

  const header = Buffer.alloc(12 + 8 + fmtChunkSize + 8);
  header.write('RIFF', 0, 4, 'ascii');
  header.writeUInt32LE(riffChunkSize, 4);
  header.write('WAVE', 8, 4, 'ascii');
  header.write('fmt ', 12, 4, 'ascii');
  header.writeUInt32LE(fmtChunkSize, 16);
  header.writeUInt16LE(first.format, 20); // audioFormat
  header.writeUInt16LE(first.channels, 22); // numChannels
  header.writeUInt32LE(first.sampleRate, 24); // sampleRate
  const byteRate = first.sampleRate * first.channels * (first.bitsPerSample / 8);
  header.writeUInt32LE(byteRate, 28);
  const blockAlign = first.channels * (first.bitsPerSample / 8);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(first.bitsPerSample, 34);
  header.write('data', 36, 4, 'ascii');
  header.writeUInt32LE(totalData, 40);

  const pcmParts = infos.map(i => i.buffer.subarray(i.dataOffset, i.dataOffset + i.dataLength));
  await fs.mkdir(path.dirname(outputFile), { recursive: true });
  await fs.writeFile(outputFile, Buffer.concat([header, ...pcmParts]));
}

// Check if Vite dev server is running
async function isViteServerRunning(): Promise<boolean> {
  try {
    return new Promise((resolve) => {
      const req = http.get(`http://localhost:${VITE_PORT}`, {
        timeout: 3000, // Increased timeout
      }, (res) => {
        console.log(`Vite server response: ${res.statusCode}`);
        resolve(res.statusCode === 200);
      });
      
      req.on('error', (err) => {
        console.log(`Vite server check error: ${err.message}`);
        resolve(false);
      });
      
      req.on('timeout', () => {
        console.log('Vite server check timeout');
        req.destroy();
        resolve(false);
      });
    });
  } catch (error) {
    console.log(`Vite server check exception: ${error}`);
    return false;
  }
}

// Removed localhost health/port checks – offline only

async function createWindow() {
  try {
    mainWindow = new BrowserWindow({
      width: 800,
      height: 600,
      backgroundColor: '#f8f4e6',
      useContentSize: true,
      title: 'arBooks',
      webPreferences: {
        preload: path.join(__dirname, "preload.js"),
        nodeIntegration: false,
        contextIsolation: true,
        webSecurity: true,
      },
    });

    // Keep a distinct, constant title so you can identify the Electron window
    mainWindow.on('page-title-updated', (event) => {
      event.preventDefault();
      if (!mainWindow) return;
      mainWindow.setTitle('arBooks');
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
      // F11: Toggle window frame visibility (true fullscreen) with debouncing
      const now = Date.now();
      if (now - lastF11Toggle > F11_DEBOUNCE_MS) {
        lastF11Toggle = now;
        if (mainWindow.isFullScreen()) {
          mainWindow.setFullScreen(false);
        } else {
          mainWindow.setFullScreen(true);
        }
      }
    } else if (input.key.toLowerCase() === 'escape') {
      // ESC key exits fullscreen mode
      if (mainWindow.isFullScreen()) {
        mainWindow.setFullScreen(false);
      }
    }
    // Ctrl+Alt+F toggles immersive reading mode (like clicking the fullscreen button in Book page)
    if ((input.control || input.meta) && input.alt && input.key.toLowerCase() === 'f') {
      // Send message to renderer to toggle immersive reading mode
      mainWindow.webContents.send('toggle-immersive-reading');
    }
  })



    // Check if Vite dev server is running and load accordingly (unless forced offline)
    let viteRunning = !FORCE_OFFLINE && await isViteServerRunning();
    console.log(`Vite server running check: ${viteRunning}`);
    
    // If Vite is not running, wait a bit and retry (useful for development)
    if (!viteRunning && isDev && !FORCE_OFFLINE) {
      console.log("Vite not running, waiting 2 seconds and retrying...");
      await new Promise(resolve => setTimeout(resolve, 2000));
      viteRunning = await isViteServerRunning();
      console.log(`Vite server running check after retry: ${viteRunning}`);
    }
    
    if (viteRunning && !FORCE_OFFLINE) {
      const devUrl = `http://localhost:${VITE_PORT}`;
      console.log("Loading from Vite dev server:", devUrl);
      mainWindow.loadURL(devUrl);
    } else {
      // Fallback to local index.html. Try dist-electron first (both __dirname and project root), then project root.
      const fs = await import('fs/promises');
      const exists = async (p: string) => !!(await fs.access(p).then(() => true).catch(() => false));
      const distIndex = path.join(__dirname, 'index.html');
      const projRoot = getAppRootDir();
      const rendererDistIndex = path.join(projRoot, 'dist-electron', 'index.html');
      const rootIndex = path.join(projRoot, 'index.html');
      let indexPath = rootIndex;
      if (await exists(distIndex)) {
        indexPath = distIndex;
      } else if (await exists(rendererDistIndex)) {
        indexPath = rendererDistIndex;
      }
      console.log("Loading from built files:", indexPath);
      console.log("Available files in __dirname:", await fs.readdir(__dirname).catch(() => 'Error reading dir'));
      console.log("Available files in root:", await fs.readdir(getAppRootDir()).catch(() => 'Error reading root dir'));
      mainWindow.loadFile(indexPath);
    }
    
    // Open dev tools automatically in development to surface renderer errors
    if (isDev) {
      mainWindow.webContents.openDevTools({ mode: 'detach' });
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

// Removed backend server startup – offline only

// Removed HTTP helpers – offline only

app.whenReady().then(async () => {
  // Initialize voice samples directory from persisted config
  try {
    const dirInit = await loadVoiceSamplesDirFromConfig();
    console.log('[ELECTRON] Voice samples directory:', dirInit);
  } catch (e) {
    console.warn('[ELECTRON] Failed to initialize voice samples directory:', e);
  }

  // Removed backend startup/health checks – offline only

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

  // Toggle immersive reading mode from renderer
  ipcMain.handle('toggle-immersive-reading', async () => {
    if (!mainWindow) return { success: false };
    // Send message to renderer to toggle immersive reading mode
    mainWindow.webContents.send('toggle-immersive-reading');
    return { success: true };
  });

  ipcMain.handle("list-books", async (event: any, directoryPath: string) => {
    try {
      const fs = await import('fs/promises');
      const pathMod = await import('path');
      const entries = await fs.readdir(directoryPath, { withFileTypes: true });
      const books: any[] = [];
      for (const e of entries) {
        if (!e.isDirectory()) continue;
        const folder = pathMod.join(directoryPath, e.name);
        const infoPath = pathMod.join(folder, 'info.json');
        try {
          const buf = await fs.readFile(infoPath, { encoding: 'utf-8' });
          const info = JSON.parse(buf);
          books.push(info);
        } catch {}
      }
      return { success: true, result: books };
    } catch (error) {
      console.error("Error listing books:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("read-book", async (event: any, bookPath: string) => {
    try {
      const fs = await import('fs/promises');
      const pathMod = await import('path');
      const mapPath = pathMod.join(bookPath, 'text_mappings.json');
      let textMappings: any[] = [];
      try {
        const buf = await fs.readFile(mapPath, { encoding: 'utf-8' });
        textMappings = JSON.parse(buf);
      } catch (e) {
        // Fallback: build basic mappings from markdown if transcription not present
        try {
          const mdPath = pathMod.join(bookPath, 'pdf_result.md');
          const md = await fs.readFile(mdPath, { encoding: 'utf-8' }).catch(() => '');
          const rawText = md || '';
          const sentences = rawText.split(/(?<=[.!?])\s+/).filter(Boolean);
          let t = 0;
          const approxPerSentence = 3; // seconds
          textMappings = sentences.map((s) => {
            const start = t;
            const end = t + approxPerSentence;
            t = end;
            return { text: s, start, end };
          });
          console.warn('[ELECTRON] text_mappings.json missing; using basic sentence mapping fallback');
        } catch (e2) {
          console.error('[ELECTRON] Failed to create fallback text mappings:', e2);
          textMappings = [];
        }
      }
      const audioDir = pathMod.join(bookPath, 'audio');
      const vc = pathMod.join(audioDir, 'voice_cloned_output.wav');
      const reg = pathMod.join(audioDir, 'output.wav');
      let audioFile: string | null = null;
      try { await fs.access(vc); audioFile = 'voice_cloned_output.wav'; } catch {}
      if (!audioFile) { try { await fs.access(reg); audioFile = 'output.wav'; } catch {} }
      return { success: true, result: { textMappings, audioFile } };
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
      const fs = await import('fs/promises');
      const data = await fs.readFile(filePath);
      return { success: true, result: Buffer.from(data) };
    } catch (error) {
      console.error("Error getting file data:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("validate-system", async () => {
    try {
      const fs = await import('fs/promises');
      const pathMod = await import('path');
      const backendPath = getAppRootDir() ? pathMod.join(getAppRootDir(), "..", "backend-api") : path.join(process.resourcesPath, "backend-api");
      const checks: any = { offline: true };
      // Compiled CLI
      const compiled = process.platform === 'win32'
        ? pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli.exe')
        : pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli');
      try { await fs.access(compiled); checks.compiledCli = compiled; } catch { checks.compiledCli = null; }
      // Python fallback
      const pythonPath = process.platform === 'win32'
        ? pathMod.join(backendPath, 'venv-unified', 'Scripts', 'python.exe')
        : pathMod.join(backendPath, 'venv-unified', 'bin', 'python');
      try { await fs.access(pythonPath); checks.python = pythonPath; } catch { checks.python = null; }
      // Model config and directory
      const modelCfg = pathMod.join(backendPath, 'model-config.json');
      const modelsDir = pathMod.join(backendPath, 'chatterbox_models');
      try { await fs.access(modelCfg); checks.modelConfig = modelCfg; } catch { checks.modelConfig = null; }
      try { await fs.access(modelsDir); checks.modelsDir = modelsDir; } catch { checks.modelsDir = null; }
      const ok = !!(checks.compiledCli || checks.python) && !!checks.modelConfig;
      return { success: ok, result: checks, error: ok ? null : 'Missing CLI/Python or model-config.json' };
    } catch (error) {
      console.error("Error validating system:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("get-processing-status", async (event: any, jobId: string) => {
    try {
      const status = activeJobs[jobId];
      if (!status) {
        return { success: false, error: 'Job not found' };
      }
      return { success: true, result: status };
    } catch (error) {
      console.error("[ELECTRON] Error getting processing status:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("process-book-complete", async (event: any, bookData: any) => {
    try {
      console.log("[ELECTRON] Starting enhanced book processing...");
      console.log("[ELECTRON] Received bookData:", JSON.stringify(bookData, null, 2));
      
      const fileName = bookData.name;
      const pdfData = bookData.pdfData;
      const title = bookData.title;
      const author = bookData.author;
      const description = bookData.description || '';
      const bookPath = bookData.bookPath;
      const voiceCloning = bookData.voiceCloning;

      if (!fileName || !pdfData || !title || !author || !bookPath) {
        throw new Error('Missing required fields for processing');
      }

      {
        const fs = await import('fs/promises');
        const pathMod = await import('path');
        const backendPath = pathMod.join(getAppRootDir(), "..", "backend-api");
        const dirName = title.replace(/[^a-z0-9-_]/gi, '_');
        const outputDir = pathMod.join(bookPath, dirName);
        await fs.mkdir(pathMod.join(outputDir, 'audio'), { recursive: true });

        // Create a job id and initialize status
        const jobId = `${(fileName || 'book')}_${Date.now()}`;
        activeJobs[jobId] = {
          stage: 'starting',
          progress: 0,
          message: `Started processing '${title || fileName}'`,
          status: 'processing',
          filename: fileName,
          title,
        };

        // Kick work to background so renderer can poll status
        (async () => {
          const update = (patch: Partial<JobStatus>) => {
            activeJobs[jobId] = { ...activeJobs[jobId], ...patch } as JobStatus;
          };
          try {
            update({ stage: 'saving_pdf', progress: 5, message: 'Saving PDF (0–30%)...' });
            // Save PDF
            const pdfOut = pathMod.join(outputDir, fileName || 'input.pdf');
            await fs.writeFile(pdfOut, Buffer.from(new Uint8Array(pdfData)));

            // PDF -> MD
            update({ stage: 'converting_pdf', progress: 10, message: 'Converting PDF to text (0–30%)...' });
            const compiledPdfCli = process.platform === 'win32'
              ? pathMod.join(backendPath, 'dist-cli', 'pdf_cli.exe')
              : pathMod.join(backendPath, 'dist-cli', 'pdf_cli');
            const hasCompiledPdf = await (async () => { try { await fs.access(compiledPdfCli); return true; } catch { return false; } })();
            await new Promise<void>((resolve, reject) => {
              let child;
              if (hasCompiledPdf) {
                child = spawn(compiledPdfCli, ['--pdf', pdfOut, '--outdir', outputDir], { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'] });
              } else {
                const pythonPath = process.platform === 'win32'
                  ? pathMod.join(backendPath, 'venv-unified', 'Scripts', 'python.exe')
                  : pathMod.join(backendPath, 'venv-unified', 'bin', 'python');
                const pdfCliPy = pathMod.join(backendPath, 'pdf_cli.py');
                child = spawn(pythonPath, [pdfCliPy, '--pdf', pdfOut, '--outdir', outputDir], { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'] });
              }
              let stderr = Buffer.alloc(0);
              child.stderr?.on('data', d => { stderr = Buffer.concat([stderr, d]); });
              child.on('close', code => code === 0 ? resolve() : reject(new Error(`pdf_cli exited ${code}: ${stderr.toString()}`)));
            });

            // Cap PDF phase at ~30%
            update({ stage: 'converted_pdf', progress: 30, message: 'PDF converted (30%)' });

            // TTS
            const mdPath = pathMod.join(outputDir, 'pdf_result.md');
            const mdText = await fs.readFile(mdPath, { encoding: 'utf-8' });
            const audioDir = pathMod.join(outputDir, 'audio');
            const outPath = pathMod.join(audioDir, voiceCloning?.enabled ? 'voice_cloned_output.wav' : 'output.wav');

            // Determine prompt
            let promptPath: string | undefined = undefined;
            if (voiceCloning?.enabled && voiceCloning?.selectedSampleId) {
              const samplesDir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
              const entries = await fs.readdir(samplesDir, { withFileTypes: true }).catch(() => [] as any);
              for (const e of entries) {
                if (!e.isFile()) continue;
                const base = pathMod.parse(e.name).name;
                if (base === voiceCloning.selectedSampleId || e.name.startsWith(`${voiceCloning.selectedSampleId}_`)) {
                  promptPath = pathMod.join(samplesDir, e.name);
                  break;
                }
              }
              if (!promptPath) {
                console.warn(`[ELECTRON] Voice sample not found for id ${voiceCloning.selectedSampleId} in ${samplesDir}`);
              }
            }

            // Chunked TTS generation
            update({ stage: 'generating_audio', progress: 30, message: 'Preparing audio generation (30–60%)...' });
            const chunks = splitTextIntoChunks(mdText, 300);
            const totalChunks = chunks.length;
            const tempChunkFiles: string[] = [];
            const compiledTtsCli = process.platform === 'win32'
              ? pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli.exe')
              : pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli');
            const hasCompiledTts = await (async () => { try { await fs.access(compiledTtsCli); return true; } catch { return false; } })();

            for (let i = 0; i < totalChunks; i++) {
              const chunk = chunks[i];
              const osMod = await import('os');
              const fsNode = await import('fs/promises');
              const tmpTextPath = pathMod.join(osMod.tmpdir(), `tts_text_${Date.now()}_${i}.txt`);
              await fsNode.writeFile(tmpTextPath, chunk, { encoding: 'utf-8' });
              const chunkOut = pathMod.join(audioDir, `tmp_chunk_${String(i).padStart(4, '0')}.wav`);
              tempChunkFiles.push(chunkOut);

              update({
                stage: 'generating_audio',
                progress: 30 + Math.floor(((i) / Math.max(1, totalChunks)) * 30),
                message: `Generating audio chunk ${i + 1}/${totalChunks} (30–60%)...`
              });

              const args = ['--text-file', tmpTextPath, '--out', chunkOut];
              if (promptPath) { args.push('--prompt', promptPath); }
              if (voiceCloning) {
                args.push('--exaggeration', String(voiceCloning.exaggeration ?? 0.5));
                args.push('--cfg-weight', String(voiceCloning.cfgWeight ?? 0.5));
              }

              await new Promise<void>((resolve, reject) => {
                let child;
                if (hasCompiledTts) {
                  child = spawn(compiledTtsCli, args, { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, CHB_TTS_DEVICE: 'cpu' } });
                } else {
                  const pythonPath = process.platform === 'win32'
                    ? pathMod.join(backendPath, 'venv-unified', 'Scripts', 'python.exe')
                    : pathMod.join(backendPath, 'venv-unified', 'bin', 'python');
                  const cliPy = pathMod.join(backendPath, 'main_cli.py');
                  child = spawn(pythonPath, [cliPy, ...args], { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, CHB_TTS_DEVICE: 'cpu' } });
                }
                let stderr = Buffer.alloc(0);
                child.stderr?.on('data', d => { stderr = Buffer.concat([stderr, d]); });
                child.on('close', code => code === 0 ? resolve() : reject(new Error(`tts_cli exited ${code}: ${stderr.toString()}`)));
              });

              try { await fsNode.unlink(tmpTextPath); } catch {}
            }

            update({ stage: 'combining_audio', progress: 58, message: 'Combining audio chunks (≈60%)...' });
            await combineWavs(tempChunkFiles, outPath);
            // Cleanup chunk files
            for (const f of tempChunkFiles) {
              try { await fs.unlink(f); } catch {}
            }

            // Write metadata for library listing
            try {
              const info = {
                title: title || '',
                author: author || '',
                description: description || '',
                folder: dirName,
                cover: ''
              };
              await fs.writeFile(pathMod.join(outputDir, 'info.json'), JSON.stringify(info, null, 2), { encoding: 'utf-8' });
            } catch (e) {
              // Non-fatal: library may not display without info.json
              console.warn('Failed to write info.json:', e);
            }

            // Transcribe with WhisperX CLI (allocate ~30%)
            try {
              update({ stage: 'transcribing', progress: 60, message: 'Transcribing audio (60–90%)...' });
              const compiledWhisperCli = process.platform === 'win32'
                ? pathMod.join(backendPath, 'dist-cli', 'whisperx_cli.exe')
                : pathMod.join(backendPath, 'dist-cli', 'whisperx_cli');
              const hasCompiledWhisper = await (async () => { try { await fs.access(compiledWhisperCli); return true; } catch { return false; } })();
              const audioInput = outPath;
              const whisperArgs = ['--audio', audioInput, '--outdir', outputDir];
              await new Promise<void>((resolve, reject) => {
                let child;
                if (hasCompiledWhisper) {
                  child = spawn(compiledWhisperCli, whisperArgs, { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'] });
                } else {
                  const pythonPath = process.platform === 'win32'
                    ? pathMod.join(backendPath, 'venv-unified', 'Scripts', 'python.exe')
                    : pathMod.join(backendPath, 'venv-unified', 'bin', 'python');
                  const cliPy = pathMod.join(backendPath, 'whisperx_cli.py');
                  child = spawn(pythonPath, [cliPy, ...whisperArgs], { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'] });
                }
                let stderr = Buffer.alloc(0);
                child.stderr?.on('data', d => { stderr = Buffer.concat([stderr, d]); });
                child.on('close', code => code === 0 ? resolve() : reject(new Error(`whisperx_cli exited ${code}: ${stderr.toString()}`)));
              });
              update({ stage: 'transcribed', progress: 90, message: 'Transcription complete (90%)' });
            } catch (e) {
              console.warn('Transcription failed; continuing without text mappings:', e);
            }

            update({ stage: 'finalizing', progress: 95, message: 'Finalizing...' });
            update({ stage: 'completed', progress: 100, message: `Processed '${title || fileName}' successfully`, status: 'completed', files_created: { audio: true } });
          } catch (e: any) {
            const msg = e?.message || String(e);
            update({ stage: 'failed', progress: -1, message: msg, status: 'failed' });
          }
        })();

        return { success: true, result: { job_id: jobId, message: `Started processing '${title}' (${fileName})`, filename: fileName, title } };
      }
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

      const runLocalTts = async (): Promise<{ success: boolean; result?: any; error?: string }> => {
        const tmp = await import('os');
        const fs = await import('fs/promises');
        const p = await import('path');

        const promptPath = p.join(tmp.tmpdir(), `voice_prompt_${Date.now()}.wav`);
        await fs.writeFile(promptPath, Buffer.from(new Uint8Array(voicePromptFile)));

        const outPath = p.join(tmp.tmpdir(), `voice_cloned_tts_${Date.now()}.wav`);

        const backendPath = getAppRootDir() ? p.join(getAppRootDir(), "..", "backend-api") : p.join(process.resourcesPath, "backend-api");
        // Prefer compiled CLI if available
        const compiledCli = process.platform === "win32"
          ? p.join(backendPath, "dist-cli", "chatterbox_tts_cli.exe")
          : p.join(backendPath, "dist-cli", "chatterbox_tts_cli");

        // Write text to a temp file to avoid long-arg issues
        const fs2 = await import('fs/promises');
        const textPath = p.join(tmp.tmpdir(), `vc_text_${Date.now()}.txt`);
        await fs2.writeFile(textPath, text, { encoding: 'utf-8' });

        const baseEnv = {
          ...process.env,
          CHB_TTS_DEVICE: 'cpu',
          OMP_NUM_THREADS: '1',
          MKL_NUM_THREADS: '1',
          OPENBLAS_NUM_THREADS: '1',
          NUMEXPR_NUM_THREADS: '1',
          PYTHONWARNINGS: 'ignore',
          TOKENIZERS_PARALLELISM: 'false',
          HF_HUB_DISABLE_TELEMETRY: '1',
          KMP_AFFINITY: 'disabled',
          KMP_DUPLICATE_LIB_OK: 'TRUE',
        } as NodeJS.ProcessEnv;

        const modelCfgPath = p.join(backendPath, 'model-config.json');

        let child;
        if (await (async () => { try { const fs3 = await import('fs/promises'); await fs3.access(compiledCli); return true; } catch { return false; } })()) {
          const args = ["--text-file", textPath, "--prompt", promptPath, "--out", outPath, "--exaggeration", String(exaggeration ?? 0.5), "--cfg-weight", String(cfgWeight ?? 0.5), "--config", modelCfgPath];
          child = spawn(compiledCli, args, { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'], env: baseEnv });
        } else {
          const pythonPath = process.platform === "win32"
            ? p.join(backendPath, "venv-unified", "Scripts", "python.exe")
            : p.join(backendPath, "venv-unified", "bin", "python");
          const cliPath = p.join(backendPath, "main_cli.py");
          const args = [cliPath, "--text-file", textPath, "--prompt", promptPath, "--out", outPath, "--exaggeration", String(exaggeration ?? 0.5), "--cfg-weight", String(cfgWeight ?? 0.5), "--config", modelCfgPath];
          child = spawn(pythonPath, args, { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'], env: baseEnv });
        }

        const waitExit = () => new Promise<string>((resolve, reject) => {
          let stdout = Buffer.alloc(0);
          let stderr = Buffer.alloc(0);
          child.stdout?.on('data', (d) => { stdout = Buffer.concat([stdout, d]); });
          child.stderr?.on('data', (d) => { stderr = Buffer.concat([stderr, d]); });
          child.on('close', (code) => {
            if (code === 0) {
              // Always trust the known output path; stdout may contain logs
              resolve(outPath);
            } else {
              const msg = stderr.toString() || stdout.toString();
              if (String(code) === '3221225477') {
                reject(new Error(`Memory/access violation during generation. Try shorter text and ensure sufficient system memory. Details: ${msg}`));
              } else {
                reject(new Error(`CLI exited ${code}: ${msg}`));
              }
            }
          });
        });

        const resolvedOut = await waitExit();
        // Cleanup temp text/prompt files
        try { await fs.unlink(promptPath); } catch {}
        try { await fs2.unlink(textPath); } catch {}
        return { success: true, result: { audioPath: resolvedOut } };
      };

      return await runLocalTts();
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

      {
        const pathMod = await import('path');
        const fs = await import('fs/promises');
        const tmp = await import('os');

        // Determine prompt file if voice cloning is enabled
        let promptPath: string | undefined = undefined;
        if (voiceCloningOptions?.enabled && voiceCloningOptions.selectedSampleId) {
          // Look up sample in persisted voice samples directory
          const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
          const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [] as any);
          for (const e of entries) {
            if (!e.isFile()) continue;
            const base = pathMod.parse(e.name).name;
            if (base === voiceCloningOptions.selectedSampleId || e.name.startsWith(`${voiceCloningOptions.selectedSampleId}_`)) {
              promptPath = pathMod.join(dir, e.name);
              break;
            }
          }
          if (!promptPath) {
            console.warn(`[ELECTRON] Voice sample not found for id ${voiceCloningOptions.selectedSampleId} in ${dir}`);
          }
        }

        // Use main_cli once to generate voice-cloned WAV for the whole markdown if needed.
        // For now, we follow the previous behavior: generate a single output at book folder.
        const audioDir = pathMod.join(bookData.bookPath, pathMod.parse(bookData.title).name, 'audio');
        await fs.mkdir(audioDir, { recursive: true });
        const outPath = pathMod.join(audioDir, voiceCloningOptions?.enabled ? 'voice_cloned_output.wav' : 'output.wav');

        // Assemble command
        const backendPath = pathMod.join(getAppRootDir(), "..", "backend-api");
        const compiledCli = process.platform === 'win32'
          ? pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli.exe')
          : pathMod.join(backendPath, 'dist-cli', 'chatterbox_tts_cli');

        const runCli = async (text: string, prompt?: string) => {
          const args = ["--text", text, "--out", outPath];
          if (prompt) { args.push("--prompt", prompt); }
          if (voiceCloningOptions) {
            args.push("--exaggeration", String(voiceCloningOptions.exaggeration ?? 0.5));
            args.push("--cfg-weight", String(voiceCloningOptions.cfgWeight ?? 0.5));
          }
          return await new Promise<void>((resolve, reject) => {
            const child = spawn(compiledCli, args, { cwd: backendPath, stdio: ['ignore', 'pipe', 'pipe'] });
            let stderr = Buffer.alloc(0);
            child.stderr?.on('data', d => { stderr = Buffer.concat([stderr, d]); });
            child.on('close', code => {
              if (code === 0) resolve(); else reject(new Error(`CLI exited ${code}: ${stderr.toString()}`));
            });
          });
        };

        // Load markdown text
        const outputDir = pathMod.join(bookData.bookPath, pathMod.parse(bookData.title).name);
        const mdPath = pathMod.join(outputDir, 'pdf_result.md');
        const mdText = await fs.readFile(mdPath, { encoding: 'utf-8' });

        await runCli(mdText, promptPath);
        return { success: true, result: { audioPath: outPath } };
      }
    } catch (error) {
      console.error("[ELECTRON] Error generating TTS:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  // Voice sample management IPC handlers
  ipcMain.handle("upload-voice-sample", async (event: any, file: ArrayBuffer, name: string, sampleId: string, fileName: string) => {
    try {
      console.log("[ELECTRON] Uploading voice sample:", name, fileName);
      const pathMod = await import('path');
      const fs = await import('fs/promises');
      const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
      await fs.mkdir(dir, { recursive: true });
      // Preserve original extension
      const ext = pathMod.extname(fileName) || '.wav';
      const sanitized = name.replace(/[^a-z0-9-_]/gi, '_');
      const fname = `${sampleId}_${sanitized}${ext}`;
      const full = pathMod.join(dir, fname);
      const data = Buffer.from(new Uint8Array(file));
      await fs.writeFile(full, data);
      const stat = await fs.stat(full).catch(() => ({ size: data.byteLength } as any));
      return { success: true, result: { id: sampleId, name, fileName: fname, filePath: full, fileSize: stat.size } };
    } catch (error) {
      console.error("[ELECTRON] Error uploading voice sample:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });



  ipcMain.handle("get-voice-samples-directory", async (event: any) => {
    try {
      console.log("[ELECTRON] Getting voice samples directory");
      const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
      return { success: true, result: { directory: dir } };
    } catch (error) {
      console.error("[ELECTRON] Error getting voice samples directory:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("list-voice-samples", async (event: any) => {
    try {
      console.log("[ELECTRON] Listing voice samples");
      const pathMod = await import('path');
      const fs = await import('fs/promises');
      const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
      await fs.mkdir(dir, { recursive: true });
      const entries = await fs.readdir(dir, { withFileTypes: true });
      const audioExt = new Set(['.wav', '.mp3', '.flac', '.ogg', '.m4a']);
      const files = await Promise.all(entries.filter(e => e.isFile() && audioExt.has(pathMod.extname(e.name).toLowerCase())).map(async e => {
        const full = pathMod.join(dir, e.name);
        const stat = await fs.stat(full);
        const ext = pathMod.extname(e.name);
        const baseNoExt = pathMod.basename(e.name, ext);
        let id = baseNoExt;
        let displayName = baseNoExt;
        if (baseNoExt.startsWith('sample_')) {
          const parts = baseNoExt.split('_');
          if (parts.length >= 3) {
            id = `${parts[0]}_${parts[1]}`; // sample_timestamp
            displayName = parts.slice(2).join('_');
          }
        } else {
          const parts = baseNoExt.split('_');
          if (parts.length >= 2) {
            id = parts[0];
            displayName = parts.slice(1).join('_');
          }
        }
        return { id, name: displayName, fileName: e.name, filePath: full, fileSize: stat.size, uploadTime: stat.mtimeMs };
      }));
      files.sort((a, b) => (b.uploadTime || 0) - (a.uploadTime || 0));
      return { success: true, result: files };
    } catch (error) {
      console.error("[ELECTRON] Error listing voice samples:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("set-voice-samples-directory", async (event: any, directory: string) => {
    try {
      console.log("[ELECTRON] Setting voice samples directory:", directory);
      const fs = await import('fs/promises');
      await fs.mkdir(directory, { recursive: true });
      const testPath = path.join(directory, '.__writetest.tmp');
      await fs.writeFile(testPath, 'ok');
      await fs.unlink(testPath).catch(() => {});
      voiceSamplesDir = directory;
      await saveVoiceSamplesDirToConfig(directory);
      return { success: true, result: { directory } };
    } catch (error) {
      console.error("[ELECTRON] Error setting voice samples directory:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("delete-voice-sample", async (event: any, sampleId: string) => {
    try {
      console.log("[ELECTRON] Deleting voice sample:", sampleId);
      const pathMod = await import('path');
      const fs = await import('fs/promises');
      const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const e of entries) {
        if (!e.isFile()) continue;
        const name = pathMod.parse(e.name).name;
        if (name === sampleId || e.name.startsWith(`${sampleId}_`)) {
          await fs.unlink(pathMod.join(dir, e.name));
          return { success: true, result: `Voice sample ${sampleId} deleted` };
        }
      }
      return { success: false, error: `Voice sample ${sampleId} not found` };
    } catch (error) {
      console.error("[ELECTRON] Error deleting voice sample:", error);
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });

  ipcMain.handle("get-voice-sample-url", async (event: any, sampleId: string) => {
    try {
      const pathMod = await import('path');
      const fs = await import('fs/promises');
      const dir = voiceSamplesDir || await loadVoiceSamplesDirFromConfig();
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const e of entries) {
        if (!e.isFile()) continue;
        const name = pathMod.parse(e.name).name;
        if (name === sampleId || e.name.startsWith(`${sampleId}_`)) {
          const full = pathMod.join(dir, e.name);
          return { success: true, result: full };
        }
      }
      return { success: false, error: 'Not found' };
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
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {});
