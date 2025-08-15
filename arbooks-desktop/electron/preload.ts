import { contextBridge, ipcRenderer } from 'electron';
import { BookUploadData } from '../src/types/book';

// Debug logging
console.log("Preload script starting...");

const electronAPI = {
  selectDirectory: () => {
    console.log("selectDirectory called from preload");
    return ipcRenderer.invoke("select-directory");
  },
  listBooks: (directoryPath: string) => ipcRenderer.invoke("list-books", directoryPath),
  readBook: (bookPath: string) => ipcRenderer.invoke("read-book", bookPath),
  getFileData: (filePath: string) => ipcRenderer.invoke("get-file-data", filePath),

  // Enhanced complete book processing
  processBookComplete: (bookData: BookUploadData) => ipcRenderer.invoke("process-book-complete", bookData),
  getProcessingStatus: (jobId: string) => ipcRenderer.invoke("get-processing-status", jobId),
  validateSystem: () => ipcRenderer.invoke("validate-system"),

  // TTS (offline-only)
  generateTTS: (bookData: BookUploadData, voiceCloningOptions?: { enabled: boolean; selectedSampleId: string | null; exaggeration: number; cfgWeight: number }) => ipcRenderer.invoke("generate-tts", bookData, voiceCloningOptions),

  // Voice cloning functions (offline-only)
  generateVoiceClonedSpeech: (text: string, voicePromptFile: ArrayBuffer, exaggeration?: number, cfgWeight?: number) => ipcRenderer.invoke("generate-voice-cloned-speech", text, voicePromptFile, exaggeration, cfgWeight),

  // Voice sample management functions
  uploadVoiceSample: (file: ArrayBuffer, name: string, sampleId: string, fileName: string) => ipcRenderer.invoke("upload-voice-sample", file, name, sampleId, fileName),
  deleteVoiceSample: (sampleId: string) => ipcRenderer.invoke("delete-voice-sample", sampleId),
  getVoiceSampleUrl: (sampleId: string) => ipcRenderer.invoke("get-voice-sample-url", sampleId),
  
  // Voice samples directory management
  getVoiceSamplesDirectory: () => ipcRenderer.invoke("get-voice-samples-directory"),
  setVoiceSamplesDirectory: (directory: string) => ipcRenderer.invoke("set-voice-samples-directory", directory),
  listVoiceSamples: () => ipcRenderer.invoke("list-voice-samples"),

  // Window controls
  toggleKiosk: () => ipcRenderer.invoke('toggle-kiosk'),
};

try {
  contextBridge.exposeInMainWorld('electron', electronAPI);
  console.log("contextBridge.exposeInMainWorld succeeded");
  console.log("Exposed electron API:", Object.keys(electronAPI));
} catch (error) {
  console.error("Failed to expose electron API:", error);
}

console.log("Preload loaded!");

export { };
