/// <reference types="vite/client" />

import { BookUploadData } from './types/book';
import { Result } from './types/result';
import { VoiceSample } from './store/settings';

interface ElectronAPI {
  selectDirectory: () => Promise<{ success: boolean; result?: string; error?: any }>;
  listBooks: (directoryPath: string) => Promise<any>;
  readBook: (bookPath: string) => Promise<any>;
  getFileData: (filePath: string) => Promise<any>;
  processBookComplete: (bookData: BookUploadData) => Promise<any>;
  getProcessingStatus: (jobId: string) => Promise<any>;
  validateSystem: () => Promise<any>;
  uploadPDF: (bookData: BookUploadData) => Promise<any>;
  convertPDFToMarkdown: (bookData: BookUploadData) => Promise<any>;
  generateTTS: (bookData: BookUploadData) => Promise<any>;
  transcribeAudio: (bookData: BookUploadData) => Promise<any>;
  saveMetadata: (bookData: BookUploadData) => Promise<any>;
  generateVoiceClonedTTS: (bookData: BookUploadData, voicePromptFile: ArrayBuffer) => Promise<any>;
  generateVoiceClonedSpeech: (text: string, voicePromptFile: ArrayBuffer, exaggeration?: number, cfgWeight?: number) => Promise<{ success: boolean; result?: string; error?: string }>;
  
  // Voice sample management
  uploadVoiceSample: (file: ArrayBuffer, name: string, sampleId: string, fileName: string) => Promise<Result<VoiceSample>>;
  deleteVoiceSample: (sampleId: string) => Promise<any>;
  getVoiceSampleUrl: (sampleId: string) => Promise<any>;
  
  // Voice samples directory management
  getVoiceSamplesDirectory: () => Promise<Result<string>>;
  setVoiceSamplesDirectory: (directory: string) => Promise<Result<void>>;
}

interface Window {
  electron: ElectronAPI;
}
