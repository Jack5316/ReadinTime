import { BookInfo, BookData, BookUploadData } from './types/book';
import { Result } from './types/result';
import { Buffer } from 'buffer';

export { };

declare global {
  interface Window {
    electron: {
      selectDirectory: () => Promise<Result<string>>;
      listBooks: (directoryPath: string) => Promise<Result<BookInfo[]>>;
      readBook: (bookPath: string) => Promise<Result<BookData>>;
      getFileData: (filePath: string) => Promise<Result<Buffer>>;

      // Enhanced complete book processing
      processBookComplete: (bookData: BookUploadData) => Promise<Result<any>>;
      getProcessingStatus: (jobId: string) => Promise<Result<any>>;
      validateSystem: () => Promise<Result<any>>;

      // uploading book pipeline (legacy individual steps)
      uploadPDF: (bookData: BookUploadData) => Promise<Result<string>>,
      convertPDFToMarkdown: (bookData: BookUploadData) => Promise<Result<string>>,
      generateTTS: (bookData: BookUploadData, voiceCloningOptions?: { enabled: boolean; selectedSampleId: string | null; exaggeration: number; cfgWeight: number }) => Promise<Result<string>>,
      transcribeAudio: (bookData: BookUploadData) => Promise<Result<string>>,
      saveMetadata: (bookData: BookUploadData) => Promise<Result<string>>,

        // Voice cloning functions
  generateVoiceClonedTTS: (bookData: BookUploadData, voicePromptFile: ArrayBuffer) => Promise<Result<string>>;
  generateVoiceClonedSpeech: (text: string, voicePromptFile: ArrayBuffer, exaggeration?: number, cfgWeight?: number) => Promise<Result<string>>;
  
  // Voice sample management
  uploadVoiceSample: (file: ArrayBuffer, name: string, sampleId: string, fileName: string) => Promise<Result<any>>;
  deleteVoiceSample: (sampleId: string) => Promise<Result<any>>;
  getVoiceSampleUrl: (sampleId: string) => Promise<Result<string>>;
  
  // Voice samples directory management
  getVoiceSamplesDirectory: () => Promise<Result<string>>;
  setVoiceSamplesDirectory: (directory: string) => Promise<Result<string>>;
  listVoiceSamples: () => Promise<Result<VoiceSample[]>>;
  
  // Window controls
  toggleKiosk: () => Promise<Result<any>>;
  toggleImmersiveReading: () => Promise<Result<any>>;
  onToggleImmersiveReading: (callback: () => void) => () => void;
    }
  }
}
