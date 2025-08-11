export interface TextSegment {
  text: string;
  start: number;
  end: number;
}


export interface BookInfo {
  title: string;
  author: string;
  description: string;
  folder: string;
  cover: string;
}

export interface BookData {
  textMappings: TextSegment[];
  audioFile?: string; // 'output.wav' or 'voice_cloned_output.wav'
}

export interface VoiceCloningOptions {
  enabled: boolean;
  exaggeration: number;
  cfgWeight: number;
  voicePromptFile?: File;
}

export interface BookUploadData {
  name: string;
  pdfData: ArrayBuffer;
  title: string;
  author: string;
  description: string;
  bookPath: string;
  voiceCloning?: VoiceCloningOptions;
}
