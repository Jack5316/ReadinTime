// Neurodiverse-friendly color options
export const FONT_COLORS = {
  black: '#000000',
  darkBlue: '#1a365d',
  darkBrown: '#5c4d3c',
  darkGreen: '#2d5a3d'
};

export const BG_COLORS = {
  white: '#ffffff',
  cream: '#f8f4e6',
  lightBlue: '#e6f2ff',
  lightYellow: '#fff9e6'
};

export const FONT_STYLES = ["Open-Dyslexic", "Source Serif 4", "Lexend Deca", "Arial", "Helvetica", "Tahoma"] as const;
export const MODES = ["Speech and Images", "Text Only"] as const;

export interface VoiceSample {
  id: string;
  name: string;
  fileName: string;
  filePath: string;
  dateAdded: string;
  duration?: number; // in seconds
  fileSize?: number; // in bytes
}

export interface VoiceCloningSettings {
  enabled: boolean;
  exaggeration: number; // 0.0 to 1.0
  cfgWeight: number; // 0.0 to 1.0
  selectedVoiceSampleId: string | null;
  voiceSamples: VoiceSample[];
  // Legacy field for backward compatibility
  voicePromptPath: string | null;
}

export interface SettingsState {
  mode: typeof MODES[number];
  fontSize: number;
  fontStyle: typeof FONT_STYLES[number];
  fontColour: keyof typeof FONT_COLORS;
  bgColour: keyof typeof BG_COLORS;
  lineSpacing: number;
  wordSpacing: number;
  autoScroll: boolean;
  autoPageTurn: boolean;
  autoPageTurnDelay: number;
  isFullScreen: boolean;
  bookPath: string | null;
  voiceSamplesDirectory: string | null;
  voiceCloning: VoiceCloningSettings;
}
