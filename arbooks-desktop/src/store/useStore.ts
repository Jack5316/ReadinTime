import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { SettingsState, VoiceSample, MODES } from "./settings";

type ProcessingStatus = {
  stage: string;
  progress: number;
  message: string;
  status: 'processing' | 'completed' | 'failed';
  files_created?: Record<string, boolean>;
  filename?: string;
  title?: string;
};

type ProcessingState = {
  currentJobId: string | null;
  uploading: boolean;
  status: ProcessingStatus | null;
  startedAt: number | null;
  meta?: { title?: string; filename?: string } | null;
};

interface AppState {
  settings: SettingsState;
  updateSettings: (settings: SettingsState) => void;
  addVoiceSample: (voiceSample: VoiceSample) => void;
  removeVoiceSample: (voiceSampleId: string) => void;
  selectVoiceSample: (voiceSampleId: string | null) => void;
  updateVoiceSample: (voiceSampleId: string, updates: Partial<VoiceSample>) => void;
  loadVoiceSamples: (voiceSamples: VoiceSample[]) => void;
  processing: ProcessingState;
  startProcessing: (jobId: string, meta?: { title?: string; filename?: string }) => void;
  updateProcessingStatus: (status: ProcessingStatus) => void;
  clearProcessing: () => void;
}

const defaultSettings: SettingsState = {
  mode: 'Speech and Images',
  fontSize: 18,
  fontStyle: 'Source Serif 4',
  fontColour: 'black',
  bgColour: 'white',
  lineSpacing: 1.5,
  wordSpacing: 1,
  autoScroll: false,
  autoPageTurn: false,
  autoPageTurnDelay: 5,
  isFullScreen: false,
  bookPath: null,
  voiceSamplesDirectory: null, // Add the missing field
  voiceCloning: {
    enabled: false,
    exaggeration: 0.5,
    cfgWeight: 0.5,
    selectedVoiceSampleId: null,
    voiceSamples: [],
    voicePromptPath: null,
  },
}

const defaultProcessingState: ProcessingState = {
  currentJobId: null,
  uploading: false,
  status: null,
  startedAt: null,
  meta: null,
};

// Helper function to validate and fix settings
const validateAndFixSettings = (settings: any): SettingsState => {
  if (!settings || typeof settings !== 'object') {
    console.warn('Invalid settings object, using defaults');
    return defaultSettings;
  }

  // Ensure all required properties exist
  const validatedSettings: SettingsState = {
    mode: (MODES as readonly string[]).includes(settings.mode) ? settings.mode as typeof MODES[number] : defaultSettings.mode,
    fontSize: typeof settings.fontSize === 'number' ? settings.fontSize : defaultSettings.fontSize,
    fontStyle: settings.fontStyle || defaultSettings.fontStyle,
    fontColour: settings.fontColour || defaultSettings.fontColour,
    bgColour: settings.bgColour || defaultSettings.bgColour,
    lineSpacing: typeof settings.lineSpacing === 'number' ? settings.lineSpacing : defaultSettings.lineSpacing,
    wordSpacing: typeof settings.wordSpacing === 'number' ? settings.wordSpacing : defaultSettings.wordSpacing,
    autoScroll: typeof settings.autoScroll === 'boolean' ? settings.autoScroll : defaultSettings.autoScroll,
    autoPageTurn: typeof settings.autoPageTurn === 'boolean' ? settings.autoPageTurn : defaultSettings.autoPageTurn,
    autoPageTurnDelay: typeof settings.autoPageTurnDelay === 'number' ? settings.autoPageTurnDelay : defaultSettings.autoPageTurnDelay,
    isFullScreen: typeof settings.isFullScreen === 'boolean' ? settings.isFullScreen : defaultSettings.isFullScreen,
    bookPath: settings.bookPath || defaultSettings.bookPath,
    voiceSamplesDirectory: settings.voiceSamplesDirectory || defaultSettings.voiceSamplesDirectory,
    voiceCloning: {
      enabled: typeof settings.voiceCloning?.enabled === 'boolean' ? settings.voiceCloning.enabled : defaultSettings.voiceCloning.enabled,
      exaggeration: typeof settings.voiceCloning?.exaggeration === 'number' ? settings.voiceCloning.exaggeration : defaultSettings.voiceCloning.exaggeration,
      cfgWeight: typeof settings.voiceCloning?.cfgWeight === 'number' ? settings.voiceCloning.cfgWeight : defaultSettings.voiceCloning.cfgWeight,
      selectedVoiceSampleId: settings.voiceCloning?.selectedVoiceSampleId || defaultSettings.voiceCloning.selectedVoiceSampleId,
      voiceSamples: Array.isArray(settings.voiceCloning?.voiceSamples) ? settings.voiceCloning.voiceSamples : defaultSettings.voiceCloning.voiceSamples,
      voicePromptPath: settings.voiceCloning?.voicePromptPath || defaultSettings.voiceCloning.voicePromptPath,
    }
  };

  return validatedSettings;
};

const useStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        settings: defaultSettings,
        processing: defaultProcessingState,
        updateSettings: (settings) => {
          const validatedSettings = validateAndFixSettings(settings);
          set({ settings: validatedSettings });
        },
        startProcessing: (jobId, meta) => {
          set({
            processing: {
              currentJobId: jobId,
              uploading: true,
              status: null,
              startedAt: Date.now(),
              meta: meta || null,
            },
          });
        },
        updateProcessingStatus: (status) => {
          const current = get().processing;
          const uploading = status.status === 'processing';
          set({
            processing: {
              ...current,
              uploading,
              status,
            },
          });
        },
        clearProcessing: () => {
          set({ processing: defaultProcessingState });
        },
        
        addVoiceSample: (voiceSample) => {
          const currentSettings = get().settings;
          const currentVoiceSamples = Array.isArray(currentSettings.voiceCloning.voiceSamples) 
            ? currentSettings.voiceCloning.voiceSamples 
            : [];
          const updatedSettings = {
            ...currentSettings,
            voiceCloning: {
              ...currentSettings.voiceCloning,
              voiceSamples: [...currentVoiceSamples, voiceSample]
            }
          };
          set({ settings: updatedSettings });
        },
        
        removeVoiceSample: (voiceSampleId) => {
          const currentSettings = get().settings;
          const currentVoiceSamples = Array.isArray(currentSettings.voiceCloning.voiceSamples) 
            ? currentSettings.voiceCloning.voiceSamples 
            : [];
          const updatedVoiceSamples = currentVoiceSamples.filter(
            sample => sample.id !== voiceSampleId
          );
          const selectedId = currentSettings.voiceCloning.selectedVoiceSampleId === voiceSampleId 
            ? null 
            : currentSettings.voiceCloning.selectedVoiceSampleId;
          
          const updatedSettings = {
            ...currentSettings,
            voiceCloning: {
              ...currentSettings.voiceCloning,
              voiceSamples: updatedVoiceSamples,
              selectedVoiceSampleId: selectedId
            }
          };
          set({ settings: updatedSettings });
        },
        
        selectVoiceSample: (voiceSampleId) => {
          console.log('Store: Selecting voice sample:', voiceSampleId);
          const currentSettings = get().settings;
          const updatedSettings = {
            ...currentSettings,
            voiceCloning: {
              ...currentSettings.voiceCloning,
              selectedVoiceSampleId: voiceSampleId
            }
          };
          console.log('Store: Updated settings:', updatedSettings.voiceCloning);
          set({ settings: updatedSettings });
        },

        loadVoiceSamples: (voiceSamples) => {
          console.log('Store: Loading voice samples:', voiceSamples.length);
          const currentSettings = get().settings;
          const currentSelectedId = currentSettings.voiceCloning?.selectedVoiceSampleId;
          
          // Check if the currently selected voice sample still exists in the loaded samples
          const selectedIdExists = currentSelectedId && voiceSamples.some(sample => sample.id === currentSelectedId);
          
          console.log('Store: Voice sample selection check:', {
            currentSelectedId,
            selectedIdExists,
            availableIds: voiceSamples.map(s => s.id)
          });
          
          const updatedSettings = {
            ...currentSettings,
            voiceCloning: {
              ...currentSettings.voiceCloning,
              voiceSamples: voiceSamples,
              // Only preserve the selected voice sample ID if it still exists in the loaded samples
              selectedVoiceSampleId: selectedIdExists ? currentSelectedId : null,
            }
          };
          console.log('Store: Updated voice cloning settings:', updatedSettings.voiceCloning);
          set({ settings: updatedSettings });
        },
        
        updateVoiceSample: (voiceSampleId, updates) => {
          const currentSettings = get().settings;
          const currentVoiceSamples = Array.isArray(currentSettings.voiceCloning.voiceSamples) 
            ? currentSettings.voiceCloning.voiceSamples 
            : [];
          const updatedVoiceSamples = currentVoiceSamples.map(
            sample => sample.id === voiceSampleId ? { ...sample, ...updates } : sample
          );
          
          const updatedSettings = {
            ...currentSettings,
            voiceCloning: {
              ...currentSettings.voiceCloning,
              voiceSamples: updatedVoiceSamples
            }
          };
          set({ settings: updatedSettings });
        },
      }),
      {
        name: "app-storage",
        migrate: (persistedState: any, version: number) => {
          console.log('Migrating persisted state, version:', version);
          
          // Validate and fix the persisted state
          if (persistedState && persistedState.settings) {
            console.log('Validating persisted settings...');
            persistedState.settings = validateAndFixSettings(persistedState.settings);
            if (!persistedState.processing) {
              persistedState.processing = defaultProcessingState;
            } else {
              const p = persistedState.processing;
              persistedState.processing = {
                currentJobId: typeof p.currentJobId === 'string' ? p.currentJobId : null,
                uploading: !!p.uploading,
                status: p.status || null,
                startedAt: typeof p.startedAt === 'number' ? p.startedAt : null,
                meta: p.meta || null,
              } as ProcessingState;
            }
          } else {
            console.log('No valid persisted state found, using defaults');
            return { settings: defaultSettings, processing: defaultProcessingState };
          }
          
          return persistedState;
        },
      }
    )
  )
);

export default useStore;
