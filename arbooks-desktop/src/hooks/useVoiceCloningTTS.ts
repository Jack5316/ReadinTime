import { useCallback, useState } from 'react';
import { Result } from '../types/result';
import useStore from '../store/useStore';
import '../global';

interface UseVoiceCloningTTSReturn {
  generateSpeech: (text: string, voicePromptFile?: File) => Promise<Result<string>>;
  isGenerating: boolean;
  error: string | null;
  audioUrl: string | null;
}

const useVoiceCloningTTS = (): UseVoiceCloningTTSReturn => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const { settings } = useStore();

  const generateSpeech = useCallback(async (text: string, voicePromptFile?: File): Promise<Result<string>> => {
    setIsGenerating(true);
    setError(null);
    setAudioUrl(null);

    try {
      if (!text.trim()) {
        throw new Error('Text cannot be empty');
      }

      // Use provided file or get from settings
      const promptFile = voicePromptFile || (settings.voiceCloning.voicePromptPath ? 
        // In a real implementation, you'd load the file from the stored path
        // For now, we'll require the file to be provided
        null : null);

      if (!promptFile) {
        throw new Error('Voice prompt file is required for voice cloning');
      }

      const voicePromptArrayBuffer = await promptFile.arrayBuffer();
      
      if (!window.electron || !window.electron.generateVoiceClonedSpeech) {
        throw new Error('Electron API not available. Please run the app with Electron.');
      }

      const result = await window.electron.generateVoiceClonedSpeech(
        text,
        voicePromptArrayBuffer,
        settings.voiceCloning.exaggeration,
        settings.voiceCloning.cfgWeight
      );

      if (result.success) {
        // The result should contain the path to the generated audio file
        // In a real implementation, you might convert this to a blob URL for playback
        setAudioUrl(result.result);
        return { success: true, result: result.result };
      } else {
        setError(result.error || 'Failed to generate speech');
        return { success: false, error: result.error || 'Failed to generate speech' };
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsGenerating(false);
    }
  }, [settings.voiceCloning]);

  return {
    generateSpeech,
    isGenerating,
    error,
    audioUrl,
  };
};

export default useVoiceCloningTTS;
