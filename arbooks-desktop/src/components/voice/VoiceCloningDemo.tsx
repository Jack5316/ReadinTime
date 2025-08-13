import React, { useState, useRef } from 'react';
import useVoiceCloningTTS from '../../hooks/useVoiceCloningTTS';
import useStore from '../../store/useStore';
import '../../global';

const VoiceCloningDemo: React.FC = () => {
  const [text, setText] = useState('Hello! This is a test of voice cloning technology using Chatterbox.');
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const { generateSpeech, isGenerating, error } = useVoiceCloningTTS();
  const { settings } = useStore();
  
  // Check if running in Electron environment
  const isElectronAvailable = typeof window !== 'undefined' && window.electron;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      setVoiceFile(file);
    }
  };

  const handleGenerate = async () => {
    if (!voiceFile) return;
    
    const result = await generateSpeech(text, voiceFile);
    
    if (result.success && result.result) {
      // Use direct file path (offline mode). Backend URL only if needed.
      const resultData = result.result as any;
      const audioPath = resultData.audioPath || result.result;
      setAudioUrl(audioPath);
    }
  };

  const isReady = settings.voiceCloning.enabled && voiceFile && text.trim().length > 0;

  return (
    <div className="card bg-base-100 w-full max-w-2xl shadow-xl">
      <div className="card-body">
        <h2 className="card-title">Voice Cloning Demo</h2>
        
        {!settings.voiceCloning.enabled && (
          <div className="alert alert-warning">
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span>Voice cloning is disabled. Please enable it in Settings first.</span>
          </div>
        )}

        <div className="form-control">
          <label className="label">
            <span className="label-text">Text to speak</span>
          </label>
          <textarea
            className="textarea textarea-bordered h-24"
            placeholder="Enter text to convert to speech..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isGenerating}
          />
        </div>

        <div className="form-control">
          <label className="label">
            <span className="label-text">Voice reference audio</span>
          </label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="file-input file-input-bordered w-full"
            disabled={isGenerating}
          />
          {voiceFile && (
            <div className="text-sm text-success mt-2">
              ✓ {voiceFile.name} selected
            </div>
          )}
        </div>

        {settings.voiceCloning.enabled && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">
                <span className="label-text">Exaggeration: {settings.voiceCloning.exaggeration.toFixed(2)}</span>
              </label>
              <div className="text-xs text-base-content/70">
                Current setting from preferences
              </div>
            </div>
            <div>
              <label className="label">
                <span className="label-text">Voice Adherence: {settings.voiceCloning.cfgWeight.toFixed(2)}</span>
              </label>
              <div className="text-xs text-base-content/70">
                Current setting from preferences
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="alert alert-error">
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        <div className="card-actions justify-end">
          <button
            className={`btn btn-primary ${isGenerating ? 'loading' : ''}`}
            onClick={handleGenerate}
            disabled={!isReady || isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate Speech'}
          </button>
        </div>

        {audioUrl && (
          <div className="mt-4">
            <label className="label">
              <span className="label-text">Generated speech</span>
            </label>
            <audio
              ref={audioRef}
              controls
              className="w-full"
              src={audioUrl}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceCloningDemo;
