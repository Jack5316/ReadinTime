import React, { useEffect, useMemo, useRef, useState } from 'react';
import useVoiceCloningTTS from '../../hooks/useVoiceCloningTTS';
import useStore from '../../store/useStore';
import '../../global';

const VoiceCloningDemo: React.FC = () => {
  const [text, setText] = useState('Hello! This is a test of voice cloning technology using Chatterbox.');
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [source, setSource] = useState<'upload' | 'saved'>('upload');
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const { generateSpeech, isGenerating, error } = useVoiceCloningTTS();
  const { settings, loadVoiceSamples } = useStore();
  
  // Check if running in Electron environment
  const isElectronAvailable = typeof window !== 'undefined' && window.electron;

  // Load saved samples initially
  useEffect(() => {
    const refresh = async () => {
      if (!isElectronAvailable || !window.electron.listVoiceSamples) return;
      try {
        const res = await window.electron.listVoiceSamples();
        if (res?.success && Array.isArray(res.result)) {
          loadVoiceSamples(res.result);
        }
      } catch {}
    };
    refresh();
  }, [isElectronAvailable, loadVoiceSamples]);

  // Initialize selected sample if available
  useEffect(() => {
    if (!selectedSampleId && (settings.voiceCloning.voiceSamples?.length || 0) > 0) {
      setSelectedSampleId(settings.voiceCloning.voiceSamples[0].id);
    }
  }, [settings.voiceCloning.voiceSamples, selectedSampleId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      setVoiceFile(file);
    }
  };

  const loadAudioAsBlobUrl = useMemo(() => {
    return async (pathOrUrl: string) => {
      try {
        if (!isElectronAvailable || !window.electron.getFileData) return pathOrUrl;
        const res = await window.electron.getFileData(pathOrUrl);
        if (!res?.success) return pathOrUrl;
        const blob = new Blob([res.result], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        return url;
      } catch {
        return pathOrUrl;
      }
    };
  }, [isElectronAvailable]);

  const handleGenerate = async () => {
    let promptToUse: File | null = null;
    try {
      if (source === 'upload') {
        if (!voiceFile) return;
        promptToUse = voiceFile;
      } else {
        if (!selectedSampleId) return;
        if (!isElectronAvailable) throw new Error('Electron API not available');
        const urlRes = await window.electron.getVoiceSampleUrl(selectedSampleId);
        if (!urlRes?.success) throw new Error('Failed to resolve voice sample path');
        const filePath: string = urlRes.result;
        const dataRes = await window.electron.getFileData(filePath);
        if (!dataRes?.success) throw new Error('Failed to read voice sample file');
        const blob = new Blob([dataRes.result], { type: 'audio/wav' });
        promptToUse = new File([blob], 'voice_prompt.wav', { type: 'audio/wav' });
      }

      // Revoke old URL if any before generating new one
      if (audioUrl) {
        try { URL.revokeObjectURL(audioUrl); } catch {}
      }

      const result = await generateSpeech(text, promptToUse);
      if (result.success && result.result) {
        const raw = result.result as any;
        const path = typeof raw === 'string' ? raw : (raw?.audioPath || '');
        const playable = await loadAudioAsBlobUrl(path);
        setAudioUrl(playable);
      }
    } catch (e: any) {
      console.error('[VoiceCloningDemo] Generate failed:', e);
    }
  };

  const isReady = useMemo(() => {
    if (!settings.voiceCloning.enabled) return false;
    if (source === 'upload') return !!voiceFile && text.trim().length > 0;
    return !!selectedSampleId && text.trim().length > 0;
  }, [settings.voiceCloning.enabled, source, voiceFile, selectedSampleId, text]);

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

        {/* Source selection */}
        <div className="form-control">
          <label className="label">
            <span className="label-text">Voice reference source</span>
          </label>
          <div className="join">
            <button className={`btn join-item ${source==='upload'?'btn-active':''}`} onClick={()=>setSource('upload')} disabled={isGenerating}>Upload</button>
            <button className={`btn join-item ${source==='saved'?'btn-active':''}`} onClick={()=>setSource('saved')} disabled={isGenerating}>Saved Samples</button>
          </div>
        </div>

        {source === 'upload' ? (
          <div className="form-control">
            <label className="label">
              <span className="label-text">Upload voice reference</span>
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
        ) : (
          <div className="form-control">
            <label className="label">
              <span className="label-text">Choose a saved sample</span>
            </label>
            <select
              className="select select-bordered"
              value={selectedSampleId || ''}
              onChange={(e)=>setSelectedSampleId(e.target.value || null)}
              disabled={isGenerating}
            >
              {(settings.voiceCloning.voiceSamples || []).length === 0 && (
                <option value="">No samples found</option>
              )}
              {(settings.voiceCloning.voiceSamples || []).map(s => (
                <option key={s.id} value={s.id}>{s.name || s.id}</option>
              ))}
            </select>
            <div className="mt-2">
              <button
                className="btn btn-sm"
                onClick={async ()=>{
                  if (!window.electron?.listVoiceSamples) return;
                  const res = await window.electron.listVoiceSamples();
                  if (res?.success && Array.isArray(res.result)) {
                    loadVoiceSamples(res.result);
                  }
                }}
                disabled={isGenerating}
              >Refresh</button>
            </div>
          </div>
        )}

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
