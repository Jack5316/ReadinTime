import React, { FC, useCallback, useState, useEffect, useRef } from 'react';
import useStore from '../../store/useStore';
import { VoiceSample } from '../../store/settings';
import { Result } from '../../types/result';
import '../../global';

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const VoiceCloningSettings: FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [newVoiceName, setNewVoiceName] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [newlyAddedId, setNewlyAddedId] = useState<string | null>(null);
  // Avoid blocking the whole panel; render immediately
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingVoiceSamples, setIsLoadingVoiceSamples] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [voiceSamplesDirectory, setVoiceSamplesDirectory] = useState<string | { directory: string } | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (newlyAddedId && editInputRef.current) {
      editInputRef.current.focus();
    }
  }, [newlyAddedId]);

  const {
    settings,
    updateSettings,
    addVoiceSample,
    removeVoiceSample,
    selectVoiceSample,
    updateVoiceSample,
    loadVoiceSamples,
  } = useStore();

  const storeRef = useRef({ settings, updateSettings, addVoiceSample, removeVoiceSample, selectVoiceSample, updateVoiceSample, loadVoiceSamples });
  useEffect(() => {
    storeRef.current = { settings, updateSettings, addVoiceSample, removeVoiceSample, selectVoiceSample, updateVoiceSample, loadVoiceSamples };
  }, [settings, updateSettings, addVoiceSample, removeVoiceSample, selectVoiceSample, updateVoiceSample, loadVoiceSamples]);

  const isElectronAvailable = typeof window !== 'undefined' && !!window.electron;

  const loadVoiceSamplesFromBackend = useCallback(async () => {
    if (!isElectronAvailable || !window.electron?.listVoiceSamples) return;
    try {
      setIsLoadingVoiceSamples(true);
      const result = await window.electron.listVoiceSamples();
      if (result.success && Array.isArray(result.result)) {
        storeRef.current.loadVoiceSamples?.(result.result);
        if (result.result.length > 0) {
          setSuccessMessage(`Loaded ${result.result.length} voice sample(s)`);
          setTimeout(() => setSuccessMessage(null), 2000);
        }
      }
    } catch (error) {
      console.warn('Failed to load voice samples from backend:', error);
    } finally {
      setIsLoadingVoiceSamples(false);
    }
  }, [isElectronAvailable]);

  const initializeComponent = useCallback(async () => {
    try {
      setConnectionError(null);

      const currentSettings = storeRef.current.settings;
      if (!currentSettings) {
        throw new Error('Settings object is not available.');
      }

      if (!currentSettings.voiceCloning) {
        storeRef.current.updateSettings?.({
          ...currentSettings,
          voiceCloning: {
            enabled: false,
            exaggeration: 0.5,
            cfgWeight: 0.5,
            selectedVoiceSampleId: null,
            voiceSamples: [],
            voicePromptPath: null,
          },
        });
      } else if (!Array.isArray(currentSettings.voiceCloning.voiceSamples)) {
        storeRef.current.updateSettings?.({
          ...currentSettings,
          voiceCloning: {
            ...currentSettings.voiceCloning,
            voiceSamples: [],
            // Preserve the selected voice sample ID when resetting the array
            selectedVoiceSampleId: currentSettings.voiceCloning.selectedVoiceSampleId,
          },
        });
      }

      // Directory: use cached value first, then fetch lazily if missing
      if (typeof currentSettings.voiceSamplesDirectory === 'string') {
        setVoiceSamplesDirectory(currentSettings.voiceSamplesDirectory);
      } else if (isElectronAvailable && window.electron?.getVoiceSamplesDirectory) {
        window.electron.getVoiceSamplesDirectory()
          .then((res: any) => {
            const dir = res?.result?.directory || res?.directory;
            if (dir) {
              setVoiceSamplesDirectory(dir);
              const cur = storeRef.current.settings;
              storeRef.current.updateSettings?.({ ...cur, voiceSamplesDirectory: dir });
            }
          })
          .catch(() => {});
      }

      // Voice samples: show cached immediately; refresh in background if empty
      const cached = currentSettings.voiceCloning?.voiceSamples || [];
      if (!Array.isArray(cached) || cached.length === 0) {
        loadVoiceSamplesFromBackend();
      }
    } catch (error) {
      console.error('Component initialization failed:', error);
      setConnectionError(`Settings initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsInitialized(true);
    }
  }, [isElectronAvailable, loadVoiceSamplesFromBackend]);

  useEffect(() => {
    if (!isInitialized) {
      initializeComponent();
    }
  }, [isInitialized, initializeComponent]);

  const handleAddVoiceSample = async (file: File) => {
    if (!newVoiceName.trim()) {
      setConnectionError('Please enter a name for the new voice sample.');
      return;
    }
    setIsUploading(true);
    setConnectionError(null);
    try {
      if (!isElectronAvailable || !window.electron?.uploadVoiceSample) {
        throw new Error('Voice sample upload API is not available.');
      }

      const fileArrayBuffer = await file.arrayBuffer();
      const sampleId = `sample_${Date.now()}`;

      const result: Result<VoiceSample> = await window.electron.uploadVoiceSample(
        fileArrayBuffer,
        newVoiceName.trim(),
        sampleId,
        file.name
      );

      if (result.success) {
        // Ensure the returned sample has a valid name. If the backend did not include
        // the name, fall back to the name the user entered.
        const sampleWithName = {
          ...result.result,
          name: (result.result.name && result.result.name.trim() !== '') ? result.result.name : newVoiceName.trim(),
        } as VoiceSample;

        storeRef.current.addVoiceSample?.(sampleWithName);
        setNewlyAddedId(sampleId);
        setEditingId(sampleId);
        setEditName(newVoiceName.trim());
        setNewVoiceName('');
        setIsAdding(false);
        setSuccessMessage('Voice sample added successfully!');
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        throw new Error(result.error || 'Failed to add voice sample.');
      }
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : 'An unexpected error occurred.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleAddVoiceSample(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleAddVoiceSample(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => e.preventDefault();
  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleSelectVoiceSample = (id: string) => {
    console.log('Selecting voice sample:', id);
    storeRef.current.selectVoiceSample?.(id);
  };

  const handleStartEdit = (sample: VoiceSample) => {
    setEditingId(sample.id);
    setEditName(sample.name);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  const handleSaveEdit = () => {
    if (!editingId || !editName.trim()) {
      setConnectionError('Please enter a valid name for the voice sample.');
      return;
    }
    storeRef.current.updateVoiceSample?.(editingId, { name: editName.trim() });
    setEditingId(null);
    setEditName('');
    setSuccessMessage(`Voice name updated to "${editName.trim()}" successfully!`);
    setTimeout(() => setSuccessMessage(null), 3000);
  };

  const [isDeletingId, setIsDeletingId] = useState<string | null>(null);

  const handleDeleteVoiceSample = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this voice sample?')) return;
    try {
      setIsDeletingId(id);
      if (isElectronAvailable && window.electron?.deleteVoiceSample) {
        const result = await window.electron.deleteVoiceSample(id) as unknown as Result<any>;
        if (!result?.success) {
          throw new Error(result?.error || 'Delete failed');
        }
      }
      storeRef.current.removeVoiceSample?.(id);
      setSuccessMessage('Voice sample deleted');
      setTimeout(() => setSuccessMessage(null), 2000);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : 'Failed to delete voice sample');
    } finally {
      setIsDeletingId(null);
    }
  };

  const handleChangeDirectory = async () => {
    if (!window.electron?.selectDirectory || !window.electron?.setVoiceSamplesDirectory) return;
    try {
      const dirResult = await window.electron.selectDirectory();
      if (dirResult.success && dirResult.result) {
        const newDirectory = dirResult.result;
        const setResult = await window.electron.setVoiceSamplesDirectory(newDirectory);
        if (setResult.success) {
          const anyResult: any = setResult as any;
          const dir = typeof anyResult.result === 'string' ? anyResult.result : (anyResult.result?.directory || newDirectory);
          setVoiceSamplesDirectory(dir);
          // Persist into global settings so it survives navigation
          const cur = storeRef.current.settings;
          storeRef.current.updateSettings?.({ ...cur, voiceSamplesDirectory: dir });
          setSuccessMessage('Directory updated successfully.');
          setTimeout(() => setSuccessMessage(null), 3000);
        } else {
          throw new Error(setResult.error || 'Failed to set directory.');
        }
      }
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : 'Failed to change directory.');
    }
  };

  // Do not block UI on initialization

  if (connectionError) return (
    <div className="alert alert-error">
      <span>{connectionError}</span>
      <div className="flex-none">
        <button className="btn btn-sm" onClick={initializeComponent}>Retry</button>
      </div>
    </div>
  );

  if (!settings?.voiceCloning) {
    return <div>Voice cloning settings not available.</div>;
  }

  const { voiceCloning } = settings;
  const voiceSamples = voiceCloning.voiceSamples || [];
  
  // Debug logging
  console.log('VoiceCloningSettings render:', {
    voiceCloning,
    voiceSamples: voiceSamples.length,
    selectedVoiceSampleId: voiceCloning.selectedVoiceSampleId,
    hasSelectedSample: voiceCloning.selectedVoiceSampleId && voiceSamples.some(s => s.id === voiceCloning.selectedVoiceSampleId)
  });

  return (
    <div className="space-y-3 p-3">
      <h2 className="text-2xl font-bold text-primary">Voice Cloning Settings</h2>
      {successMessage && <div className="alert alert-success"><span>{successMessage}</span></div>}

      <div className="form-control">
        <label className="label cursor-pointer">
          <span className="label-text font-semibold">Enable Voice Cloning</span>
          <input
            type="checkbox"
            className="toggle toggle-primary"
            checked={voiceCloning.enabled || false}
            onChange={(e) => storeRef.current.updateSettings?.({ ...settings, voiceCloning: { ...voiceCloning, enabled: e.target.checked } })}
          />
        </label>
      </div>

      {voiceCloning.enabled && (
        <>
          {/* Directory visible at first level */}
          <div className="form-control">
            <label className="label">
              <span className="label-text">Voice Samples Directory</span>
            </label>
            <div className="flex items-center space-x-2">
              <span className="p-2 bg-base-200 rounded flex-grow font-mono truncate">
                {typeof voiceSamplesDirectory === 'string' ? voiceSamplesDirectory : voiceSamplesDirectory?.directory || 'Not set'}
              </span>
              {isElectronAvailable && <button className="btn btn-sm btn-outline" onClick={handleChangeDirectory}>Change</button>}
            </div>
          </div>

          {/* Voice Library (always visible) */}
          <div className="space-y-3 p-3 border border-base-300 rounded-lg">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Voice Library</h3>
              <button 
                className="btn btn-sm btn-outline" 
                onClick={() => loadVoiceSamplesFromBackend()}
                disabled={isLoadingVoiceSamples}
                title="Refresh voice samples from backend"
              >
                {isLoadingVoiceSamples ? (
                  <span className="loading loading-spinner loading-xs"></span>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                )}
              </button>
            </div>
            <div className="space-y-2">
              {isLoadingVoiceSamples ? (
                <div className="text-center text-base-content/60 py-4">
                  <span className="loading loading-spinner loading-sm mr-2"></span>
                  Loading voice samples...
                </div>
              ) : voiceSamples.length === 0 ? (
                <div className="text-center text-base-content/60 py-4">No voice samples added.</div>
              ) : (
                voiceSamples.map((sample) => (
                   <div key={sample.id} className="bg-base-200 rounded-lg p-2 flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <input
                        type="radio"
                        name="voiceSample"
                        className="radio radio-primary"
                        checked={voiceCloning.selectedVoiceSampleId === sample.id}
                        onChange={() => handleSelectVoiceSample(sample.id)}
                      />
                      {editingId === sample.id ? (
                         <div className="flex items-center gap-2 min-w-0">
                          <input 
                            ref={sample.id === newlyAddedId ? editInputRef : null}
                             type="text"
                             className="input input-sm w-48"
                            value={editName} 
                            onChange={(e) => setEditName(e.target.value)} 
                            onKeyDown={(e) => e.key === 'Enter' && handleSaveEdit()} 
                            autoFocus 
                          />
                          <button className="btn btn-sm btn-primary" onClick={handleSaveEdit}>Save</button>
                          <button className="btn btn-sm btn-ghost" onClick={handleCancelEdit}>Cancel</button>
                        </div>
                      ) : (
                        <div className="min-w-0">
                          <div className="font-medium truncate max-w-[16rem]" title={sample.name?.trim() || 'Unnamed Voice'}>
                            {sample.name?.trim() ? sample.name : 'Unnamed Voice'}
                          </div>
                          {/* Hide long file details to keep actions visible; show as title tooltip */}
                          <div className="text-xs text-base-content/70 truncate max-w-[16rem]" title={`${sample.fileName} • ${formatFileSize(sample.fileSize || 0)}`}>
                            {sample.fileName} • {formatFileSize(sample.fileSize || 0)}
                          </div>
                        </div>
                      )}
                    </div>
                    {editingId !== sample.id && (
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <button className="btn btn-ghost btn-sm" onClick={() => handleStartEdit(sample)}>Rename</button>
                        <button className={`btn btn-ghost btn-sm text-error ${isDeletingId===sample.id ? 'loading' : ''}`} disabled={isDeletingId===sample.id} onClick={() => handleDeleteVoiceSample(sample.id)}>
                          {isDeletingId===sample.id ? '' : 'Delete'}
                        </button>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            {!isAdding ? (
              <button className="btn btn-outline w-full mt-2" onClick={() => setIsAdding(true)}>Add Voice Sample</button>
            ) : (
              <div className="space-y-3 mt-4">
                <input type="text" placeholder="Enter voice name..." className={`input input-bordered w-full ${!newVoiceName.trim() ? 'input-error' : ''}`} value={newVoiceName} onChange={(e) => setNewVoiceName(e.target.value)} />
                <div className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer ${isDragging ? 'border-primary' : ''}`} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onDragEnter={handleDragEnter} onClick={() => document.getElementById('file-input')?.click()}>
                  {isUploading ? <span className="loading loading-spinner"></span> : 'Drop audio file or click to select'}
                </div>
                <input id="file-input" type="file" accept="audio/*" className="hidden" onChange={handleFileSelect} />
                <button className="btn btn-ghost" onClick={() => { setIsAdding(false); setNewVoiceName(''); }}>Cancel</button>
              </div>
            )}
          </div>

          <button className="btn btn-primary btn-sm w-full" onClick={() => setShowAdvanced(s => !s)}>
            {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
          </button>

          {showAdvanced && (
            <>
              <div className="form-control">
                <label className="label">
                  <span>Exaggeration</span>
                  <span>{voiceCloning.exaggeration?.toFixed(2)}</span>
                </label>
                <input type="range" min="0" max="1" step="0.01" value={voiceCloning.exaggeration || 0.5} className="range range-primary" onChange={(e) => storeRef.current.updateSettings?.({ ...settings, voiceCloning: { ...voiceCloning, exaggeration: parseFloat(e.target.value) } })} />
              </div>

              <div className="form-control">
                <label className="label">
                  <span>Voice Adherence</span>
                  <span>{voiceCloning.cfgWeight?.toFixed(2)}</span>
                </label>
                <input type="range" min="0" max="1" step="0.01" value={voiceCloning.cfgWeight || 0.5} className="range range-secondary" onChange={(e) => storeRef.current.updateSettings?.({ ...settings, voiceCloning: { ...voiceCloning, cfgWeight: parseFloat(e.target.value) } })} />
              </div>

              {/* Tips panel for advanced tuning */}
              <div className="p-3 mt-2 rounded-lg border border-base-300 bg-base-100 text-sm leading-relaxed">
                <div className="font-semibold mb-1">Tips</div>
                <div className="mb-1">
                  <span className="font-medium">General Use (TTS and Voice Agents):</span>
                  <ul className="list-disc ml-5 mt-1">
                    <li>The default settings (exaggeration = 0.5, voice adherence = 0.5) work well for most prompts.</li>
                    <li>If the reference speaker talks fast, try lowering voice adherence to about 0.3 to improve pacing.</li>
                  </ul>
                </div>
                <div>
                  <span className="font-medium">Expressive or Dramatic Speech:</span>
                  <ul className="list-disc ml-5 mt-1">
                    <li>Lower voice adherence (≈ 0.3) and increase exaggeration (≈ 0.7 or higher) for more emotion.</li>
                    <li>Higher exaggeration often speeds up speech; reduce voice adherence to slow and clarify delivery.</li>
                    <li>
                      If it sounds too fast: decrease exaggeration a bit or reduce voice adherence further until pacing feels right.
                    </li>
                  </ul>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default VoiceCloningSettings;
