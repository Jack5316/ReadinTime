import React, { FC, useState, useEffect } from 'react'
import useStore from '../store/useStore';
import ChangeBookPathButton from '../components/settings/ChangeBookPathButton';
import VoiceCloningSettings from '../components/settings/VoiceCloningSettings';
import VoiceCloningDemo from '../components/voice/VoiceCloningDemo';
import ErrorBoundary from '../components/ErrorBoundary';
import '../global';

const Settings: FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  console.log('Settings component: Starting render');
  
  let settings;
  let isElectronAvailable;
  
  try {
    const storeData = useStore();
    settings = storeData.settings;
    console.log('Settings component: useStore successful', settings);
    
    // Check if running in Electron environment
    isElectronAvailable = typeof window !== 'undefined' && window.electron;
    console.log('Settings component: isElectronAvailable =', isElectronAvailable);
  } catch (err) {
    console.error('Settings component: Error in initialization:', err);
    setError(err instanceof Error ? err.message : 'Unknown error');
    return (
      <div className="alert alert-error">
        <h3>Settings failed to load</h3>
        <p>Error: {err instanceof Error ? err.message : 'Unknown error'}</p>
        <button onClick={() => window.location.reload()} className="btn btn-sm">Reload</button>
      </div>
    );
  }

  useEffect(() => {
    console.log('Settings component: useEffect running');
    setIsLoading(false);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <span className="loading loading-spinner loading-lg"></span>
        <span className="ml-2">Loading settings...</span>
      </div>
    );
  }

  return (
    <div className='grid gap-4' style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(22rem, 1fr))' }}>
      {!isElectronAvailable && (
        <div className="alert alert-warning w-full">
          <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <span>Some features require Electron. Run the app with <code>npm run start:with-vite</code> for full functionality.</span>
        </div>
      )}
      
      <ErrorBoundary fallback={<div className="alert alert-error">Books folder settings failed to load</div>}>
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body">
            <h2 className="card-title">Books folder</h2>
            <div>
              <p className="text-sm">{settings?.bookPath || 'No path set'}</p>
              <ChangeBookPathButton />
            </div>
          </div>
        </div>
      </ErrorBoundary>

      <ErrorBoundary fallback={<div className="alert alert-error">Voice Cloning Settings failed to load</div>}>
        <div className="card bg-base-100 shadow-xl" style={{ maxHeight: 'calc(100vh - 180px)' }}>
          <div className="card-body overflow-auto">
            <h2 className="card-title">Voice Cloning</h2>
            <VoiceCloningSettings />
          </div>
        </div>
      </ErrorBoundary>

      {/* Voice Cloning Demo - Full width */}
      <ErrorBoundary fallback={<div className="alert alert-error">Voice Cloning Demo failed to load</div>}>
        <div className="min-w-[22rem]" style={{ maxHeight: 'calc(100vh - 180px)', overflow: 'auto' }}>
          <VoiceCloningDemo />
        </div>
      </ErrorBoundary>
    </div>
  )
}

export default Settings
