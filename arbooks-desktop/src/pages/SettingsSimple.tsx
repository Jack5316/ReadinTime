import React, { FC } from 'react'
import ChangeBookPathButton from '../components/settings/ChangeBookPathButton';
import VoiceCloningSettings from '../components/settings/VoiceCloningSettings';
import useStore from '../store/useStore';
import ErrorBoundary from '../components/ErrorBoundary';

const SettingsSimple: FC = () => {
  console.log('SettingsSimple: Component rendering');
  
  const { settings } = useStore();
  
  return (
    <div className='p-4'>
      <h1 className="text-2xl font-bold mb-4">Settings</h1>
      
      <div className="alert alert-info mb-4">
        <span>Testing settings components one by one...</span>
      </div>
      
      {/* Test ChangeBookPathButton */}
      <ErrorBoundary fallback={<div className="alert alert-error">Books folder settings failed to load</div>}>
        <div className="card bg-base-100 shadow-xl mb-4">
          <div className="card-body">
            <h2 className="card-title">Books Folder</h2>
            <div>
              <p className="text-sm mb-2">{settings?.bookPath || 'No path set'}</p>
              <ChangeBookPathButton />
            </div>
          </div>
        </div>
      </ErrorBoundary>

      {/* Test VoiceCloningSettings - MOST LIKELY CULPRIT */}
      <ErrorBoundary fallback={<div className="alert alert-error">🚨 VoiceCloningSettings failed to load - THIS IS LIKELY THE ISSUE!</div>}>
        <div className="card bg-base-100 shadow-xl mb-4">
          <div className="card-body">
            <h2 className="card-title">Voice Cloning Settings</h2>
            <VoiceCloningSettings />
          </div>
        </div>
      </ErrorBoundary>
      
      <div className="card bg-base-100 shadow-xl">
        <div className="card-body">
          <h2 className="card-title">Debug Info</h2>
          <p>✅ Simple page works</p>
          <p>✅ FontSettings works</p>
          <p>✅ ChangeBookPathButton works</p>
          <p>🚨 Testing VoiceCloningSettings component... (likely culprit!)</p>
        </div>
      </div>
    </div>
  )
}

export default SettingsSimple 