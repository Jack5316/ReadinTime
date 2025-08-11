import React from 'react'
import useStore from '../../store/useStore';
import '../../global';

const ChangeBookPathButton = () => {
  const { settings, updateSettings } = useStore();

  const selectFolder = async () => {
    try {
      console.log("selectFolder called");
      console.log("window.electron:", window.electron);
      console.log("window.electron.selectDirectory:", window.electron?.selectDirectory);
      
      if (!window.electron || !window.electron.selectDirectory) {
        alert("Electron API not available. Please run the app with Electron (npm run start:with-vite)");
        return;
      }
      
      const result = await window.electron.selectDirectory();
      console.log("selectDirectory result:", result);

      if (result.success && result.result) {
        updateSettings({ ...settings, bookPath: result.result });
      } else {
        const errorMsg = result.success ? 'Unknown error' : result.error;
        console.error('Failed to select directory:', errorMsg);
        alert('Failed to select directory: ' + errorMsg);
      }
    } catch (error) {
      console.error('Error selecting folder:', error);
      alert('Error selecting folder: ' + error);
    }
  }

  return (
    <button
      onClick={selectFolder}
      className="btn btn-secondary"
    >
      Select Books Folder
    </button>
  )
}

export default ChangeBookPathButton
