import React, { FC, useState } from 'react'
import { NavLink } from "react-router"

const Navbar: FC = () => {
  const [showAbout, setShowAbout] = useState(false)
  const [showHelp, setShowHelp] = useState(false)

  return (
    <nav className="w-full navbar bg-base-100">
      <div className="flex-1">
        <NavLink to="/library" className="btn btn-ghost text-2xl">
          ReadInTime
        </NavLink>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1 gap-2">
          <li>
            <NavLink to="/library" className="btn btn-ghost btn-sm">📚 Library</NavLink>
          </li>
          <li>
            <NavLink to="/settings" className="btn btn-ghost btn-sm">⚙️ Settings</NavLink>
          </li>
          <li>
            <button 
              onClick={() => setShowHelp(!showHelp)}
              className="btn btn-ghost btn-sm"
            >
              ❓ Help
            </button>
          </li>
          <li>
            <button 
              onClick={() => setShowAbout(!showAbout)}
              className="btn btn-ghost btn-sm"
            >
              ℹ️ About
            </button>
          </li>
        </ul>
      </div>

      {/* About Modal - Credits and License Only */}
      {showAbout && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-base-100 p-6 rounded-lg shadow-xl max-w-md mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold">ℹ️ About ReadInTime (Voice)</h3>
              <button 
                onClick={() => setShowAbout(false)}
                className="btn btn-sm btn-ghost"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">🎓 Academic Credits</h4>
                <div className="text-sm space-y-2 opacity-80">
                  <p><strong>ReadInTime (Voice)</strong> is developed at the <strong>University College London (UCL)</strong></p>
                  <p><strong>Department of Computer Science</strong></p>
                  <p>Developed by <strong>Jiawei Tan</strong></p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">V1.0 Developers - ReadInTime</h4>
                <div className="text-sm space-y-2 opacity-80">
               
                  <ul className="list-disc list-inside ml-4 space-y-1">
                    <li>Salman Mohd Azhan</li>
                    <li>Pranay Vaka</li>
                    <li>Pratik Rai</li>
                    <li>Sai Tenneti</li>
                  </ul>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">📜 License</h4>
                <div className="text-sm opacity-80">
                  <p><strong>MIT License</strong> - Open Source</p>
                  <p className="text-xs mt-1 opacity-70">
                    This project is open source and available under the MIT License. 
                    You are free to use, modify, and distribute this software.
                  </p>
                </div>
              </div>

              <div className="text-center pt-2">
                <button 
                  onClick={() => setShowAbout(false)}
                  className="btn btn-primary btn-sm"
                >
                  Got it! 👍
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Help Modal - User Guide Only */}
      {showHelp && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-base-100 p-6 rounded-lg shadow-xl max-w-md mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold">❓ How to Use ReadInTime</h3>
              <button 
                onClick={() => setShowHelp(false)}
                className="btn btn-sm btn-ghost"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">📚 Getting Started</h4>
                <ul className="text-sm space-y-2 opacity-80">
                  <li>• <strong>Upload a PDF:</strong> Click "Add Book" and select your PDF file</li>
                  <li>• <strong>Set book details:</strong> Enter title, author, and description</li>
                  <li>• <strong>Choose voice options:</strong> Enable voice cloning if desired</li>
                  <li>• <strong>Process book:</strong> Click "Process Book" to start conversion</li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2">🎵 Voice Cloning</h4>
                <ul className="text-sm space-y-2 opacity-80">
                  <li>• <strong>Upload voice sample:</strong> Record or upload a short audio clip</li>
                  <li>• <strong>Adjust settings:</strong> Fine-tune exaggeration and voice strength</li>
                  <li>• <strong>Generate audio:</strong> Create personalized voice narration</li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2">📖 Reading Your Book</h4>
                <ul className="text-sm space-y-2 opacity-80">
                  <li>• <strong>Open book:</strong> Click on any book in your library</li>
                  <li>• <strong>Audio playback:</strong> Use the audio player controls</li>
                  <li>• <strong>Text sync:</strong> Follow along with highlighted text</li>
                  <li>• <strong>Immersive reading:</strong> Press Ctrl+Alt+F to hide all controls</li>
                  <li>• <strong>Window frame:</strong> Press F11 to hide window borders</li>
                </ul>
              </div>

              <div>
                <h4 className="font-semibold mb-2">⌨️ Quick Shortcuts</h4>
                <ul className="text-sm space-y-1 opacity-80">
                  <li>• <kbd className="kbd kbd-sm">F11</kbd> Toggle window frame (true fullscreen)</li>
                  <li>• <kbd className="kbd kbd-sm">Ctrl</kbd> + <kbd className="kbd kbd-sm">Alt</kbd> + <kbd className="kbd kbd-sm">F</kbd> Toggle immersive reading mode</li>
                </ul>
              </div>

              <div className="text-center pt-2">
                <button 
                  onClick={() => setShowHelp(false)}
                  className="btn btn-primary btn-sm"
                >
                  Got it! 👍
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}

export default Navbar
