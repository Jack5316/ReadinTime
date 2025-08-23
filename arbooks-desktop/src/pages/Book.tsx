import React, { FC, useEffect, useRef, useState } from 'react'
import { HiCog, HiOutlineXMark, HiArrowsPointingIn, HiArrowLeft, HiPlay, HiPause, HiChevronLeft, HiChevronRight } from "react-icons/hi2";
import FlipBook, { FlipBookHandle } from '../components/book/FlipBook';
import { TextSegment, BookData } from '../types/book';
import { useParams, useNavigate } from "react-router-dom"
import useStore from '../store/useStore';
import { FONT_COLORS, BG_COLORS } from '../store/settings';
import { Result } from '../types/result';
import '../global'; // Import global types

const BookPage: FC = () => {
  const params = useParams<{ title: string }>();
  const navigate = useNavigate();
  const [textMappings, setTextMappings] = useState<TextSegment[]>([]);
  const [audioFile, setAudioFile] = useState<string>('output.wav'); // Default to regular audio
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const fontSettingsModalRef = useRef<HTMLDialogElement>(null);
  const bookApiRef = useRef<FlipBookHandle | null>(null);
  const { settings, updateSettings } = useStore();
  const { bookPath } = settings;
  const [currentPage, setCurrentPage] = useState<number>(0);

  const handleToggleAudio = () => {
    if (bookApiRef.current) {
      bookApiRef.current.toggleAudio();
      setIsPlaying(prev => !prev);
    }
  };

  const handleNextPage = () => {
    const api = bookApiRef.current;
    if (!api) return;
    try {
      api.nextPage();
    } catch (e) {
      console.warn('nextPage failed:', e);
    }
  };

  const handlePrevPage = () => {
    const api = bookApiRef.current;
    if (!api) return;
    try {
      api.prevPage();
    } catch (e) {
      console.warn('prevPage failed:', e);
    }
  };

  // Persist current page index on flip events from child
  const onFlip = (pageIndex: number) => setCurrentPage(pageIndex);

  useEffect(() => {
    const handler = (e: any) => setCurrentPage(e?.detail?.index ?? 0);
    window.addEventListener('flipbook:onFlip', handler as EventListener);
    return () => window.removeEventListener('flipbook:onFlip', handler as EventListener);
  }, []);

  // Listen for immersive reading toggle from main process (Ctrl+Alt+F)
  useEffect(() => {
    if (window.electron?.onToggleImmersiveReading) {
      const cleanup = window.electron.onToggleImmersiveReading(() => {
        updateSettings({ ...settings, isFullScreen: !settings.isFullScreen });
      });
      return cleanup;
    }
  }, [settings.isFullScreen, updateSettings]);

  useEffect(() => {
    if (!bookPath) {
      setError('Book path is not set');
      setLoading(false);
      return;
    }

    const path = `${bookPath}/${params.title}`;
    console.log("Path: ", path);
    console.log("Auto-scroll setting:", settings.autoScroll);

    /* Retrieve initial data on page load */
    window.electron.readBook(path)
      .then((result: Result<BookData>) => {
        if (!result.success) throw result.error;

        setTextMappings(result.result.textMappings);
        setAudioFile(result.result.audioFile || 'output.wav'); // Use voice cloned audio if available

        console.log("Text mappings data: ", result.result.textMappings);
        console.log("Audio file: ", result.result.audioFile);
      })
      .catch(err => {
        setError(err.message || 'Failed to load book');
        console.error(err);
      })
      .finally(() => setLoading(false));

  }, [bookPath, params.title, settings.autoScroll])

  return (
    <>
      <dialog ref={fontSettingsModalRef} className="modal modal-bottom sm:modal-middle">
        <div className="modal-box">
          <h3 className="font-bold text-lg">Settings</h3>
          <div className="py-4">
            {/* Add new settings here */}
            <div className="form-control">
              <label className="label">
                <span className="label-text">Font Size: {settings.fontSize}px</span>
              </label>
              <input 
                type="range" 
                min="16" 
                max="48" 
                value={settings.fontSize} 
                onChange={(e) => updateSettings({ ...settings, fontSize: Number(e.target.value) })}
                className="range range-primary"
              />
            </div>
            <div className="form-control">
              <label className="label">
                <span className="label-text">Line Spacing: {settings.lineSpacing}</span>
              </label>
              <input 
                type="range" 
                min="1.2" 
                max="2.5" 
                step="0.1" 
                value={settings.lineSpacing} 
                onChange={(e) => updateSettings({ ...settings, lineSpacing: Number(e.target.value) })}
                className="range range-primary"
              />
            </div>
            <div className="form-control">
              <label className="label">
                <span className="label-text">Word Spacing: {settings.wordSpacing}px</span>
              </label>
              <input 
                type="range" 
                min="0" 
                max="10" 
                step="0.5" 
                value={settings.wordSpacing} 
                onChange={(e) => updateSettings({ ...settings, wordSpacing: Number(e.target.value) })}
                className="range range-primary"
              />
            </div>
            <div className="form-control">
              <label className="label cursor-pointer">
                <span className="label-text">Auto-Scroll</span>
                <input 
                  type="checkbox" 
                  className="toggle toggle-primary"
                  checked={settings.autoScroll}
                  onChange={(e) => updateSettings({ ...settings, autoScroll: e.target.checked })}
                />
              </label>
            </div>
            <div className="color-controls">
              <label>Font Colour</label>
              <div className="flex flex-wrap gap-2">
                {Object.entries(FONT_COLORS).map(([name, hex]) => (
                  <button
                    key={name}
                    className={`btn btn-sm ${settings.fontColour === name ? 'btn-active' : ''}`}
                    style={{ backgroundColor: hex }}
                    onClick={() => updateSettings({ ...settings, fontColour: name as keyof typeof FONT_COLORS })}
                  />
                ))}
              </div>
              <label>Background Colour</label>
              <div className="flex flex-wrap gap-2">
                {Object.entries(BG_COLORS).map(([name, hex]) => (
                  <button
                    key={name}
                    className={`btn btn-sm ${settings.bgColour === name ? 'btn-active' : ''}`}
                    style={{ backgroundColor: hex }}
                    onClick={() => updateSettings({ ...settings, bgColour: name as keyof typeof BG_COLORS })}
                  />
                ))}
              </div>
            </div>
          </div>
          <div className="modal-action">
            <button className="btn" onClick={() => fontSettingsModalRef.current?.close()}>Close</button>
          </div>
        </div>
      </dialog>

      <div className='w-full h-screen flex flex-col bg-base-100' style={{ overflow: 'hidden', backgroundColor: '#f8f4e6' }}>
        { /* HEADER */}
        {!settings.isFullScreen && (
          <header className='w-full flex items-center justify-between px-4 py-2 bg-base-200 border-b border-base-300 flex-shrink-0 z-10'>
            <div className="flex items-center gap-3">
              <button 
                className="btn btn-ghost btn-circle"
                onClick={() => navigate('/')}
              >
                <HiArrowLeft className="h-6 w-6" />
              </button>
              <h3 className='text-lg font-bold'>Book View</h3>
              
              {/* Auto-scroll status indicator */}
              {settings.autoScroll && (
                <div className="badge badge-accent gap-2 animate-pulse">
                  <span className="w-2 h-2 bg-accent-content rounded-full"></span>
                  AUTO SCROLL ON
                </div>
              )}
            </div>

            <div className='flex items-center gap-2'>
              <button
                onClick={() => updateSettings({ ...settings, isFullScreen: !settings.isFullScreen })}
                className="btn btn-ghost btn-sm"
              >
                <HiArrowsPointingIn className='h-5 w-5' />
              </button>

              <button
                onClick={() => fontSettingsModalRef.current?.showModal()}
                className="btn btn-ghost btn-sm"
              >
                <HiCog className='h-5 w-5' />
              </button>
            </div>
          </header>
        )}

        <main className="flex-1 w-full h-full flex flex-col items-center justify-center relative overflow-hidden pb-0">
          <div className='w-full flex-1 flex justify-center items-center min-h-0'>
              <FlipBook
              ref={bookApiRef}
              folder={params.title!}
              textMappings={textMappings}
              audioFile={audioFile}
            />
          </div>

          {/* BOTTOM CONTROLS - Hidden in fullscreen mode */}
          {!settings.isFullScreen && (
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20 transition-all duration-300">
              <div className="flex items-center gap-3 px-5 py-2 rounded-full shadow-xl transition-all duration-300 bg-base-200 border border-base-300">
                <button className="btn btn-circle btn-sm btn-ghost" onClick={handlePrevPage}>
                  <HiChevronLeft className="h-5 w-5" />
                </button>
                
                <button
                  className="btn btn-circle btn-primary btn-md"
                  onClick={handleToggleAudio}
                >
                  {isPlaying ? <HiPause className="h-6 w-6" /> : <HiPlay className="h-6 w-6" />}
                </button>

                <button className="btn btn-circle btn-sm btn-ghost" onClick={handleNextPage}>
                  <HiChevronRight className="h-5 w-5" />
                </button>
              </div>
            </div>
          )}

          {/* EXIT FULLSCREEN BUTTON - Only show exit button in fullscreen mode */}
          {settings.isFullScreen && (
            <div className="absolute top-4 right-4 z-20">
              <div className="tooltip tooltip-left" data-tip="Exit Fullscreen">
                <button
                  onClick={() => updateSettings({ ...settings, isFullScreen: false })}
                  className="btn btn-circle btn-ghost shadow-lg hover:shadow-xl transition-shadow"
                >
                  <HiArrowsPointingIn className='h-6 w-6' />
                </button>
              </div>
            </div>
          )}

          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-base-100 z-50">
              <span className="loading loading-spinner loading-lg"></span>
            </div>
          )}
          {error && (
            <div className="alert alert-error absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
              <HiOutlineXMark className="h-6 w-6" />
              <span>{error}</span>
            </div>
          )}
        </main>
      </div >
    </>
  )
}

export default BookPage
