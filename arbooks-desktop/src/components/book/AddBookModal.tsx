import React, { forwardRef, useState, useEffect, useRef } from 'react'
import { HiArrowUpTray, HiOutlineXMark, HiExclamationTriangle, HiCheckCircle } from 'react-icons/hi2'
import useStore from '../../store/useStore';
import useListBooks from '../../hooks/useListBooks';
import '../../global';

interface ProcessingStatus {
  stage: string;
  progress: number;
  message: string;
  status: 'processing' | 'completed' | 'failed';
  files_created?: Record<string, boolean>;
}

const AddBookModal = forwardRef<HTMLDialogElement, {}>((props, addBookModalRef) => {
  const [title, setTitle] = useState("");
  const [author, setAuthor] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState("");
  const [uploadInfo, setUploadInfo] = useState("");
  const [success, setSuccess] = useState("");
  const [progress, setProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [voicePromptFile, setVoicePromptFile] = useState<File | null>(null);
  // Uploading/job progress moved to global store to persist across navigation
  const useEnhancedPipeline = true; // Always use enhanced pipeline for better UX
  const [useVoiceCloning, setUseVoiceCloning] = useState(false);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const { listBooks } = useListBooks();
  const { 
    settings: { bookPath, voiceCloning },
    processing,
    startProcessing,
    updateProcessingStatus,
    clearProcessing,
  } = useStore();

  // Performance tracking
  const processingStartTime = useRef<number | null>(null);
  const bookCharacterCount = useRef<number>(0);

  // Debug state changes (can be removed in production)
  // console.log(`[DEBUG] AddBookModal render - jobId: ${jobId}, uploading: ${uploading}, progress: ${progress}`);

  // Check system status on component mount
  useEffect(() => {
    // No-op: removed system status check from Add Book to keep UI minimal
  }, []);

  // Poll processing status when we have a job ID (persisted in store)
  const jobNotFoundAttempts = useRef<number>(0);
  useEffect(() => {
    if (processing.currentJobId && processing.uploading) {
      const pollStatus = async () => {
        try {
          const result = await window.electron.getProcessingStatus(processing.currentJobId!);
          
          if (result.success) {
            jobNotFoundAttempts.current = 0;
            const status = result.result as ProcessingStatus;
            updateProcessingStatus(status);
            setUploadInfo(status.message);
            setProgress(status.progress);

            // Log processing progress
            if (processingStartTime.current) {
              const elapsedTime = (Date.now() - processingStartTime.current) / 1000;
              const wordCount = Math.round(bookCharacterCount.current / 5);
              
              // Enhanced progress logging for large books
              if (wordCount > 8000) {
                const estimatedTotalTime = (wordCount / 1000) * 60; // ~1 minute per 1000 words
                const remainingTime = Math.max(0, estimatedTotalTime - elapsedTime);
                console.log(`⏳ Large Book Progress: ${status.progress}% - ${status.stage}`);
                console.log(`   📊 Words: ${wordCount.toLocaleString()} | Elapsed: ${(elapsedTime/60).toFixed(1)}m | ETA: ${(remainingTime/60).toFixed(1)}m`);
              } else {
                console.log(`⏳ Progress Update: ${status.progress}% - ${status.stage} (${elapsedTime.toFixed(1)}s elapsed)`);
              }
            }

            if (status.status === 'completed') {
              setSuccess(`Book "${title}" processed successfully!`);
              
              // Log performance metrics
              if (processingStartTime.current) {
                const processingTime = (Date.now() - processingStartTime.current) / 1000;
                const wordCount = Math.round(bookCharacterCount.current / 5);
                const wordsPerSecond = Math.round(wordCount / processingTime);
                
                console.log(`📊 BOOK PROCESSING COMPLETE:`);
                console.log(`📖 Title: "${title}"`);
                console.log(`📝 Characters: ${bookCharacterCount.current.toLocaleString()}`);
                console.log(`📊 Words: ${wordCount.toLocaleString()}`);
                console.log(`⏱️ Processing Time: ${processingTime.toFixed(2)} seconds (${(processingTime / 60).toFixed(1)} minutes)`);
                console.log(`🚀 Speed: ${(bookCharacterCount.current / processingTime).toFixed(0)} chars/sec, ${wordsPerSecond} words/sec`);
                console.log(`📄 File Size: ${selectedFile ? (selectedFile.size / 1024 / 1024).toFixed(2) : 'Unknown'} MB`);
                console.log(`🎵 Voice Cloning: ${useVoiceCloning ? 'Enabled' : 'Disabled'}`);
                
                // Show efficiency metrics for large books
                if (wordCount > 8000) {
                  const efficiency = processingTime / (wordCount / 1000); // seconds per 1k words
                  console.log(`⚡ Large Book Efficiency: ${efficiency.toFixed(1)} seconds per 1,000 words`);
                }
                
                console.log(`─────────────────────────────────────────────────────`);
              }
              
              if (bookPath) listBooks(bookPath);
              clearProcessing();
              resetForm();
            } else if (status.status === 'failed') {
              setError(status.message);
              
              // Log failed processing metrics
              if (processingStartTime.current) {
                const processingTime = (Date.now() - processingStartTime.current) / 1000;
                console.log(`❌ BOOK PROCESSING FAILED:`);
                console.log(`📖 Title: "${title}"`);
                console.log(`⏱️ Failed after: ${processingTime.toFixed(2)} seconds`);
                console.log(`💥 Error: ${status.message}`);
                console.log(`─────────────────────────────────────────────────────`);
              }
              clearProcessing();
            }
          } else {
            console.error(`Status poll failed:`, result.error);
            // Gracefully retry a few times on transient "Job not found"
            const errMsg = (result.error || '').toLowerCase();
            if (errMsg.includes('job not found') && jobNotFoundAttempts.current < 3) {
              jobNotFoundAttempts.current += 1;
              return; // keep polling
            }
            // If the backend lost the job (e.g., app reload), stop polling and unlock UI.
            // For 'job not found' specifically, clear silently (no error banner).
            if (errMsg.includes('job not found')) {
              setUploadInfo("");
              setProgress(0);
              clearProcessing();
              return;
            }
            setError(result.error || 'Processing failed, please try again.');
            clearProcessing();
          }
        } catch (err) {
          console.error('Error polling status:', err);
          // Stop polling on repeated failures to avoid locking the modal
          clearProcessing();
        }
      };

      // Start polling immediately and then every second
      pollStatus();
      const interval = setInterval(pollStatus, 1000);

      return () => {
        clearInterval(interval);
      };
    }
  }, [processing.currentJobId, processing.uploading, title, bookPath, listBooks, updateProcessingStatus, clearProcessing]);

  const checkSystemStatus = async () => {};

  const resetForm = () => {
    setTitle("");
    setAuthor("");
    setDescription("");
    setSelectedFile(null);
    setVoicePromptFile(null);
    setUseVoiceCloning(false);
    
    // Reset performance tracking
    processingStartTime.current = null;
    bookCharacterCount.current = 0;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const file = e.target.files[0];

    if (file.type !== "application/pdf") {
      setError("Please upload a PDF file");
      return;
    }

    setError("");
    setSelectedFile(e.target.files[0]);
  }

  const handleVoicePromptChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const file = e.target.files[0];

    if (!file.type.startsWith("audio/")) {
      setError("Please upload an audio file for voice cloning");
      return;
    }

    setError("");
    setVoicePromptFile(file);
  }

  const uploadBook = async () => {
    setUploadInfo("");
    setSuccess("");
    setProgress(0);
    setError("");

    if (!bookPath) {
      setError("Please choose a folder to store yours books.");
      return;
    }

    if (!selectedFile) {
      setError("Please upload a file");
      return;
    }

    if (!title) {
      setError("Please enter a title");
      return;
    }

    if (!author) {
      setError("Please enter an author");
      return;
    }

    if (description.length < 20) {
      setError("Please enter a description of at least 20 characters");
      return;
    }

    // Validate voice cloning setup
    if (useVoiceCloning && !voicePromptFile) {
      setError("Please upload a voice prompt file for voice cloning");
      return;
    }
    
    // If voice cloning via settings is enabled but no sample is selected,
    // fall back to regular TTS instead of blocking the upload.

    const reader = new FileReader();
    reader.onload = async () => {
      const fileData = reader.result as ArrayBuffer;

      const bookInfo = {
        name: selectedFile.name,
        pdfData: fileData,
        title,
        author,
        description,
        bookPath,
        voiceCloning: (() => {
          // Debug logging
          console.log(`🐛 DEBUG Voice Cloning Settings:`, {
            enabled: voiceCloning.enabled,
            selectedVoiceSampleId: voiceCloning.selectedVoiceSampleId,
            voiceSamples: voiceCloning.voiceSamples?.length || 0,
            useVoiceCloning,
            hasVoicePromptFile: !!voicePromptFile
          });
          
          // Priority 1: Direct upload in modal (if user uploaded file directly)
          if (useVoiceCloning && voicePromptFile) {
            console.log(`🎵 Using DIRECT UPLOAD voice cloning`);
            return {
              enabled: true,
              mode: 'direct_upload',
              exaggeration: voiceCloning.exaggeration,
              cfgWeight: voiceCloning.cfgWeight,
              voicePromptFile: voicePromptFile
            };
          }
          
          // Priority 2: Settings-based voice cloning (if enabled and sample selected)
          if (voiceCloning.enabled && voiceCloning.selectedVoiceSampleId) {
            console.log(`🎵 Using SETTINGS-BASED voice cloning with sample: ${voiceCloning.selectedVoiceSampleId}`);
            return {
              enabled: true,
              mode: 'settings_sample',
              selectedSampleId: voiceCloning.selectedVoiceSampleId,
              exaggeration: voiceCloning.exaggeration,
              cfgWeight: voiceCloning.cfgWeight
            };
          }
          
          // No voice cloning
          console.log(`🚫 NO voice cloning enabled - using standard TTS`);
          return undefined;
        })()
      }

      try {
        // Use the enhanced pipeline manager for better UX
        setUploadInfo("Starting book processing...");
        
        // Start performance tracking
        processingStartTime.current = Date.now();
        
        // Estimate character count (rough estimation based on file size)
        // Average PDF has about 1,800 characters per page, and ~1MB = ~300-500 pages
        const estimatedPages = Math.round((selectedFile.size / 1024 / 1024) * 400);
        bookCharacterCount.current = estimatedPages * 1800;
        
        // Determine book size category
        const wordCount = Math.round(bookCharacterCount.current / 5); // Rough words estimate
        let sizeCategory = "";
        if (wordCount < 1000) sizeCategory = "Short";
        else if (wordCount < 5000) sizeCategory = "Medium";
        else if (wordCount < 15000) sizeCategory = "Long";
        else sizeCategory = "Very Long";
        
        console.log(`🚀 STARTING BOOK PROCESSING:`);
        console.log(`📖 Title: "${title}"`);
        console.log(`📄 File Size: ${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`);
        console.log(`📝 Estimated Characters: ${bookCharacterCount.current.toLocaleString()}`);
        console.log(`📊 Estimated Words: ${wordCount.toLocaleString()} (${sizeCategory} book)`);
        // Voice cloning status
        const voiceCloningStatus = (() => {
          if (useVoiceCloning && voicePromptFile) {
            return `🎵 Voice Cloning: Enabled (Direct Upload - ${voicePromptFile.name})`;
          }
          if (voiceCloning.enabled && voiceCloning.selectedVoiceSampleId) {
            const selectedSample = voiceCloning.voiceSamples?.find(s => s.id === voiceCloning.selectedVoiceSampleId);
            return `🎵 Voice Cloning: Enabled (Settings Sample - ${selectedSample?.name || 'Unknown'})`;
          }
          return `🎵 Voice Cloning: Disabled`;
        })();
        
        console.log(voiceCloningStatus);
        console.log(`⏳ Processing started at: ${new Date().toLocaleTimeString()}`);
        if (wordCount > 8000) {
          console.log(`🔧 Large book detected - using chunked processing for full audio`);
        }
        console.log(`─────────────────────────────────────────────────────`);
        
        const result = await window.electron.processBookComplete(bookInfo);
        
        if (!result.success) {
          throw new Error(result.error);
        }
        
        // Handle double-wrapped response structure
        const actualResult = (result.result as any)?.result || result.result;
        const newJobId = actualResult?.job_id;
        
        if (!newJobId) {
          throw new Error('No job ID received from server');
        }
        
        startProcessing(newJobId, { title, filename: actualResult.filename || selectedFile?.name });
        
        const displayName = actualResult.filename || actualResult.title || selectedFile?.name || 'your book';
        setUploadInfo(`Started processing ${displayName}...`);
        
        // Status polling will continue via useEffect

      } catch (error: any) {
        setError(error.message);
        console.error(error);
        setUploadInfo("");
        
        // Log initial processing failure
        if (processingStartTime.current) {
          const processingTime = (Date.now() - processingStartTime.current) / 1000;
          console.log(`❌ BOOK PROCESSING FAILED (Initial):`);
          console.log(`📖 Title: "${title}"`);
          console.log(`⏱️ Failed after: ${processingTime.toFixed(2)} seconds`);
          console.log(`💥 Error: ${error.message}`);
          console.log(`─────────────────────────────────────────────────────`);
        }
      }
    }
    reader.readAsArrayBuffer(selectedFile);
  };

  return (
    <dialog ref={addBookModalRef} id="addBookModal" className="modal modal-bottom sm:modal-middle">
      <div className="modal-box">
        <h3 className="font-bold text-lg">Add a book</h3>

        {error && (
          <div role="alert" className="alert alert-error mt-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* System status and voice cloning controls removed for a simpler Add Book modal */}

        {uploadInfo && (
          <div className='mt-2 space-y-2'>
            <div role="alert" className="alert alert-info">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="h-6 w-6 shrink-0 stroke-current">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <div>
                <span>{uploadInfo}</span>
                {processing.status && (
                  <div className="text-xs mt-1 opacity-75">
                    Stage: {processing.status.stage} | Status: {processing.status.status}
                  </div>
                )}
              </div>
            </div>
            <progress className="progress progress-info w-full transition-all" value={progress} max="100"></progress>
            {progress > 0 && (
              <div className="text-xs text-center opacity-75">{progress}% Complete</div>
            )}
          </div>
        )}

        {success && (
          <div role="alert" className="alert alert-success mt-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 shrink-0 stroke-current" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{success}</span>
          </div>
        )}

        <div className="my-4 space-y-3">
          <fieldset className="fieldset space-y-2">
            <legend className="fieldset-legend">Title</legend>
            <input
              type="text"
              value={title}
              disabled={processing.uploading}
              onChange={(e) => setTitle(e.target.value)}
              className="input input-secondary w-full"
              placeholder="Alice in Wonderland" />
          </fieldset>

          <fieldset className="fieldset space-y-2">
            <legend className="fieldset-legend">Author</legend>
            <input
              type="text"
              value={author}
              disabled={processing.uploading}
              onChange={(e) => setAuthor(e.target.value)}
              className="input input-secondary w-full"
              placeholder="Lewis Carroll" />
          </fieldset>

          <fieldset className="fieldset space-y-2">
            <legend className="fieldset-legend">Description</legend>
            <textarea
              className="textarea textarea-secondary w-full h-24 resize-none"
              value={description}
              disabled={processing.uploading}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Girl falls into surreal world, meets odd creatures, adventures ensue."
            ></textarea>
          </fieldset>

          <fieldset className="fieldset space-y-2">
            <legend className="fieldset-legend">Books must be uploaded in PDF Format</legend>
            <input
              type="file"
              disabled={processing.uploading}
              onChange={handleFileChange}
              className="file-input file-input-secondary w-full" />
          </fieldset>



          {/* Voice cloning controls removed */}
        </div>

        <div className="space-x-3 mt-5">
          <button
            onClick={uploadBook}
            disabled={processing.uploading}
            className="btn btn-secondary">
            <HiArrowUpTray className='w-5 h-5' />
            {processing.uploading ? 'Processing...' : 'Add Book'}
          </button>
          
          
        </div>

        <form method="dialog">
          <button className="btn btn-sm btn-circle btn-ghost absolute right-2 top-2"><HiOutlineXMark className='w-5 h-5' /></button>
        </form>

      </div>
    </dialog>
  )
})

export default AddBookModal;