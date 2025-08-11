import { useEffect, useRef, useState } from "react";
import { Howl } from "howler";

const useAudioTime = (filePath: string) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [blobURL, setBlobURL] = useState("");
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef<Howl | null>(null);
  const rafIdRef = useRef<number | null>(null);
  
  useEffect(() => {
    let isMounted = true;
    
    const fetchData = async () => {
      console.log("Fetching data for audio file: ", filePath);
      
      try {
        const audioDataResult = await window.electron.getFileData(filePath);
        if (!audioDataResult.success) {
          console.error(audioDataResult.error);
          return;
        }

        // Only proceed if component is still mounted
        if (!isMounted) return;

        const blob = new Blob([new Uint8Array(audioDataResult.result)], { type: 'audio/wav' });
        const newBlobURL = URL.createObjectURL(blob);
        setBlobURL(newBlobURL);

        // Destroy previous instance if it exists
        if (audioRef.current) {
          audioRef.current.unload();
        }

        audioRef.current = new Howl({
          src: [newBlobURL],
          format: ['wav'],
          autoplay: false,
          loop: false,
          volume: 1,
          onplay: () => {
            if (isMounted) {
              setPlaying(true);
              // Start the animation frame loop
              if (rafIdRef.current === null) {
                rafIdRef.current = requestAnimationFrame(updateProgress);
              }
            }
          },
          onend: () => {
            if (isMounted) {
              setPlaying(false);
              // Cancel animation frame when audio ends
              if (rafIdRef.current !== null) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
              }
            }
          },
          onpause: () => {
            if (isMounted) {
              setPlaying(false);
              // Cancel animation frame when audio pauses
              if (rafIdRef.current !== null) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
              }
            }
          },
          onstop: () => {
            if (isMounted) {
              setPlaying(false);
              // Cancel animation frame when audio stops
              if (rafIdRef.current !== null) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
              }
            }
          },
          onload: () => {
            if (isMounted && audioRef.current) {
              setDuration(audioRef.current.duration());
            }
          }
        });
      } catch (error) {
        console.error("Error loading audio:", error);
      }
    };

    fetchData();

    // Cleanup function
    return () => {
      isMounted = false;
      
      // Cancel any ongoing animation frame
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      
      // Unload and destroy the Howl instance
      if (audioRef.current) {
        audioRef.current.unload();
        audioRef.current = null;
      }
      
      // Revoke the blob URL to prevent memory leaks
      if (blobURL) {
        URL.revokeObjectURL(blobURL);
      }
    };
  }, [filePath]);

  const updateProgress = () => {
    if (audioRef.current && audioRef.current.playing()) {
      setCurrentTime(audioRef.current.seek());
      // Store the id so we can cancel it later if needed
      rafIdRef.current = requestAnimationFrame(updateProgress);
    }
  };

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (audioRef.current.playing()) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
  };

  return {
    currentTime,
    audioRef,
    blobURL,
    duration,
    playing,
    togglePlayPause,
  };
};

export default useAudioTime;
