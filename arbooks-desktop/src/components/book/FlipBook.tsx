import { SettingsState } from '../../store/settings';
import React, { useState, useEffect, useRef, useMemo, useLayoutEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import HTMLFlipBook from 'react-pageflip';
import { TextSegment } from '../../types/book';
import useAudioTime from '../../hooks/useAudioTime';
import useStore from '../../store/useStore';
import { FONT_COLORS, BG_COLORS } from '../../store/settings';
import './FlipBook.css';

export interface FlipBookHandle {
  toggleAudio: () => void;
  isPlaying: () => boolean;
  nextPage: () => void;
  prevPage: () => void;
}

interface FlipBookProps {
  textMappings: TextSegment[];
  folder: string;
  audioFile: string;
}

interface PageSize {
  width: number;
  height: number;
  pageWidth: number;
  pageHeight: number;
  scale?: number;
}

interface BookDimensions {
  scale: number;
  width: number;
  height: number;
}

const DEFAULT_PAGE_SIZE: PageSize = {
  width: 1200,
  height: 800,
  pageWidth: 0,
  pageHeight: 0,
  scale: 1
};

// Page layout constants
const PAGE_PADDING_TOP = 30;
const PAGE_PADDING_SIDE = 35;
const PAGE_PADDING_BOTTOM = 35;
const FOOTER_RESERVED = 40;
const BOOK_WRAPPER_PADDING = 15; // Reduced from 30 to match new padding

const FlipBook = forwardRef<FlipBookHandle, FlipBookProps>(({ textMappings, folder, audioFile }, ref) => {
  const settings = useStore(state => state.settings);
  const updateSettings = useStore(state => state.updateSettings);
  
  // State management
  const [pageSize, setPageSize] = useState<PageSize>(DEFAULT_PAGE_SIZE);
  const [pages, setPages] = useState<TextSegment[][]>([]);
  const [pageTimeRanges, setPageTimeRanges] = useState<Array<{ start: number; end: number }>>([]);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [flipBookState, setFlipBookState] = useState<any>(null);
  const [bookDimensions, setBookDimensions] = useState<BookDimensions>({
    scale: 1,
    width: DEFAULT_PAGE_SIZE.width,
    height: DEFAULT_PAGE_SIZE.height
  });
  const [viewport, setViewport] = useState<{ width: number; height: number }>({
    width: typeof window !== 'undefined' ? window.innerWidth : DEFAULT_PAGE_SIZE.width,
    height: typeof window !== 'undefined' ? window.innerHeight : DEFAULT_PAGE_SIZE.height,
  });
  
  // Refs
  const lastAutoFlippedPage = useRef<number>(-1);
  const lastWordRef = useRef<HTMLSpanElement>(null);
  const pageContainerRef = useRef<HTMLDivElement>(null);
  const bookRef = useRef<any>(null);
  
  // Construct full path to audio file
  const fullAudioPath = `${settings.bookPath}/${folder}/audio/${audioFile}`;
  
  const { currentTime, togglePlayPause, playing } = useAudioTime(fullAudioPath);

  // expose controls to parent
  useImperativeHandle(ref, () => ({
    toggleAudio: () => togglePlayPause(),
    isPlaying: () => playing,
    nextPage: () => {
      const api = (bookRef.current && typeof (bookRef.current as any).pageFlip === 'function')
        ? (bookRef.current as any).pageFlip()
        : null;
      if (api && typeof api.flipNext === 'function') {
        api.flipNext();
      } else if (bookRef.current && typeof (bookRef.current as any).flipNext === 'function') {
        (bookRef.current as any).flipNext();
      }
    },
    prevPage: () => {
      const api = (bookRef.current && typeof (bookRef.current as any).pageFlip === 'function')
        ? (bookRef.current as any).pageFlip()
        : null;
      if (api && typeof api.flipPrev === 'function') {
        api.flipPrev();
      } else if (bookRef.current && typeof (bookRef.current as any).flipPrev === 'function') {
        (bookRef.current as any).flipPrev();
      }
    },
  }), [togglePlayPause, playing]);

  // Memoized page styles to prevent unnecessary re-renders
  const pageStyle = useMemo(() => ({
    fontSize: `${settings.fontSize}px`,
    lineHeight: settings.lineSpacing,
    letterSpacing: `${settings.wordSpacing}px`,
    color: FONT_COLORS[settings.fontColour as keyof typeof FONT_COLORS],
    backgroundColor: BG_COLORS[settings.bgColour as keyof typeof BG_COLORS],
    fontFamily: settings.fontStyle
  }), [settings.fontSize, settings.lineSpacing, settings.wordSpacing, settings.fontColour, settings.bgColour, settings.fontStyle]);

  // Track viewport for responsive centering and width
  useEffect(() => {
    const onResize = () => {
      setViewport({ width: window.innerWidth, height: window.innerHeight });
      // Log requested screen information
      // Using both window and screen to help tune for TVs/projectors
      // eslint-disable-next-line no-console
      console.log('[FlipBook] viewport', window.innerWidth, window.innerHeight);
      // eslint-disable-next-line no-console
      console.log('[FlipBook] screen', window.screen.width, window.screen.height);
    };
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Compute a comfortable paragraph width based on aspect ratio
  const paragraphWidthPercent = useMemo(() => {
    const aspect = viewport.width / Math.max(1, viewport.height);
    if (aspect >= 2.0) return 48;      // ultra-wide / many TVs
    if (aspect >= 1.7) return 56;      // 16:9
    if (aspect >= 1.5) return 62;      // 3:2 / 16:10
    if (aspect >= 1.3) return 70;      // typical laptops
    return 82;                         // portrait or squarer screens
  }, [viewport.width, viewport.height]);

  // Dynamic outer padding and available height for the flipbook area
  const layoutMetrics = useMemo(() => {
    const aspect = viewport.width / Math.max(1, viewport.height);
    const clampPx = (min: number, preferred: number, max: number) => Math.min(max, Math.max(min, preferred));

    const headerReserve = settings.isFullScreen ? 0 : 64; // approximate header height
    const controlsReserve = 0; // controls overlay floats; don't subtract here

    const topFactor = aspect >= 1.7 ? 0.10 : 0.08;    // increase top padding for wide screens
    const bottomFactor = aspect >= 1.7 ? 0.12 : 0.10; // give a bit more space at the bottom

    const paddingTop = clampPx(36, Math.round(viewport.height * topFactor), 180);
    const paddingBottom = clampPx(32, Math.round(viewport.height * bottomFactor), 180);

    const availableHeight = Math.max(
      420,
      viewport.height - headerReserve - controlsReserve - paddingTop - paddingBottom
    );

    return { paddingTop, paddingBottom, availableHeight };
  }, [viewport.width, viewport.height, settings.isFullScreen]);

  // Pagination: pack up to 3 sentences but ensure content fits by measuring
  const performPagination = useCallback(() => {
    if (textMappings.length === 0) return;

    try {
      // 1) First split the whole content into sentences (arrays of segments)
      const sentences: TextSegment[][] = [];
      let currentSentence: TextSegment[] = [];
      textMappings.forEach((segment) => {
        currentSentence.push(segment);
        const t = segment.text.trim();
        if (t.endsWith('.') || t.endsWith('!') || t.endsWith('?')) {
          sentences.push(currentSentence);
          currentSentence = [];
        }
      });
      if (currentSentence.length > 0) sentences.push(currentSentence);

      // 2) Measure available text area per page and greedily pack sentences
      const containerWidth = pageContainerRef.current?.clientWidth ?? viewport.width;
      const containerHeight = pageContainerRef.current?.clientHeight ?? viewport.height;
      const pageWidth = (containerWidth / 2) - (PAGE_PADDING_SIDE * 2);
      const contentWidth = pageWidth * (paragraphWidthPercent / 100);
      const topPad = PAGE_PADDING_TOP + 20;
      const bottomPad = PAGE_PADDING_TOP;
      const numberSpace = 34;
      const maxContentHeight = Math.max(200, containerHeight - topPad - bottomPad - numberSpace);

      const meas = document.createElement('div');
      meas.style.cssText = `position:absolute;visibility:hidden;top:-9999px;left:-9999px;width:${contentWidth}px;`+
        `font-size:${settings.fontSize}px;line-height:${settings.lineSpacing};letter-spacing:${settings.wordSpacing}px;`+
        `font-family:${settings.fontStyle};word-spacing:0.1em;text-align:left;hyphens:auto;overflow-wrap:break-word;word-break:break-word;`;
      document.body.appendChild(meas);

      const renderHTML = (sents: TextSegment[][]) => `<div>${sents.map(s => s.map(seg => seg.text).join(' ')).join(' ')}</div>`;

      const newPages: TextSegment[][] = [];
      let idx = 0;
      while (idx < sentences.length) {
        let take = 1;
        for (let tryTake = 3; tryTake >= 1; tryTake--) {
          const slice = sentences.slice(idx, Math.min(idx + tryTake, sentences.length));
          meas.innerHTML = renderHTML(slice);
          if (meas.offsetHeight <= maxContentHeight) {
            take = tryTake;
            break;
          }
        }
        const pageSlice = sentences.slice(idx, Math.min(idx + take, sentences.length));
        newPages.push(pageSlice.flat());
        idx += take;
      }

      document.body.removeChild(meas);

      setPages(newPages);
      // compute time ranges for auto-scroll
      const ranges = newPages.map((pg) => {
        const starts = pg.map(s => s.start);
        const ends = pg.map(s => s.end);
        return { start: Math.min(...starts), end: Math.max(...ends) };
      });
      setPageTimeRanges(ranges);
    } catch (error) {
      console.error('[FlipBook] Error during pagination:', error);
      setPages([textMappings]);
    }
  }, [textMappings, viewport.width, viewport.height, paragraphWidthPercent, settings.fontSize, settings.lineSpacing, settings.wordSpacing, settings.fontStyle]);

  // Helper to programmatically go to a specific page index
  const goToPage = useCallback((targetIndex: number) => {
    const instance: any = bookRef.current;
    if (!instance) return;
    const api = typeof instance.pageFlip === 'function' ? instance.pageFlip() : null;
    if (api && typeof api.flip === 'function') {
      api.flip(targetIndex);
      return;
    }
    if (api && typeof api.turnToPage === 'function') {
      api.turnToPage(targetIndex);
      return;
    }
    if (typeof instance.flip === 'function') {
      instance.flip(targetIndex);
      return;
    }
    if (typeof instance.turnToPage === 'function') {
      instance.turnToPage(targetIndex);
    }
  }, []);

  // Auto-scroll: sync page with audio time when enabled
  useEffect(() => {
    if (!settings.autoScroll || pageTimeRanges.length === 0) return;

    const t = currentTime;
    const idx = pageTimeRanges.findIndex(r => t >= r.start && t <= r.end);
    if (idx === -1) return;

    if (idx !== currentPageIndex && lastAutoFlippedPage.current !== idx) {
      lastAutoFlippedPage.current = idx;
      goToPage(idx);
    }
  }, [currentTime, settings.autoScroll, pageTimeRanges, currentPageIndex, goToPage]);

  // Pagination effect
  useLayoutEffect(() => {
    const runPagination = async () => {
      // Wait for fonts to load
      if (document.fonts && document.fonts.ready) {
        await document.fonts.ready;
      }
      
      // Small delay to ensure DOM is ready
      setTimeout(performPagination, 10);
    };

    runPagination();
  }, [performPagination]);

  // Apply font preferences globally (memoized to prevent unnecessary updates)
  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--font-size', `${settings.fontSize}px`);
    root.style.setProperty('--line-height', `${settings.lineSpacing}`);
    root.style.setProperty('--letter-spacing', `${settings.wordSpacing}px`);
    root.style.setProperty('--font-family', settings.fontStyle);
  }, [settings.fontSize, settings.lineSpacing, settings.wordSpacing, settings.fontStyle]);

  // Load saved preferences (optimized)
  useEffect(() => {
    try {
      const savedPrefs = localStorage.getItem('readingPreferences');
      if (!savedPrefs) return;
      
      const loadedPrefs = JSON.parse(savedPrefs);
      const validKeys = ['fontSize', 'lineSpacing', 'wordSpacing', 'fontStyle', 'fontColour', 'bgColour', 'autoPageTurn', 'autoPageTurnDelay'];
      
      const updates = validKeys.reduce((acc, key) => {
        if (key in loadedPrefs && loadedPrefs[key] !== undefined) {
          acc[key as keyof SettingsState] = loadedPrefs[key];
        }
        return acc;
      }, {} as Partial<SettingsState>);
      
      if (Object.keys(updates).length > 0) {
        updateSettings({ ...settings, ...updates });
      }
    } catch (error) {
      console.error('[FlipBook] Error loading preferences:', error);
    }
  }, []);

  // Optimized settings save function
  const saveSettings = useCallback(() => {
    try {
      const readingPrefs = {
        fontSize: settings.fontSize,
        lineSpacing: settings.lineSpacing,
        wordSpacing: settings.wordSpacing,
        fontStyle: settings.fontStyle,
        fontColour: settings.fontColour,
        bgColour: settings.bgColour,
        autoPageTurn: settings.autoPageTurn,
        autoPageTurnDelay: settings.autoPageTurnDelay
      };
      localStorage.setItem('readingPreferences', JSON.stringify(readingPrefs));
    } catch (error) {
      console.error('[FlipBook] Error saving preferences:', error);
    }
  }, [settings]);

  // Memoized book wrapper dimensions
  const bookWrapperStyle = useMemo(() => ({
    width: `min(100vw, 100%)`,
    height: settings.isFullScreen ? `${viewport.height}px` : `${layoutMetrics.availableHeight}px`,
    maxWidth: `${bookDimensions.width * 2}px`,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: BG_COLORS[settings.bgColour as keyof typeof BG_COLORS]
  }), [bookDimensions, layoutMetrics.availableHeight, settings.bgColour, settings.isFullScreen, viewport.height]);

  // Memoized flip book props
  const flipBookProps = useMemo(() => ({
    width: (pageSize.width - 80) / 2,
    height: Math.max(420, (settings.isFullScreen ? viewport.height : layoutMetrics.availableHeight) - 0),
    flippingTime: 600,
    size: "stretch" as const,
    autoSize: true,
    minWidth: Math.floor(bookDimensions.width * 0.4),
    maxWidth: Math.floor(bookDimensions.width * 0.6),
    minHeight: Math.floor(bookDimensions.height * 0.8),
    maxHeight: Math.floor(bookDimensions.height * 1.2),
    showCover: false,
    mobileScrollSupport: false,
    startPage: currentPageIndex,
    drawShadow: true,
    usePortrait: false,
    startZIndex: 0,
    maxShadowOpacity: 0.5,
    showPageCorners: true,
    disableFlipByClick: false,
    clickEventForward: true,
    useMouseEvents: true,
    swipeDistance: 30
  }), [pageSize, bookDimensions, layoutMetrics.availableHeight, settings.isFullScreen, viewport.height, currentPageIndex]);

  return (
    <div
      className="w-full h-full flex flex-col items-center justify-center book-reader-container"
      style={{
        paddingTop: 0,
        paddingBottom: 0,
        boxSizing: 'border-box',
        height: '100vh',
        backgroundColor: BG_COLORS[settings.bgColour as keyof typeof BG_COLORS],
        overflow: 'hidden',
        width: '100vw'
      }}
    >
      {/* Main book content */}
      <div className="book-content-wrapper">
        {/* HTMLFlipBook and page content */}
        <div
          ref={pageContainerRef}
          className="flex justify-center items-center"
          style={bookWrapperStyle}
        >
          <div
            className="book-wrapper"
            style={{
              position: 'relative',
              padding: 0,
              borderRadius: '12px',
              boxShadow: '0 18px 40px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06)'
            }}
          >
            {/* Removed external top/bottom shadows to avoid visible black bands */}
            {/* Book spine - centered dark line */}
            <div
              className="book-spine"
              style={{
                position: 'absolute',
                left: '50%',
                top: 0,
                bottom: 0,
                width: '3px',
                transform: 'translateX(-50%)',
                background: 'linear-gradient(180deg, #999 0%, #444 50%, #999 100%)',
                boxShadow: 'none',
                zIndex: 20,
                borderRadius: '1px'
              }}
            />

            <HTMLFlipBook
              key={`${folder}-${pages.length}-${bookDimensions.scale}`}
              {...flipBookProps}
              className="flipbook-container"
              style={{ 
                ...pageStyle,
                boxShadow: 'none',
                filter: 'none'
              }}
              onChangeState={setFlipBookState}
              onFlip={(e) => {
                const idx = e?.data ?? 0;
                setCurrentPageIndex(idx);
                lastAutoFlippedPage.current = -1;
                // notify parent if provided
                try {
                  (window as any).dispatchEvent(new CustomEvent('flipbook:onFlip', { detail: { index: idx } }));
                } catch {}
              }}
              ref={bookRef}
            >
              {pages.map((page, index) => {
                const isLeftPage = index % 2 === 0;
                return (
                  <div 
                    key={index} 
                    className={`page ${isLeftPage ? 'page-left' : 'page-right'}`}
                    style={{ 
                      position: 'relative',
                      background: BG_COLORS[settings.bgColour as keyof typeof BG_COLORS],
                      padding: '50px 45px 60px 45px',
                      display: 'flex',
                      flexDirection: 'column',
                      overflow: 'hidden',
                      backgroundImage: isLeftPage 
                        ? `linear-gradient(90deg, 
                            #FFFFFF 0%, 
                            #FFFFFF 80%, 
                            #F8F8F8 90%, 
                            #EEEEEE 95%, 
                            #DDDDDD 100%)`
                        : `linear-gradient(270deg, 
                            #FFFFFF 0%, 
                            #FFFFFF 80%, 
                            #F8F8F8 90%, 
                            #EEEEEE 95%, 
                            #DDDDDD 100%)`,
                      boxShadow: isLeftPage 
                        ? `inset -20px 0 40px -20px rgba(0,0,0,0.15),
                           inset -2px 0 4px rgba(0,0,0,0.05),
                           -3px 0 10px rgba(0,0,0,0.1)`
                        : `inset 20px 0 40px -20px rgba(0,0,0,0.15),
                           inset 2px 0 4px rgba(0,0,0,0.05),
                           3px 0 10px rgba(0,0,0,0.1)`
                    }}
                  >
                    {/* Inner spine shadow */}
                    <div 
                      className="spine-shadow"
                      style={{
                        position: 'absolute',
                        top: 0,
                        [isLeftPage ? 'right' : 'left']: 0,
                        width: '80px',
                        height: '100%',
                        background: isLeftPage
                          ? 'linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.03) 60%, rgba(0,0,0,0.08) 100%)'
                          : 'linear-gradient(270deg, transparent 0%, rgba(0,0,0,0.03) 60%, rgba(0,0,0,0.08) 100%)',
                        pointerEvents: 'none',
                        zIndex: 1
                      }}
                    />
                    
                    {/* Paper texture effect */}
                    <div 
                      className="paper-texture"
                      style={{
                        position: 'absolute',
                        inset: 0,
                        opacity: 0.02,
                        backgroundImage: `
                          repeating-linear-gradient(
                            0deg,
                            transparent,
                            transparent 2px,
                            rgba(0,0,0,0.03) 2px,
                            rgba(0,0,0,0.03) 4px
                          )
                        `,
                        pointerEvents: 'none'
                      }}
                    />
                    
                    {/* One paragraph per page */}
                    {page && page.length > 0 && (
                      <div
                        className="page-content"
                        style={{
                          ...pageStyle,
                          overflow: 'hidden',
                          zIndex: 2,
                          position: 'relative',
                          padding: `${PAGE_PADDING_TOP + 20}px ${PAGE_PADDING_SIDE}px ${PAGE_PADDING_TOP}px ${PAGE_PADDING_SIDE}px`,
                          maxWidth: '100%',
                          textAlign: 'left',
                          hyphens: 'auto',
                          WebkitHyphens: 'auto',
                          MozHyphens: 'auto',
                          msHyphens: 'auto',
                          overflowWrap: 'break-word',
                          wordBreak: 'break-word',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                        lang="en"
                      >
                        <div style={{
                          width: `${paragraphWidthPercent}%`,
                          maxWidth: '800px',
                          textAlign: 'left',
                          wordSpacing: 'normal',
                          textIndent: '2ch',
                          textWrap: 'pretty',
                          hyphens: 'auto',
                          WebkitHyphens: 'auto',
                          MozHyphens: 'auto',
                          msHyphens: 'auto',
                          wordBreak: 'normal',
                          overflowWrap: 'anywhere',
                          letterSpacing: `${settings.wordSpacing}px`
                        }}>
                          {page.map((clause, clauseIndex) => {
                            const isActive = currentTime >= clause.start && currentTime <= clause.end;
                            return (
                              <span
                                key={clauseIndex}
                                ref={isActive ? lastWordRef : undefined}
                                className={`reading-segment ${isActive ? 'active-segment' : ''} ${currentTime < clause.start ? 'future-segment' : ''}`}
                                style={{
                                  backgroundColor: isActive ? 'rgba(255, 235, 59, 0.3)' : 'transparent',
                                  padding: isActive ? '0 0' : '0',
                                  borderRadius: isActive ? '2px' : '0',
                                  transition: 'all 0.3s ease',
                                  opacity: currentTime < clause.start ? 0.5 : 1,
                                  whiteSpace: 'normal'
                                }}
                              >
                                {clause.text}{' '}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* Page number in the corner to avoid covering text */}
                    <div
                      className="page-number"
                      style={{
                        position: 'absolute',
                        bottom: '18px',
                        [isLeftPage ? 'left' : 'right']: '18px',
                        fontSize: '13px',
                        color: '#6b7280',
                        fontFamily: 'Georgia, serif',
                        fontStyle: 'italic',
                        textAlign: isLeftPage ? 'left' : 'right',
                        minWidth: '24px',
                        fontWeight: '500',
                        zIndex: 3,
                        pointerEvents: 'none',
                        opacity: 0.9
                      }}
                    >
                      {index + 1}
                    </div>
                  </div>
                );
              })}
            </HTMLFlipBook>
          </div>
        </div>
      </div>
    </div>
  );
});

export default FlipBook;