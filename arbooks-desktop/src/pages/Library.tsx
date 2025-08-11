import React, { FC, useEffect, useRef, useMemo, useState } from 'react'
import { NavLink } from "react-router"
import { HiArrowPath, HiArrowUpTray } from "react-icons/hi2"
import ChangeBookPathButton from '../components/settings/ChangeBookPathButton';
import useStore from '../store/useStore';
import AddBookModal from '../components/book/AddBookModal';
import useListBooks from '../hooks/useListBooks';

const Library: FC = () => {
  const addBookModalRef = useRef<HTMLDialogElement>(null);
  const { books, listBooks } = useListBooks();
  const { settings } = useStore();
  const { bookPath } = settings;
  const gridRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [pageCapacity, setPageCapacity] = useState(12);

  useEffect(() => {
    if (!bookPath) return;

    listBooks(bookPath);
  }, [bookPath])

  // Compute how many cards fit in the viewport without vertical scrolling
  useEffect(() => {
    const recalc = () => {
      const grid = gridRef.current;
      if (!grid) return;
      const gridRect = grid.getBoundingClientRect();
      const gap = 16; // Tailwind gap-4 ~ 1rem
      const CARD_W = 256; // w-64
      const CARD_H = 420; // approx min-h-96 with padding/shadow
      const cols = Math.max(1, Math.floor((gridRect.width + gap) / (CARD_W + gap)));
      const availableHeight = Math.max(0, window.innerHeight - gridRect.top - 24);
      const rows = Math.max(1, Math.floor((availableHeight + gap) / (CARD_H + gap)));
      const capacity = cols * rows;
      setPageCapacity(capacity);
      setCurrentPage(0);
    };
    // slight delay to ensure layout is ready
    const id = setTimeout(recalc, 50);
    window.addEventListener('resize', recalc);
    return () => { clearTimeout(id); window.removeEventListener('resize', recalc); };
  }, []);

  const pagedBooks = useMemo(() => {
    if (pageCapacity <= 0) return books;
    const start = currentPage * pageCapacity;
    return books.slice(start, start + pageCapacity);
  }, [books, currentPage, pageCapacity]);

  const totalPages = Math.max(1, Math.ceil(books.length / Math.max(1, pageCapacity)));

  // Child-friendly pastel gradient palette for placeholders
  const CHILD_GRADIENTS = [
    'bg-gradient-to-b from-rose-200 to-rose-400',
    'bg-gradient-to-b from-orange-200 to-orange-400',
    'bg-gradient-to-b from-amber-200 to-amber-400',
    'bg-gradient-to-b from-lime-200 to-lime-400',
    'bg-gradient-to-b from-emerald-200 to-emerald-400',
    'bg-gradient-to-b from-teal-200 to-teal-400',
    'bg-gradient-to-b from-sky-200 to-sky-400',
    'bg-gradient-to-b from-indigo-200 to-indigo-400',
    'bg-gradient-to-b from-fuchsia-200 to-fuchsia-400',
    'bg-gradient-to-b from-pink-200 to-pink-400'
  ] as const;

  const getGradientFor = (seed: string) => {
    const h = seed.split('')
      .reduce((acc, ch) => ((acc * 31) + ch.charCodeAt(0)) >>> 0, 0);
    return CHILD_GRADIENTS[h % CHILD_GRADIENTS.length];
  };

  return (
    <>
      <AddBookModal ref={addBookModalRef} />

      { /* IF YOU HAVEN'T SELECTED A BOOK PATH */}
      {!bookPath && (
        <div role="alert" className="alert bg-secondary text-secondary-content">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="h-6 w-6 shrink-0 stroke-current">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <span>You have not selected a folder that contains the books.</span>
          <ChangeBookPathButton />
        </div>
      )}

      <div className='card w-full shadow-xl'>
        <div className="card-body">
          <div className="w-full flex justify-between items-start">
            <div>
              <h2 className="card-title">Your library</h2>
              <p className="text-sm">Showing {books.length} of {books.length} book{books.length !== 1 ? "s" : ""}</p>
            </div>
            <div className='space-x-3'>
              <button className="btn btn-primary" onClick={() => {
                if (bookPath) { listBooks(bookPath); }
              }}><HiArrowPath className='w-5 h-5' />Refresh</button>
              <button className="btn btn-primary" onClick={() => addBookModalRef.current?.showModal()}><HiArrowUpTray className='w-5 h-5' />Add Book</button>
            </div>
          </div>

          {/* Grid with pagination to avoid page scrolling */}
          <div ref={gridRef} className='grid gap-4 mt-4' style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(16rem, 1fr))' }}>

            {pagedBooks.map((book, index) => (
              <NavLink key={book.folder} to={`/book/${book.folder}`}>
                <div ref={index === 0 ? cardRef : undefined} className="card w-64 min-h-96 shadow-xl hover:scale-105 transition-all">
                  <figure className="relative">
                    {book.cover ? (
                      <img
                        src={book.cover}
                        alt={book.title}
                        className="w-full h-48 object-cover"
                        onError={(e) => {
                          // Fallback to a simple colored placeholder if image fails to load
                          const img = e.target as HTMLImageElement;
                          img.style.display = 'none';
                          const placeholder = document.createElement('div');
                          const gradient = getGradientFor(book.title + '|' + (book.author || '')); 
                          placeholder.className = `w-full h-48 ${gradient} flex items-center justify-center text-white text-2xl font-bold`;
                          placeholder.textContent = book.title.charAt(0).toUpperCase();
                          img.parentElement?.appendChild(placeholder);
                        }}
                      />
                    ) : (
                      <div className={`w-full h-48 ${getGradientFor(book.title + '|' + (book.author || ''))} flex items-center justify-center text-white text-2xl font-bold`}>
                        {book.title.charAt(0).toUpperCase()}
                      </div>
                    )}
                  </figure>
                   <div className="card-body p-3">
                    <p className="text-sm">{book.author}</p>
                    <h3 className="card-title text-sm">{book.title}</h3>
                    <p className="text-sm">{book.description}</p>
                  </div>
                </div>
              </NavLink>
            ))}

          </div>

          {/* Pagination controls */}
          {totalPages > 1 && (
            <div className='flex justify-center items-center gap-2 mt-4'>
              <button className='btn btn-sm' disabled={currentPage === 0} onClick={() => setCurrentPage(p => Math.max(0, p - 1))}>Prev</button>
              <span className='text-sm'>Page {currentPage + 1} / {totalPages}</span>
              <button className='btn btn-sm' disabled={currentPage >= totalPages - 1} onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))}>Next</button>
            </div>
          )}

          {books.length === 0 && (
            <div role="alert" className="alert bg-secondary text-secondary-content">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="h-6 w-6 shrink-0 stroke-current">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span>You have not added any books to your library.</span>
            </div>
          )
          }
        </div>
      </div>
    </>
  )
}

export default Library
