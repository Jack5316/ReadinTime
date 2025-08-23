import { useCallback, useState } from "react"
import { BookInfo } from '../types/book';

const useListBooks = () => {
  const [books, setBooks] = useState<BookInfo[]>([]);

  const listBooks = useCallback(async (bookPath: string) => {
    try {
      const listBooksResult = await window.electron.listBooks(bookPath);

      if (!listBooksResult.success) throw listBooksResult.error;

      const isAbsolutePath = (p: string) => /^(?:[a-zA-Z]:\\|\/)/.test(p) || p.startsWith('\\\\');
      const joinPath = (a: string, b: string, c?: string) => [a, b, c].filter(Boolean).join('/');

      const booksLoaded = await Promise.all(listBooksResult.result.map(async book => {
        try {
          // Resolve a usable cover path if possible
          const tryLoadCover = async (path: string) => {
            try {
              const imageData = await window.electron.getFileData(path);
              if (imageData.success) {
                // naive content-type detection by extension
                const lower = path.toLowerCase();
                const mime = lower.endsWith('.jpg') || lower.endsWith('.jpeg') ? 'image/jpeg' : 'image/png';
                const blob = new Blob([imageData.result], { type: mime });
                return URL.createObjectURL(blob);
              }
            } catch {}
            return '';
          };

          const resolveCoverPath = (p: string) => {
            if (!p) return '';
            if (isAbsolutePath(p)) return p;
            // treat relative paths as inside the book folder
            return joinPath(bookPath, book.folder, p);
          };

          // Attempt 1: explicit cover path if provided
          let coverPath = resolveCoverPath(book.cover || '');
          console.log(`Book "${book.title}": cover field="${book.cover}", resolved path="${coverPath}"`);
          let blobURL = coverPath ? await tryLoadCover(coverPath) : '';
          
          // Attempt 2: common filenames inside folder (including extracted covers)
          if (!blobURL) {
            console.log(`Book "${book.title}": trying common cover filenames...`);
            const candidates = ['cover.png', 'cover.jpg', 'cover.jpeg', 'thumbnail.png', 'thumbnail.jpg'];
            for (const name of candidates) {
              const p = joinPath(bookPath, book.folder, name);
              console.log(`Book "${book.title}": trying ${p}`);
              blobURL = await tryLoadCover(p);
              if (blobURL) {
                console.log(`Book "${book.title}": found cover at ${p}`);
                break;
              }
            }
          }

          if (!blobURL) {
            console.log(`Book "${book.title}" has no cover image, using placeholder`);
            return { ...book, cover: '' };
          }
          
          console.log(`Book "${book.title}" loaded cover: ${blobURL}`);
          return { ...book, cover: blobURL };
        } catch (error) {
          console.error(`Error processing book "${book.title}":`, error);
          return { ...book, cover: '' }; // Always return the book, even if cover processing fails
        }
      }))
      setBooks(booksLoaded);
    } catch (error) {
      console.error(error);
    }
  }, [])

  return { books, listBooks };
}

export default useListBooks;
