import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Route, Routes } from "react-router-dom";
import './index.css'
import App from './App.tsx'
import Book from './pages/Book.tsx'
import Library from './pages/Library.tsx'
import Settings from './pages/Settings.tsx'
import SettingsSimple from './pages/SettingsSimple.tsx'
import Home from './pages/Home.tsx';
import ErrorBoundary from './components/ErrorBoundary.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/library' element={<App><Library /></App>} />
        <Route path='/settings' element={
          <ErrorBoundary>
            <App><Settings /></App>
          </ErrorBoundary>
        } />
        <Route path='/book/:title' element={<App><Book /></App>} />
      </Routes>
    </HashRouter>
  </StrictMode>,
)
