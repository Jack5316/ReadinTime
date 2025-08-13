import React, { FC, ReactNode } from 'react'
import Navbar from './components/nav/Navbar'
import useStore from './store/useStore'
import { useLocation } from 'react-router-dom'

interface AppProps {
  children?: ReactNode
}

const App: FC<AppProps> = ({ children }) => {
  const { settings } = useStore();
  const location = useLocation();

  return (
    <div
      data-theme={settings.isFullScreen ? "paper" : settings.fontColour}
      style={{
        fontFamily: `'Source Serif 4', serif`
      }}
    >
      {!location.pathname.startsWith("/book") && <Navbar />}

      <div className={`${settings.isFullScreen ? 'px-0' : 'px-4'} w-full h-screen`}>
        {children}
      </div>
    </div>
  )
}

export default App
