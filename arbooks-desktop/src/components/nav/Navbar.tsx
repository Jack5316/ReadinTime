import React, { FC } from 'react'
import { NavLink } from "react-router"

const Navbar: FC = () => {
  return (
    <nav className="w-full navbar bg-base-100">
      <div className="flex-1">
        <NavLink to="/library" className="btn btn-ghost text-2xl">
          ReadInTime
        </NavLink>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1">
          <li>
            <NavLink to="/library">📚 Library</NavLink>
          </li>
          <li>
            <NavLink to="/settings">⚙️ Settings</NavLink>
          </li>
        </ul>
      </div>
    </nav>
  )
}

export default Navbar
