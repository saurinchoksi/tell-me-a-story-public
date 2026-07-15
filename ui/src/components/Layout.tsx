import { NavLink, Outlet } from 'react-router-dom';
import './Layout.css';

export default function Layout() {
  return (
    <div className="layout">
      <nav className="layout-nav">
        <span className="layout-title">Tell Me a Story</span>
        <div className="layout-links">
          <NavLink to="/sessions">Sessions</NavLink>
          <NavLink to="/profiles">Profiles</NavLink>
          <NavLink to="/monitor">Monitor</NavLink>
          <NavLink to="/name-review">Names</NavLink>
          <NavLink to="/roster">Roster</NavLink>
        </div>
      </nav>
      <main className="layout-main">
        <Outlet />
      </main>
    </div>
  );
}
