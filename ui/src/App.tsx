import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Sessions from './pages/Sessions';
import SessionSpeakers from './pages/SessionSpeakers';
import ProfileGallery from './pages/ProfileGallery';
import ProfileDetail from './pages/ProfileDetail';
import ValidatorPage from './pages/ValidatorPage';
import NamePassPage from './pages/NamePassPage';
import SessionRead from './pages/SessionRead';
import Monitor from './pages/Monitor';
import NameReview from './pages/NameReview';
import RosterPage from './pages/Roster';
import SessionDetections from './pages/SessionDetections';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Validator + name-pass render outside Layout — they need full viewport width */}
        <Route path="/sessions/:id/validate" element={<ValidatorPage />} />
        <Route path="/sessions/:id/name-pass" element={<NamePassPage />} />

        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/sessions" replace />} />
          <Route path="/sessions" element={<Sessions />} />
          <Route path="/sessions/:id" element={<SessionRead />} />
          <Route path="/sessions/:id/speakers" element={<SessionSpeakers />} />
          <Route path="/sessions/:id/detections" element={<SessionDetections />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/name-review" element={<NameReview />} />
          <Route path="/roster" element={<RosterPage />} />
          <Route path="/profiles" element={<ProfileGallery />} />
          <Route path="/profiles/:id" element={<ProfileDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
