// src/pages/AI.js
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './AI.css';
import logo from '../assets/logotara.png';
import profileImg from '../assets/profile.png';
import historyIcon from '../assets/history.png';

const AIPage = () => {
  const navigate = useNavigate();

  return (
    <div>
      <header className="ai-header">
        <img src={logo} alt="Logo" className="ai-logo" />
        <img src={profileImg} alt="Profile" className="ai-profile" />
      </header>

      <main className="ai-main">
        <button className="back-button" onClick={() => navigate('/home')}>&lt; Kembali</button>
        <img src={historyIcon} alt="History" className="history-icon" />

        <div className="camera-container">
          <div className="camera-box">Kamera</div>
          <p className="camera-instruction">Pastikan tangan anda terlihat jelas pada kamera</p>
        </div>
      </main>
    </div>
  );
};

export default AIPage;
