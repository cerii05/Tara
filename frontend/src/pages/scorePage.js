import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './scorePage.css';

const ScorePage = () => {
  const navigate = useNavigate();

  return (
    <div>
      <header className="header">
        <img src={logo} alt="Logo" className="logo" />
        <img src={profile} alt="Profile" className="profile-icon" />
      </header>

      <button className="back-button" onClick={() => navigate(-1)}>
        &#60; Kembali
      </button>

      <main>
        <h2>Skor Akhir</h2>

        <div className="score-circle">100</div>

        <p className="score-text">Total jawaban benar</p>
        <p className="score-bold-text">10 dari 10 pertanyaan!</p>

        <div className="button-container">
          <button
            className="btn-lanjutkan"
            onClick={() => navigate('/gamePage')}
          >
            Lanjutkan
          </button>
          <button
            className="btn-halaman-utama"
            onClick={() => navigate('/home')}
          >
            Halaman Utama
          </button>
        </div>
      </main>
    </div>
  );
};

export default ScorePage;

