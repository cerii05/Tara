import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './gamePage.css';

const GamePage = () => {
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

      <main className="game-main">
        <h1 className="game-title">Game 1</h1>
        <button className="start-button" onClick={() => navigate('/quizQ')}>
          Mulai
        </button>
      </main>
    </div>
  );
};

export default GamePage;

