import React from 'react';
import { Link } from 'react-router-dom';
import './home.css';
import logo from '../assets/logotara.png';
import profileImg from '../assets/profile.png';
import skytrainImg from '../assets/skytrain.png';
import forumImg from '../assets/komunitas.png';
import gameImg from '../assets/quiz.png';
import dictionaryImg from '../assets/kamus.png';
import aiImage from '../assets/ai.png';

const HomePage = () => {
  return (
    <div>
      <header className="home-header">
        <img src={logo} alt="Logo" className="home-logo" />
        <img src={profileImg} alt="Profile" className="profile-icon-img" />
      </header>

      <main className="home-main">
        <h1 className="welcome-text">Ayo Belajar!</h1>

        <Link to="/skytrain" style={{ textDecoration: 'none' }}>
          <div className="container skytrain-container">
            <img src={skytrainImg} alt="Skytrain" className="container-image" />
            <p className="container-label">Bab & Materi</p>
          </div>
        </Link>

        <Link to="/ai" style={{ textDecoration: 'none' }}>
          <div className="container ai-container">
            <img src={aiImage} alt="AI Auto Translation" className="container-image" />
            <p className="container-label">AI Auto Translation</p>
          </div>
        </Link>

        <div className="row-container">
          <Link to="/gamePage" style={{ textDecoration: 'none', flex: 1 }}>
            <div className="image-container">
              <img src={gameImg} alt="Game" className="container-image" />
              <p className="container-label">Permainan</p>
            </div>
          </Link>

          <Link to="/dictionMenu" style={{ textDecoration: 'none', flex: 1 }}>
            <div className="image-container">
              <img src={dictionaryImg} alt="Dictionary" className="container-image" />
              <p className="container-label">Kamus</p>
            </div>
          </Link>
        </div>

        <Link to="/forumPage" style={{ textDecoration: 'none' }}>
          <div className="image-container forum-container">
            <img src={forumImg} alt="Forum" className="container-image" />
            <p className="container-label">Komunitas</p>
          </div>
        </Link>
      </main>
    </div>
  );
};

export default HomePage;








