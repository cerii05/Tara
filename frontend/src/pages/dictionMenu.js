import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './dictionMenu.css';

const DictionMenu = () => {
  const navigate = useNavigate();

  return (
    <>
      <header className="header">
        <img src={logo} alt="Logo" className="logo" />
        <img src={profile} alt="Profile" className="profile-icon" />
      </header>

      <button className="back-button" onClick={() => navigate(-1)}>
        &#60; Kembali
      </button>

      <main className="main-menu">
        <div className="menu-wrapper">
          <div
            className="menu-container"
            onClick={() => navigate('/bab1')}
            style={{ cursor: 'pointer' }}
          >
            Bab 1: Alfabet
          </div>

          <div
            className="menu-container"
            onClick={() => navigate('/bab2')}
            style={{ cursor: 'pointer' }}
          >
            Bab 2: Mari Berkenalan
          </div>

          <div className="menu-container"
          onClick={() => navigate('/bab3')}
          style={{ cursor: 'pointer' }}
          >
            Bab 3: Berbelanja
            </div>
          <div className="menu-container"
          onClick={() => navigate('/bab4')}
          style={{ cursor: 'pointer' }}
          >Bab 4: Bepergian</div>
        </div>
      </main>
    </>
  );
};

export default DictionMenu;


