import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './bab4.css'; // Reuse the same CSS used for bab1, bab2, bab3

const Bab4 = () => {
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

      <main className="bab-main">
        <h2 className="bab-title">Bab 4: Bepergian</h2>

        <div className="dictionary-entry">
          <h3>Transportasi</h3>
          <p>Gerakan dua tangan seperti mengemudi kemudi.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Mobil</h3>
          <p>Gerakan tangan seperti memegang setir kecil dengan ekspresi netral.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Bus</h3>
          <p>Dua tangan bergerak maju sejajar seperti menunjukkan panjang kendaraan besar.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Jalan</h3>
          <p>Dua jari menunjuk lurus ke depan seperti jalur lurus.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Bandara</h3>
          <p>Tangan bergerak ke atas seperti pesawat lepas landas.</p>
        </div>
      </main>
    </>
  );
};

export default Bab4;
