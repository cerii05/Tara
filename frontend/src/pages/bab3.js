import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './bab3.css'; // You can reuse the same CSS as bab1/bab2 if it's consistent

const Bab3 = () => {
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
        <h2 className="bab-title">Bab 3: Berbelanja</h2>

        <div className="dictionary-entry">
          <h3>Mahal</h3>
          <p>Digambarkan dengan gerakan tangan mengangkat ke atas dengan ekspresi terkejut.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Murah</h3>
          <p>Gerakan tangan ke bawah sambil menggelengkan kepala.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Uang</h3>
          <p>Gerakan seperti menggenggam dan membuka telapak tangan secara bergantian.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Diskon</h3>
          <p>Menunjukkan angka kecil dengan jari dan ekspresi senang.</p>
        </div>

        <div className="dictionary-entry">
          <h3>Tawar</h3>
          <p>Gerakan tangan seperti memberi dan menarik kembali, diiringi anggukan.</p>
        </div>
      </main>
    </>
  );
};

export default Bab3;
