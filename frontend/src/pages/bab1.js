// src/pages/Bab1.js
import React from 'react';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './bab1.css';

const Bab1 = () => {
  return (
    <>
      <header className="page-header">
        <img src={logo} alt="Logo" className="page-logo" />
        <img src={profile} alt="Profile" className="profile-icon" />
      </header>

      <button className="back-button" onClick={() => window.history.back()}>
        &#60; Kembali
      </button>

      <h2 className="page-title">BAB 1: Alfabet</h2>

      <main className="dictionary-main">
        <section className="dictionary-content">
          <p>
            Alfabet BISINDO adalah sistem huruf yang digunakan untuk menuliskan Bahasa Isyarat Indonesia (BISINDO).
            Alfabet ini terdiri dari gerakan tangan yang mewakili setiap huruf dalam alfabet Latin, yang memudahkan komunikasi bagi komunitas tuli dan mereka yang belajar BISINDO.
          </p>
          <ul>
            <li><strong>A</strong> – Bentuk tangan mengepal dengan ibu jari mengarah ke luar.</li>
            <li><strong>B</strong> – Tangan terbuka dengan jari-jari rapat dan telapak tangan menghadap ke depan.</li>
            <li><strong>C</strong> – Bentuk tangan seperti huruf "C".</li>
            <li><strong>D</strong> – Jari telunjuk tegak ke atas, ibu jari menyentuh jari tengah dan jari manis.</li>
            <li><strong>E</strong> – Jari-jari ditekuk membentuk huruf "E".</li>
            <li><strong>F</strong> – Jempol dan telunjuk membentuk lingkaran, jari lainnya tegak.</li>
            {/* Tambahkan alfabet lain jika ingin */}
          </ul>
          <p>
            Belajar alfabet BISINDO sangat membantu dalam memahami dasar-dasar komunikasi isyarat, yang merupakan pondasi penting untuk berbicara dengan komunitas tuli di Indonesia.
          </p>
        </section>
      </main>
    </>
  );
};

export default Bab1;
