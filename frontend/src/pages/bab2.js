import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './bab2.css';

const Bab2 = () => {
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

      <main className="bab2-main">
        <h2 className="bab-title">BAB 2: Mari Berkenalan</h2>

        <section className="dictionary-content">
          <p>
            Dalam bab ini, kita akan belajar kata-kata dan frasa dasar yang digunakan dalam perkenalan diri menggunakan Bisindo.
          </p>

          <ul>
            <li>
              <strong>Nama saya</strong>: Cara untuk memperkenalkan nama diri Anda.
            </li>
            <li>
              <strong>Siapa namamu?</strong>: Pertanyaan untuk menanyakan nama seseorang.
            </li>
            <li>
              <strong>Dari mana asalmu?</strong>: Digunakan untuk bertanya tentang tempat asal.
            </li>
            <li>
              <strong>Senang bertemu denganmu</strong>: Ungkapan yang digunakan saat bertemu orang baru.
            </li>
            <li>
              <strong>Berapa umurmu?</strong>: Pertanyaan tentang usia.
            </li>
          </ul>

          <p>
            Menguasai frasa-frasa ini akan membantu Anda memulai percakapan dalam situasi sosial dan berkenalan dengan orang baru menggunakan bahasa isyarat Bisindo.
          </p>
        </section>
      </main>
    </>
  );
};

export default Bab2;
