// src/pages/mainPage.js
import React from 'react';
import { Link } from 'react-router-dom';
import logo from '../assets/logotara.png';
import './mainPage.css';

const MainPage = () => {
  return (
    <div className="app-header">
      <img src={logo} alt="App Logo" className="app-logo" />
      
      <Link to="/login">
        <button className="login-button">Masuk</button>
      </Link>
      
      <Link to="/signup">
        <button className="signup-button">Buat Akun</button>
      </Link>
    </div>
  );
};

export default MainPage;

