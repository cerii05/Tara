// src/pages/login.js
import React, {useState,} from 'react';
import { useNavigate, Link } from 'react-router-dom';
import logo from '../assets/logotara.png';
import './login.css';

const Login = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    
    try {
      const response = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, pass: password })
      });

      const data = await response.json();

      if (response.ok) {
        navigate('/home');
      } else {
        setError(data.message || 'Login gagal');
      }
    } catch (err) {
      setError('Gagal terhubung ke server');
    }
  };

  return (
    <div className="login-container">
      <img src={logo} alt="App Logo" className="login-logo" />

      <label htmlFor="email">Email / No. Telepon</label>
      <input
        type="text"
        id="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />

      <label htmlFor="password">Kata Sandi</label>
      <input 
        type="password"
        id="password" 
        value={password} 
        onChange={(e) => setPassword(e.target.value)}
      />

      {error && <p className="error-message">{error}</p>}

      <button className="login-button" onClick={handleLogin}>Masuk</button>

      <div className="signup-link-container">
        <span>Belum punya akun? </span>
        <Link to="/signup">Buat akun di sini</Link>
      </div>
    </div>
  );
};

export default Login;