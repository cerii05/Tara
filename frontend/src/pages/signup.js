import React, {useState} from 'react';
import { useNavigate, Link } from 'react-router-dom';
import logo from '../assets/logotara.png';
import './signup.css';

const Signup = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [phoneNum, setPhoneNum] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  

  const handleSignup = async () => {
    // Validasi sederhana
    if (password !== confirmPassword) {
      setError('Kata sandi tidak cocok');
      return;
    }

    try {
      const res = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, phoneNum, pass: password }),
      });

      const data = await res.json();

      if (res.ok) {
        // Registrasi berhasil
        navigate('/login');
      } else {
        // Registrasi gagal
        setError(data.message || 'Gagal mendaftar');
      }
    } catch (err) {
      setError('Tidak dapat terhubung ke server');
    }
  };

  return (
    <div className="signup-container">
      <img src={logo} alt="App Logo" className="signup-logo" />

      <label htmlFor="email">Email</label>
      <input type="text" id="email" value={email} onChange={(e) => setEmail(e.target.value)} />

      <label htmlFor="phone">No. Telepon</label>
      <input type="text" id="phone" value={phoneNum} onChange={(e) => setPhoneNum(e.target.value)} />

      <label htmlFor="password">Kata Sandi</label>
      <input type="password" id="password" value={password} onChange={(e) => setPassword(e.target.value)} />

      <label htmlFor="confirm-password">Konfirmasi Kata Sandi</label>
      <input type="password" id="confirm-password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} />

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <button className="signup-button" onClick={handleSignup}>Buat Akun</button>

      <div className="login-link-container">
        <span>Sudah punya akun? </span>
        <Link to="/login">Masuk di sini</Link>
      </div>
    </div>
  );
};

export default Signup;




