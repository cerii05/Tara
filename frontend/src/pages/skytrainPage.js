import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './skytrainPage.css';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import trainImg from '../assets/train.png';
import flagImg from '../assets/flag.png';
import bgImg from '../assets/path.png';

const flags = [
  { id: 1, label: 'Bab 1', top: 100, unlocked: true },
  { id: 2, label: 'Bab 2', top: 300, unlocked: false },
  { id: 3, label: 'Bab 3', top: 500, unlocked: false },
];

const SkytrainPage = () => {
  const navigate = useNavigate();
  const trainRef = useRef(null);
  const [trainY, setTrainY] = useState(90);
  const [isDragging, setIsDragging] = useState(false);
  const offsetRef = useRef(0);

  const handleMouseDown = (e) => {
    setIsDragging(true);
    const trainRect = trainRef.current.getBoundingClientRect();
    offsetRef.current = e.clientY - trainRect.top;
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const newY = e.clientY - offsetRef.current;
    setTrainY(newY);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleTrainClick = () => {
    // Cek apakah kereta berada cukup dekat dengan flag yang terbuka
    const matchedFlag = flags.find(
      (flag) =>
        flag.unlocked &&
        Math.abs(trainY - flag.top) < 30 // toleransi geser 30px
    );

    if (matchedFlag) {
      navigate(`/learningPage/${matchedFlag.id}`);
    } else {
      alert('Geser kereta ke bab yang sudah terbuka untuk memulai belajar.');
    }
  };

  return (
    <div
      className="skytrain-body"
      style={{ backgroundImage: `url(${bgImg})` }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <header className="skytrain-header">
        <button className="back-button" onClick={() => navigate('/home')}>
          &#8592; Kembali
        </button>
        <img src={logo} alt="Logo" className="logo-img" />
        <img src={profile} alt="Profile" className="profile-img" />
      </header>

      <main className="skytrain-main">
        <div className="skytrain-line" />
        {flags.map((flag) => (
          <div
            key={flag.id}
            className="flag-wrapper"
            style={{ top: flag.top, left: 'calc(50% + 80px)' }}
          >
            <img
              src={flagImg}
              alt={flag.label}
              style={{ opacity: flag.unlocked ? 1 : 0.3 }}
              onClick={() => navigate('/learningPage')}
            />
            {/* <div className="flag-label">{flag.label}</div> */}
          </div>
        ))}

        <img
          src={trainImg}
          ref={trainRef}
          alt="Train"
          className="train-img"
          onMouseDown={handleMouseDown}
          onClick={handleTrainClick}
          style={{ top: trainY }}
        />
      </main>
    </div>
  );
};

export default SkytrainPage;
