import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './learningPage.css';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import sampleVideo from '../assets/video.mp4';

const LearningPage = () => {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const [showRewind, setShowRewind] = useState(false);

  const handleVideoEnd = () => {
    setShowRewind(true);
  };

  const handleRewind = () => {
    videoRef.current.currentTime = 0;
    videoRef.current.play();
    setShowRewind(false);
  };

  return (
    <div className="learning-page">
      <header className="learning-header">
        <img src={logo} alt="Logo" className="learning-logo" />
        <img src={profile} alt="Profile" className="learning-profile" />
      </header>

      <button className="back-button" onClick={() => navigate('/skytrain')}>
        &lt; Kembali
      </button>

      <div className="video-container">
        <video
          ref={videoRef}
          src={sampleVideo}
          controls
          onEnded={handleVideoEnd}
          className="learning-video"
        />
        {showRewind && (
          <button className="rewind-button" onClick={handleRewind}>
            ⟲ Ulangi
          </button>
        )}
      </div>

      <div className="bottom-buttons">
        <button
          className="btn-continue"
          onClick={() => alert('Lanjutkan diklik')}
        >
          Lanjutkan
        </button>
        <button
          className="btn-home"
          onClick={() => navigate('/home')}
        >
          Halaman Utama
        </button>
      </div>
    </div>
  );
};

export default LearningPage;
