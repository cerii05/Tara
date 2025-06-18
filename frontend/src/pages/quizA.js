import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import quizImg from '../assets/game.png';
import './quizA.css';

const QuizA = () => {
  const navigate = useNavigate();
  const [selected, setSelected] = useState(null);
  const [checked, setChecked] = useState(false);

  const handleOptionClick = (option) => {
    if (!checked) setSelected(option);
  };

  const handleCheck = () => {
    if (selected !== null) {
      setChecked(true);
    }
  };

  const handleNext = () => {
    console.log('Navigating to scorePage');
    navigate('/scorePage');
  };

  return (
    <div>
      <header className="header">
        <img src={logo} alt="Logo" className="logo" />
        <img src={profile} alt="Profile" className="profile-icon" />
      </header>

      <button className="back-button" onClick={() => navigate(-1)}>
        &#60; Kembali
      </button>

      <main className="quiz-main">
        <img src={quizImg} alt="Quiz A" className="quiz-image" />

        <div className="option-container-vertical">
          {['O', 'G', 'P', 'Q'].map((option) => {
            const isSelected = selected === option;
            const isCorrect = option === 'G';
            const showCheck = checked && isSelected && isCorrect;
            const showCross = checked && isSelected && !isCorrect;
            const bgColor = checked && option === 'G' ? '#C1D9BF' : '#FFD9E8';

            return (
              <div
                key={option}
                onClick={() => handleOptionClick(option)}
                className="option-box"
                style={{
                  backgroundColor: bgColor,
                  border: isSelected ? '2px solid #000' : '1px solid #ccc',
                  position: 'relative',
                  cursor: checked ? 'default' : 'pointer',
                }}
              >
                {option}
                {showCheck && (
                  <span className="tick-mark" style={{ position: 'absolute', right: 10, top: 5 }}>
                    ✓
                  </span>
                )}
                {showCross && (
                  <span className="cross-mark" style={{ position: 'absolute', right: 10, top: 5 }}>
                    ✗
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {!checked ? (
          <button className="check-button" onClick={handleCheck} style={{ backgroundColor: '#D89999' }}>
            Periksa
          </button>
        ) : (
          <button className="next-button" onClick={handleNext} style={{ backgroundColor: '#628395' }}>
            Berikutnya
          </button>
        )}
      </main>
    </div>
  );
};

export default QuizA;

