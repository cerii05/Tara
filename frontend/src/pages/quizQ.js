import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import quizImg from '../assets/game.png'; // Replace with your image path
import './quizQ.css';

const QuizQ = () => {
  const navigate = useNavigate();

  const [selectedOption, setSelectedOption] = useState(null);
  const [checked, setChecked] = useState(false);

  const correctAnswer = 'G';

  const handleCheck = () => {
    if (selectedOption === null) return; // no option selected, ignore
    setChecked(true); // show results and change button text
  };

  const handleNext = () => {
    // For example: go to next quiz or another page
    // Here just going back to home for demo
    navigate('/scorePage');
  };

  return (
    <div>
      {/* Header */}
      <header className="header">
        <img src={logo} alt="Logo" className="logo" />
        <img src={profile} alt="Profile" className="profile-icon" />
      </header>

      {/* Back Button */}
      <button className="back-button" onClick={() => navigate(-1)}>
        &#60; Kembali
      </button>

      <main className="quiz-main">
        {/* Quiz Image */}
        <img src={quizImg} alt="Quiz" className="quiz-image" />

        {/* Options Vertical Container */}
        <div className="option-container" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {['O', 'G', 'P', 'Q'].map((opt) => (
            <div
              key={opt}
              onClick={() => !checked && setSelectedOption(opt)}
              style={{
                padding: '10px',
                border: '1px solid black',
                cursor: checked ? 'default' : 'pointer',
                backgroundColor:
                  checked
                    ? opt === correctAnswer
                      ? '#C1D9BF' // Correct answer color
                      : opt === selectedOption
                      ? '#f9c9c9' // Wrong selected answer color
                      : '#fff'
                    : selectedOption === opt
                    ? '#ddd'
                    : '#fff',
                position: 'relative',
                userSelect: 'none',
              }}
            >
              {opt}

              {/* Cross or Tick icons */}
              {checked && opt === selectedOption && opt !== correctAnswer && (
                <span style={{ color: 'red', position: 'absolute', right: '10px' }}>✗</span>
              )}
              {checked && opt === correctAnswer && (
                <span style={{ color: 'green', position: 'absolute', right: '10px' }}>✔</span>
              )}
            </div>
          ))}
        </div>

        {/* Check / Next Button */}
        <button
          onClick={checked ? handleNext : handleCheck}
          disabled={!selectedOption}
          style={{
            marginTop: '20px',
            backgroundColor: '#D89999',
            padding: '10px 20px',
            border: 'none',
            color: 'white',
            cursor: selectedOption ? 'pointer' : 'not-allowed',
            borderRadius: '4px',
            fontWeight: 'bold',
            fontSize: '16px',
          }}
        >
          {checked ? 'Berikutnya' : 'Periksa'}
        </button>
      </main>
    </div>
  );
};

export default QuizQ;


