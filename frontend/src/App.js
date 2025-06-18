import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import MainPage from './pages/mainPage';
import HomePage from './pages/home';
import Login from './pages/login';
import Signup from './pages/signup';
import AI from './pages/AI';
import DictionMenu from './pages/dictionMenu';
import LearningPage from './pages/learningPage';
import Bab1 from './pages/bab1';
import Bab2 from './pages/bab2';
import Bab3 from './pages/bab3';
import Bab4 from './pages/bab4';
import GamePage from './pages/gamePage';
import QuizQ from './pages/quizQ';
import QuizA from './pages/quizA';
import Forum from './pages/forumPage';
import AddForum from './pages/addForum';
import ScorePage from './pages/scorePage';
import SkytrainPage from './pages/skytrainPage';

import { ForumProvider } from './context/forumContext'; // ✅ Tambahkan ini

const App = () => {
  return (
    <Router>
      <ForumProvider> {/* ✅ Bungkus semua route dengan context provider */}
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/learningPage" element={<LearningPage />} />
          <Route path="/AI" element={<AI />} />
          <Route path="/dictionMenu" element={<DictionMenu />} />
          <Route path="/bab1" element={<Bab1 />} />
          <Route path="/bab2" element={<Bab2 />} />
          <Route path="/bab3" element={<Bab3 />} />
          <Route path="/bab4" element={<Bab4 />} />
          <Route path="/gamePage" element={<GamePage />} />
          <Route path="/quizQ" element={<QuizQ />} />
          <Route path="/quizA" element={<QuizA />} />
          <Route path="/forumPage" element={<Forum />} />
          <Route path="/addForum" element={<AddForum />} />
          <Route path="/scorePage" element={<ScorePage />} />
          <Route path="/skytrain" element={<SkytrainPage />} />
        </Routes>
      </ForumProvider>
    </Router>
  );
};

export default App;