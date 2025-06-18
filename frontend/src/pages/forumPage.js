import React, { useContext, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './forumPage.css';
import logo from '../assets/logotara.png';
import profileImg from '../assets/profile.png';
import { ForumContext } from '../context/forumContext';

const Forum = () => {
  const navigate = useNavigate();
  const { posts } = useContext(ForumContext);

  return (
    <div>
      <header className="forum-header">
        <img src={logo} alt="Logo" className="home-logo" />
        <img src={profileImg} alt="Profile" className="profile-icon-img" />
      </header>

      <button className="back-button" onClick={() => window.history.back()}>
        &#60; Kembali
      </button>

      {/* <main className="forum-main">
        {posts.map((post, index) => (
          <div className="forum-post-container" key={index}>
            <div className="profile-row">
              <div className="black-circle"></div>
              <span className="username-text">{post.username}</span>
            </div>

            {post.image && (
              <img src={post.image} alt="Forum Post" className="forum-image" />
            )}

            <p className="post-text">{post.caption}</p>
          </div>
        ))}

        <button className="add-button" onClick={() => navigate('/addForum')}>+</button>
      </main> */}

      <main className="forum-main">
        {posts && posts.length > 0 ? (
          posts.map((post, index) => (
            <div key={index} className="forum-post-container">
              <div className="profile-row">
                <div className="black-circle"></div>
                <span className="username-text">{post.username}</span>
              </div>
              {post.image && (
              <img src={post.image} alt="Forum Post" className="forum-image" />
            )}

            <p className="post-text">{post.caption}</p>
            </div>
          ))
        ) : (
          <p style={{ textAlign: 'center', marginTop: '20px' }}>
            Belum ada postingan.
          </p>
        )}
        <button className="add-button" onClick={() => navigate('/addForum')}>+</button>
      </main>
    </div>
  );
};

export default Forum;
