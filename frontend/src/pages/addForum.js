import React, { useContext, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logotara.png';
import profile from '../assets/profile.png';
import './addForum.css'; // Make sure to create this CSS file with the styles

// const AddForum = () => {
//   const navigate = useNavigate();

//   return (
//     <>
//       <header className="add-header">
//         <img src={logo} alt="Logo" className="home-logo" />
//         <img src={profile} alt="Profile" className="profile-icon-img" />
//       </header>

//       {/* Back button below header */}
//       <button className="back-button" onClick={() => navigate(-1)}>
//         &#60; Kembali
//       </button>

//       <main className="addforum-main">
//         <div className="post-container">
//           <div className="profile-row">
//             <div className="black-circle"></div>
//             <span className="username-text">username</span>
//           </div>

//           <p className="post-prompt"><em>Mau cerita apa hari ini?</em></p>

//           <textarea
//             className="post-textarea"
//             placeholder="Tulis cerita kamu di sini..."
//           />
//         </div>

//         <button 
//           className="post-button" 
//           onClick={() => navigate('/forum')}
//         >
//           Post
//         </button>
//       </main>
//     </>
//   );
// };

// export default AddForum;


import { ForumContext } from '../context/forumContext';

const AddForum = () => {
  const navigate = useNavigate();
  const { addPost } = useContext(ForumContext);

  const [caption, setCaption] = useState('');
  const [image, setImage] = useState(null);

  const handlePost = () => {
    if (caption.trim() !== '') {
      addPost(caption, image);
    }
    navigate('/forumPage');
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file)); // convert ke blob URL
    }
  };

  return (
    <>
      <header className="add-header">
        <img src={logo} alt="Logo" className="home-logo" />
        <img src={profile} alt="Profile" className="profile-icon-img" />
      </header>

      <button className="back-button" onClick={() => navigate(-1)}>
        &#60; Kembali
      </button>

      <main className="addforum-main">
        <div className="post-container">
          <div className="profile-row">
            <div className="black-circle"></div>
            <span className="username-text">username</span>
          </div>

          <p className="post-prompt"><em>Mau cerita apa hari ini?</em></p>

          <textarea
            className="post-textarea"
            placeholder="Tulis cerita kamu di sini..."
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
          />

          <input 
            type="file" 
            accept="image/*" 
            onChange={handleImageChange}
            className="file-input"
          />

          {image && (
            <img 
              src={image} 
              alt="Preview" 
              className="image-preview"
              style={{ marginTop: '10px', maxWidth: '100%', borderRadius: '8px' }}
            />
          )}
        </div>

        <button 
          className="post-button" 
          onClick={handlePost}
        >
          Post
        </button>
      </main>
    </>
  );
};

export default AddForum;