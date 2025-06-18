import React, { createContext, useState } from 'react';

export const ForumContext = createContext();

export const ForumProvider = ({ children }) => {
  const [posts, setPosts] = useState([
    {
      username: "username",
      image: require('../assets/forum.png'),
      caption: "Abis coba main ke Starbuck Tata Puri dan ketemu sama temen-temen tuli di sana! Tempatnya nyaman banget dan sangat recommend untuk wfc",
    }
  ]);

  const addPost = (caption, image = null) => {
    setPosts(prev => [
      ...prev,
      {
        username: "username",
        image,
        caption,
      }
    ]);
  };

  return (
    <ForumContext.Provider value={{ posts, addPost }}>
      {children}
    </ForumContext.Provider>
  );
};
