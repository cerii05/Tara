# TARA Project - Frontend & Backend Setup Guide

Hi! 👋  
Ini adalah project React + Express (Node.js) untuk aplikasi TARA.  
Ikuti langkah-langkah di bawah ini untuk menjalankan project-nya secara lokal:

---

📁 Struktur Project:
- frontend/    --> React (User Interface)
- backend/     --> Node.js Express (Backend API)

---

## ✅ PERSIAPAN AWAL

1. **Ekstrak file ZIP** ke folder manapun.
2. Pastikan sudah terinstall:
   - Node.js (https://nodejs.org/)
   - npm (biasanya sudah termasuk dengan Node.js)

---

***Semua kegiatan ini dijalankan di vscode***
a. buka vscode, lalu open project folder
b. then selanjutnya sesuai dibawah ini


## 🚀 LANGKAH MENJALANKAN BACKEND

1. Buka terminal baru

*Selanjutnya ketik diterminal*

2. Untuk masuk ke folder backend:

   cd backend

3. Install dependencies:

   npm install

4. Jalankan server:

   npm run dev

   > Jika `npm run dev` tidak tersedia, bisa gunakan:
   node server.js

5. Server akan jalan di:

   http://localhost:5000

---

## 🎨 LANGKAH MENJALANKAN FRONTEND

1. Buka terminal baru

*Selanjutnya ketik diterminal*
2. Masuk ke folder frontend:

   cd frontend

3. Install dependencies:

   npm install

4. Jalankan React app:

   npm start

5. Aplikasi akan terbuka otomatis di browser:

   http://localhost:3000

---

## ⚠️ CATATAN

- Pastikan **backend sudah jalan duluan** sebelum melakukan login/register di frontend.
- Jangan menutup terminal backend selama testing.
- Semua data user disimpan **sementara** (tanpa database) — akan hilang jika backend di-restart.

---

Happy coding! 🎉  
Jika ada error atau kendala, bisa hubungi pemilik project ini.