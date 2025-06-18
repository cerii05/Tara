const express = require('express')
const app = express()
app.use(express.json())
const cors = require('cors');
app.use(cors());

// ### for user data ###

const users = []
const usersProgres = []

users.push({
    email: "sarah@gmail.com",
    phoneNum: "081912345678",
    pass: "sarahsarah"
});
usersProgres.push({
    email: "sarah@gmail.com",
    progress: ["Bab 1", "Bab 2"]
});

app.get("/api/users", (req, res) => {
    res.json({ user: users });
});
app.get("/api/user-progress", (req, res) => {
    res.json({ userProgress: usersProgres });
});

app.post("/register", (req, res) => {
    const{email, phoneNum, pass} = req.body;

    users.push({email, phoneNum, pass});
    usersProgres.push({ email, progress: [] });
    
    res.status(201).json({message:"Pendaftaran akun berhasil!"});
});

app.post("/login", (req, res) => {
    const{email, pass} = req.body

    const user = users.find(u => u.email === email && u.pass === pass);
    if(user){
        res.json({message: "Berhasil login!"});
    }else{
        res.status(401).json({message: "Email atau password salah"});
    }
})

// ### end user data ###

// ### User Learning Progress ###

/*Update user progress*/
app.post('/progress', (req, res) => {
  const { email, bab } = req.body;

  let user = userProgress.find(u => u.email === email);

  if (!user) {
    user = { email, progress: [bab] };
    userProgress.push(user);
  } else {
    if (!user.progress.includes(bab)) {
      user.progress.push(bab);
    }
  }

  res.json({ message: 'Progress berhasil diupdate', progress: user.progress });
});

/* To take user progress */
app.get('/progress/:email', (req, res) => {
  const email = req.params.email;
  const user = userProgress.find(u => u.email === email);

  if (user) {
    res.json({ progress: user.progress });
  } else {
    res.json({ progress: [] });
  }
});

// ### End of user Learning Progress ###

// ### quiz ###

app.use('/images', express.static('public/alphabet'));
const quizQuestion = [
    {id: 1, image: 'alphabet/c.png', options:['A', 'B', 'C', 'D'], answer:'C'},
    {id: 2, image: 'alphabet/g.png', options:['F', 'J', 'X', 'G'], answer:'G'},
    {id: 3, image: 'alphabet/t.png', options:['S', 'T', 'Y', 'Z'], answer:'T'}
];

app.get('/quiz', (req, res) => {
    const questionAnswer = quizQuestion.map(q => ({
        id: q.id,
        image: q.image,
        options: q.options
    }));
    res.json(questionAnswer);
});

app.post('/quiz/submit', (req, res) => {
    const userAnswer = req.body.answer;

    let score = 0;
    userAnswer.forEach(answer => {
        const correct = quizQuestion.find(q => q.id === answer.id);
        if(correct && correct.answer === answer.answer) {
            score++;
        }
    });

    res.json({message: "Jawaban diterima", score: score, total: quizQuestion.length});
});

// ### end of quiz ###

app.listen(5000, () => {console.log("Server mulai di port 5000, http://localhost:5000/api")});