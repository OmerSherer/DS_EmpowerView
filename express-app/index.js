const express = require('express');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the "web-ui" folder
app.use(express.static('web-ui'));

// Define routes
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/web-ui/index.html');
});

app.post('/submit', (req, res) => {
  // Process form data or perform database operations here

  res.send('Form submitted successfully!');
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
