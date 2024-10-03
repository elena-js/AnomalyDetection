const express = require('express');
const path = require('path');
const cors = require('cors');
const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const app = express();
const port = 3003;

// Promisify exec to use with async/await
const execPromise = promisify(exec);

// Base of the paths of the python scripts
const basePath = 'c:/Users/eleju/Desktop/TFM/';

// Use the CORS middleware
app.use(cors({
  origin: 'http://localhost:3000' // Only allow requests from http://localhost:3000
}));

// Define the directory for serving images
const imageDirectory = path.join(__dirname, 'figures');

// Use Express static middleware to serve the image directory
app.use('/figures', express.static(imageDirectory));

// Path to execute the PREPROCESS python script
app.get('/preprocess', async (req, res) => {
  try {

    // Capture userId from query string
    const userId = req.query.userId;

    // Validate that userId exists
    if (!userId) {
        return res.status(400).json({ error: 'userId is required' });
    }

    // Execute the script and wait for the result
    const { stdout, stderr } = await execPromise(`python ${basePath}preprocess.py ${userId}`, { 
      timeout: 600000, // timeout of 10min (600000 ms = 10 min)
      maxBuffer: 1024 * 1024 * 20 }); // increase buffer length to handle the large ammount of data
    
    if (stderr) {
        console.error(`Stderr: ${stderr}`);
        return res.status(500).json({ error: 'Error output from the script.' });
    }

    try {
        // Parse the standard output (stdout) as JSON
        const result = JSON.parse(stdout);
        res.json(result);
    } catch (parseError) {
        console.error(`Parse Error: ${parseError.message}`);
        res.status(500).json({ error: 'Failed to parse the script output.' });
    }
  } catch (error) {
      console.error(`Error: ${error.message}`);
      res.status(500).json({ error: 'An error occurred while executing the script.' });
  }
});

// Path to execute the ANOMALY DETECTION python script
app.get('/anomdet', async (req, res) => {
  try {

    // Capture userId from query string
    const userId = req.query.userId;

    // Validate that userId exists
    if (!userId) {
        return res.status(400).json({ error: 'userId is required' });
    }

    // Execute the script and wait for the result
    const { stdout, stderr } = await execPromise(`python ${basePath}anom_det.py ${userId}`, { timeout: 600000 }); // timeout of 10min (600000 ms = 10 min)
    
    if (stderr) {
        console.error(`Stderr: ${stderr}`);
        return res.status(500).json({ error: 'Error output from the script.' });
    }

    try {
        // Parse the standard output (stdout) as JSON
        const result = JSON.parse(stdout);
        res.json(result);
    } catch (parseError) {
        console.error(`Parse Error: ${parseError.message}`);
        res.status(500).json({ error: 'Failed to parse the script output.' });
    }
  } catch (error) {
      console.error(`Error: ${error.message}`);
      res.status(500).json({ error: 'An error occurred while executing the script.' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});