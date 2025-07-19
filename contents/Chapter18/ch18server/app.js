// app.js
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));

const OLLAMA_URL = 'http://localhost:11434/api/generate';

// Store ongoing analyses
const analysisJobs = new Map();

async function analyzeNovel(text) {
  try {
    console.log('Sending request to Ollama...');
    const requestBody = {
      model: 'gemma2:2b',
      prompt: `You are an expert storyteller who understands story structure, nuance, and content. Attached is a novel, please evaluate this novel for storylines, and suggest improvements that could be made in character development, plot, and emotional content. Be as verbose as needed to provide an in-depth analysis that would help the author understand how their work would be accepted:\n\n${text}`,
      stream: false
    };

    const response = await fetch(OLLAMA_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Error in analyzeNovel:', error);
    throw error;
  }
}

app.post('/analyze', upload.single('novel'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send('No file uploaded');
    }

    // Generate a unique job ID
    const jobId = Date.now().toString();
    
    // Store job status
    analysisJobs.set(jobId, { status: 'processing' });

    // Start analysis in background
    const fileContent = await fs.readFile(req.file.path, 'utf8');
    
    // Clean up uploaded file
    await fs.unlink(req.file.path);

    // Return job ID immediately
    res.json({ jobId });

    // Process in background
    analyzeNovel(fileContent)
      .then(result => {
        analysisJobs.set(jobId, { 
          status: 'completed',
          result: result
        });
      })
      .catch(error => {
        analysisJobs.set(jobId, { 
          status: 'error',
          error: error.message
        });
      });

  } catch (error) {
    console.error('Error initiating analysis:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint to check job status
app.get('/status/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = analysisJobs.get(jobId);
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  res.json(job);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

