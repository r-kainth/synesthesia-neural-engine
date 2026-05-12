# Synesthetic Engine

> An AI-driven multimodal pipeline translating acoustic features into cinematic visuals using LLMs.

## Overview
The Synesthetic Engine utilizes a 'Modular Stitch' architecture. By integrating **Whisper** for speech, **Librosa** for signal processing, **Gemini 1.5** for semantic understanding, and **Stable Diffusion** for image generation, the system creates a cohesive audio-to-visual bridge.

## Architecture Pipeline

### Stage 1: The Ear
- **Acoustic Analysis**: Uses `librosa` to extract BPM, RMS energy, Spectral Centroid, Zero Crossing Rate (ZCR), and Bandwidth.
- **Speech Transcription**: Employs `faster-whisper` for real-time or batch audio-to-text processing.

### Stage 2: The Brain
- **NLU Translation**: Gemini 1.5 processes transcription and acoustic data.
- **JSON Prompts**: Generates structured visual descriptions (JSON) that map audio intensity to visual complexity.

### Stage 3: The Eye
- **Visual Synthesis**: Powered by `diffusers` and Stable Diffusion.
- **Hardware**: Optimized via DirectML for high performance on AMD GPUs and standard CUDA for NVIDIA.

## Project Structure
```bash
synesthetic-engine/
├── data/               # Raw audio samples
├── output/             # Generated visual frames
├── src/
│   ├── extractor.py    # Librosa & Whisper logic
│   ├── brain.py        # Gemini API integration
│   └── visualizer.py   # Stable Diffusion pipeline
├── .env                # API Keys
└── README.md
```

## Installation
1. **Python Requirement**: Ensure Python 3.11+ is installed.
2. **Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate
   ```
3. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configuration**: Create a `.env` file and set your `GEMINI_API_KEY`.

## Usage
Run the extraction phase:
```bash
python src/extractor.py --input sample.wav
```
Generate the visual prompts:
```bash
python src/brain.py --data output/features.json
```

## Author
**Ritik Kainth**