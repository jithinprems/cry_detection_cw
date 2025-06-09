# Cry Detection Project

## Overview
The Cry Detection project is designed to analyze audio files and detect cries using various audio processing techniques. It leverages libraries such as NumPy, Matplotlib, and Librosa to perform spectral analysis, rhythm detection, and noise detection.

## Features
- Audio preprocessing to enhance signal quality.
- Spectral analysis to compute features like spectral flatness and fundamental frequency.
- Rhythm detection to differentiate between music and non-music sounds.
- Noise detection to identify and filter out unwanted sounds.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cry-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
example:

```python
from src.cry_detection import CryDetector

file_path = 'path/to/audio/file.ogg'
detector = CryDetector(file_path)
detector.run(display=True)
```
