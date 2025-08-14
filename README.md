
<p align="center">
  <img src="https://i.ibb.co/whx9Cp5q/tool.png" alt="NeuroTune Banner" width="600"/>
</p>

# NeuroTune v0.3

NeuroTune is an advanced Python3-based audio analysis and visualization tool.  
It provides an intuitive **GUI** for recording, processing, and analyzing audio in real time.  
The tool is designed for sound engineers, researchers, and enthusiasts who want to explore sound signals interactively.  
With real-time plotting, waveform rendering, and audio playback features, NeuroTune transforms audio data into meaningful visual insights.

## Features

- 🎛 **Interactive GUI** built with PyQt6 for ease of use
- 🎙 **Live audio recording** via PyAudio
- 📈 **Dynamic waveform and spectrum visualization** using pyqtgraph
- 🎶 **Audio file manipulation** (load, split, convert) with pydub
- ➗ **Mathematical signal processing** powered by NumPy
- 🔄 Real-time updates and responsive interface

## Installation & Setup

1. **Update your system (optional):**  
```bash
sudo apt update && sudo apt upgrade -y
```

2. **Install Python3 & venv (if not already installed):**  
```bash
sudo apt install python3 python3-venv -y
```

3. **Create a Virtual Environment:**  
```bash
python3 -m venv neurotune-env
```

4. **Activate the Virtual Environment:**  
```bash
source neurotune-env/bin/activate
```

5. **Install Required Libraries:**  
```bash
pip install PyQt6 numpy pyaudio pyqtgraph pydub
```

## Running NeuroTune

- Place `NeuroTune.py` inside your working directory.
- Make sure your virtual environment is active:  
```bash
source neurotune-env/bin/activate
```
- Run the application:  
```bash
python3 NeuroTune.py
```
- Use the GUI to:
  - Record live audio from your microphone
  - Open and play existing audio files
  - Visualize signals in real time (waveform & spectrum)
  - Apply basic processing such as trimming or segmenting audio

## Notes

- **Microphone permissions** may be required depending on your operating system.  
- If PyAudio installation fails, install PortAudio manually:  
```bash
sudo apt install portaudio19-dev
```
- Best experienced on systems with a working sound card and GUI environment.

## Support

If you like this project, consider buying me a donut 🍩  
**Bitcoin Address:** `bc1q5085r7x0gak6pcsqkrafxyfdnlqjhysr7cgfkj`

## Author

Discord & Telegram: `0xnoag`
