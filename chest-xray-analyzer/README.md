# 🫁 AI Chest X-Ray Analyzer

A real-time AI-powered chest X-ray analysis tool that detects 18 different conditions using deep learning.

## Features
- **Instant Analysis**: Upload X-ray → Get results in seconds
- **18 Conditions**: Detects pneumonia, cardiomegaly, effusion, and more
- **Confidence Scores**: Shows probability for each condition
- **Visual Interface**: Clean drag-and-drop web interface
- **Privacy First**: All processing happens locally

## Quick Start

### 1. Setup (One-time)
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Run Backend
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload
```

### 3. Open Frontend
Open `frontend/index.html` in your browser

## How It Works
- Uses Stanford's CheXNet architecture (DenseNet-121)
- Trained on 100,000+ chest X-rays
- Returns probabilities for 18 pathologies
- Highlights abnormalities with confidence scores

## Conditions Detected
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Lung Lesion
- Lung Opacity
- Mass
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax

## ⚠️ Medical Disclaimer
This is a prototype for educational purposes. Not FDA approved. Always consult healthcare professionals for medical decisions.

## Next Features (Coming Soon)
- CT scan support
- Historical tracking
- PDF reports
- DICOM file support