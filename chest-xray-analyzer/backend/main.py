from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from typing import Dict, List
import torchxrayvision as xrv

app = FastAPI(title="Chest X-Ray Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
    'Lung Opacity', 'Enlarged Cardiomediastinum'
]

def load_model():
    global MODEL
    if MODEL is None:
        print("Loading DenseNet model...")
        MODEL = xrv.models.DenseNet(weights="densenet121-res224-all")
        MODEL.eval()
        print("Model loaded successfully!")
    return MODEL

@app.on_event("startup")
async def startup_event():
    load_model()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((224, 224))
    
    img_array = np.array(image)
    
    img_array = img_array.astype(np.float32) / 255.0
    
    img_array = (img_array - 0.5) / 0.5
    
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def interpret_predictions(predictions: np.ndarray, threshold: float = 0.5) -> Dict:
    results = {
        "findings": [],
        "normal_probability": 0,
        "top_conditions": []
    }
    
    prediction_dict = {}
    for i, pathology in enumerate(PATHOLOGIES[:len(predictions[0])]):
        prob = float(predictions[0][i])
        prediction_dict[pathology] = prob
        
        if prob > threshold:
            results["findings"].append({
                "condition": pathology,
                "confidence": f"{prob*100:.1f}%",
                "severity": "High" if prob > 0.7 else "Moderate"
            })
    
    sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
    results["top_conditions"] = [
        {"condition": cond, "confidence": f"{prob*100:.1f}%"} 
        for cond, prob in sorted_predictions[:3]
    ]
    
    abnormal_score = max(prediction_dict.values())
    results["normal_probability"] = f"{(1 - abnormal_score)*100:.1f}%"
    
    if not results["findings"]:
        results["summary"] = "No significant abnormalities detected"
        results["recommendation"] = "X-ray appears normal. Regular check-ups recommended."
    else:
        results["summary"] = f"Found {len(results['findings'])} potential abnormalities"
        results["recommendation"] = "Consider consulting with a healthcare professional for further evaluation."
    
    return results

@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        img_tensor = preprocess_image(image)
        
        model = load_model()
        with torch.no_grad():
            predictions = model(img_tensor).cpu().numpy()
        
        predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        
        results = interpret_predictions(predictions)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        results["image_preview"] = f"data:image/png;base64,{img_base64}"
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Chest X-Ray Analyzer API",
        "endpoints": {
            "/analyze": "POST - Upload X-ray image for analysis",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}