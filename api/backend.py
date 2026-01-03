from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from fastapi.middleware.cors import CORSMiddleware
from model.models import preprocess, model, labels

app = FastAPI()

# Allow CORS from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = []
    for i in range(top5_prob.size(0)):
        result.append({
            "label": labels[top5_catid[i]],
            "probability": round(top5_prob[i].item(), 4)
        })
    
    return {"predictions": result}
