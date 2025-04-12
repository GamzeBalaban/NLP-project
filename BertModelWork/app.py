from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# FastAPI başlatma
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  

# Model ve tokenizer yükleme
MODEL_PATH = "bert_model2.pt"

# Model ve tokenizer yükleme
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

# Kaydedilen model ağırlıklarını yükle
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Modeli cihazda çalışacak şekilde ayarla
model.to(device)
model.eval()

# API giriş modeli
class TextInput(BaseModel):
    text: str

# API endpoint
@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    # Tokenize giriş verisi
    tokens = tokenizer(input_data.text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Modeli çalıştır
        output = model(**tokens)
    
    # Tahmini al
    prediction = torch.argmax(output.logits, dim=1).item()
    sentiment = ["negative", "neutral", "positive"][prediction]
    
    return {"text": input_data.text, "sentiment": sentiment}

# API çalıştırma
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
