from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from model import get_model  # Assuming your model.py is in the same directory

# Initialize FastAPI app
app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
model = get_model(tokenizer)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)
model.eval()


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.7


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    prompt = request.prompt
    max_length = request.max_length
    temperature = request.temperature

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
