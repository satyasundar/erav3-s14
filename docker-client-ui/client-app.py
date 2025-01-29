from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests

# Initialize FastAPI app
app = FastAPI()

# URL of the model-serving FastAPI container
MODEL_API_URL = "http://model-container:8000/generate"

# Templates directory for HTML files
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the input form UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_class=HTMLResponse)
async def generate_text(request: Request, prompt: str = Form(...)):
    """
    Send the prompt to the model-serving container and display the generated text.
    """
    # Payload for the model-serving API
    payload = {"prompt": prompt, "max_length": 50, "temperature": 0.7}

    try:
        # Send a POST request to the model-serving container
        response = requests.post(MODEL_API_URL, json=payload)
        if response.status_code == 200:
            generated_text = response.json().get("generated_text", "")
        else:
            generated_text = f"Error: {response.status_code}, {response.json()}"
    except Exception as e:
        generated_text = f"An error occurred: {e}"

    # Render the UI with the result
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prompt": prompt, "generated_text": generated_text},
    )
