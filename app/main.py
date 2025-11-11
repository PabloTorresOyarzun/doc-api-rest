from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os

from service_di import LocalDIService

load_dotenv()

LOCAL_ENDPOINT = "http://azure-di-custom:5000"
DI_MODELS = ["transport_01", "inovice_01"]

app = FastAPI(title="Azure Document Intelligence Local Tester")

templates = Jinja2Templates(directory="templates")

service = LocalDIService(LOCAL_ENDPOINT, os.getenv("AZURE_KEY"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": DI_MODELS})


@app.post("/process", response_class=HTMLResponse)
async def process_file(
    request: Request,
    model: str = Form(...),
    file: UploadFile = UploadFile(...)
):
    pdf_bytes = await file.read()
    result = service.analyze(model, pdf_bytes)

    extracted = {}
    if result.documents:
        doc = result.documents[0]
        for key, field in doc.fields.items():
            extracted[key] = field.value

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": DI_MODELS,
        "selected_model": model,
        "extracted": extracted,
    })