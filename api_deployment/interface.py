from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import os


router_interface = APIRouter()

@router_interface.get("/", response_class=HTMLResponse)
def read_interface():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as archive:
        html_content = archive.read()
    
    return HTMLResponse(content=html_content)