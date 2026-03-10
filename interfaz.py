from fastapi import APIRouter
from fastapi.responses import HTMLResponse

# Creamos el "alargador" para esta ruta
router_interfaz = APIRouter()

@router_interfaz.get("/", response_class=HTMLResponse)
def leer_interfaz():
    # Leemos el archivo HTML limpio y lo devolvemos
    with open("index.html", "r", encoding="utf-8") as archivo:
        html_content = archivo.read()
    
    return HTMLResponse(content=html_content)