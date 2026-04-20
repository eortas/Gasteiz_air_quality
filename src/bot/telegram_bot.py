import os
import json
import logging
from pathlib import Path
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Cargar variables de entorno (misma lógica que predict.py)
env_path = ROOT_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_latest_predictions() -> dict:
    """Extrae el JSON de la última predicción del pipeline."""
    # En producción (Render), lo bajamos directo de GitHub para tener siempre el más reciente
    # asumiendo que el repositorio es público y se llama eortas/Gasteiz_air_quality
    url = "https://raw.githubusercontent.com/eortas/Gasteiz_air_quality/main/data/processed/predictions_latest.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.warning(f"No se pudo descargar JSON de GitHub ({e}). Usando fallback local...")

    # Fallback local para pruebas
    json_path = PROCESSED_DIR / "predictions_latest.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Error leyendo predicciones locales: {e}")
        return None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "¡Hola! Soy *VitoriaAirBot* 🌍☁️\n\n"
        "Te ayudo a consultar la calidad del aire estimada en Vitoria-Gasteiz.\n\n"
        "👇 *Comandos rápidos:*\n"
        "/prevision : Muestra un resumen del último pronóstico diario.\n\n"
        "O simplemente pregúntame de forma natural:\n"
        "💬 _\"¿Es buen día para hacer deporte en el centro?\"_ o _\"¿Cómo va a estar el NO2 mañana?\"_"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def prevision_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    preds = get_latest_predictions()
    if not preds:
        await update.message.reply_text("Lo siento, no he podido leer los datos de las predicciones más recientes.")
        return
    
    date = preds.get("prediction_date", "desconocida")
    targs = preds.get("targets", {})
    
    msg = f"📊 *Previsión para el {date}*\n\n"
    
    # Seleccionar targets principales para mostrar
    main_targets = [
        ("NO2_zbe_d1", "NO2 en ZBE (Umbral: 25)"),
        ("PM10_zbe_d1", "PM10 en ZBE (Umbral: 45)"),
        ("PM2.5_zbe_d1", "PM2.5 en ZBE (Umbral: 15)"),
        ("ICA_zbe_d1", "Índice de Calidad ICA (Umbral: 40)")
    ]
    for key, label in main_targets:
        if key in targs:
            val = targs[key].get("prediction", 0)
            msg += f"👉 *{label}:* {val:.1f} µg/m³\n"
            
    # Añadir el comentario pre-generado por LLM si el ICA_zbe tiene uno
    if "ICA_zbe_d1" in targs and "foresight" in targs["ICA_zbe_d1"]:
        narrative = targs["ICA_zbe_d1"]["foresight"].get("narrative", "")
        if narrative:
            msg += f"\n💡 *Análisis General:*\n{narrative}\n"
            
    await update.message.reply_text(msg, parse_mode="Markdown")

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_msg = update.message.text
    if not GROQ_API_KEY:
        await update.message.reply_text("Error: GROQ_API_KEY no está configurada. No puedo procesar tu lenguaje.")
        return
        
    preds = get_latest_predictions()
    if not preds:
        await update.message.reply_text("No tengo datos de previsiones activos para poder responderte.")
        return
        
    date = preds.get("prediction_date", "desconocida")
    targs = preds.get("targets", {})
    
    # Construir el contexto numérico de los targets
    context_data = []
    for k, v in targs.items():
        val = v.get("prediction", 0)
        context_data.append(f"{k}: {val:.1f} µg/m³")
    context_str = ", ".join(context_data)
    
    # Prompt de sistema que será la personalidad del bot
    system_prompt = (
        "Eres un asistente amigable y experto ambiental que responde a los ciudadanos de Vitoria-Gasteiz. "
        "Tomas decisiones sobre si es bueno salir en bici, pasear o hacer deporte "
        "basándote EXCLUSIVAMENTE en los datos predictivos.\n\n"
        f"Día de la predicción: {date}\n"
        f"Datos del modelo Predictivo:\n{context_str}\n\n"
        "Reglas para ti:\n"
        "- Sé extremadamente breve y directo en tu consejo (1-2 párrafos).\n"
        "- Usa un lenguaje cotidiano, asertivo y añade algunos emojis para dar color.\n"
        "- Si te preguntan si pueden ir en bici o salir, tienes que valorar esto: NO2 > 25 (malo), PM10 > 45 (malo), PM2.5 > 15 (malo). Si los datos son inferiores a esos umbrales, anímalos con efusividad a salir.\n"
        "- Recuerda que 'zbe' es la zona centro (Zona de Bajas Emisiones) y 'out' son las afueras."
    )
    
    # Llamar a la API de Groq
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.5,
                "max_tokens": 400
            },
            timeout=15
        )
        if response.status_code == 200:
            bot_reply = response.json()["choices"][0]["message"]["content"].strip()
        else:
            bot_reply = "Hubo un pequeño error procesando mi respuesta (API). Intenta de nuevo en unos minutos."
            logger.error(f"Groq API error: {response.text}")
    except Exception as e:
        logger.error(f"Error HTTP request a Groq: {e}")
        bot_reply = "No he podido conectar con mi cerebro procesador en este momento."
        
    await update.message.reply_text(bot_reply)

def main():
    if not TELEGRAM_TOKEN:
        logger.error("Error: No se encontró TELEGRAM_BOT_TOKEN en el archivo .env")
        return
        
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("prevision", prevision_command))
    app.add_handler(MessageHandler(filters.TEXT & ~(filters.COMMAND), chat_handler))
    
    # Soporte para Webhooks en Render (Web Service) o Polling local
    webhook_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("WEBHOOK_URL")
    
    if webhook_url:
        port = int(os.environ.get("PORT", "10000"))
        logger.info(f"Iniciando VitoriaAirBot en modo WEBHOOK en port {port}...")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=webhook_url
        )
    else:
        logger.info("Iniciando VitoriaAirBot en modo polling (desarrollo local)...")
        app.run_polling()

if __name__ == "__main__":
    main()
