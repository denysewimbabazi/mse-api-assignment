from pyngrok import ngrok
import uvicorn
import threading
import time
from dotenv import load_dotenv
import os

# --- Configuration ---
load_dotenv()

PORT = 8000
APP_MODULE = os.getenv("APP_MODULE")  
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")  
# ----------------------

def start_uvicorn():
    uvicorn.run(APP_MODULE, host="0.0.0.0", port=PORT, reload=False)

if __name__ == "__main__":
    # Authenticate ngrok if token is provided
    if NGROK_AUTHTOKEN:
        ngrok.set_auth_token(NGROK_AUTHTOKEN)

    # Start ngrok tunnel
    public_url = ngrok.connect(PORT).public_url
    print(f"\n🚀 Public URL (for instructor): {public_url}")
    print(f"📘  Docs: {public_url}/docs\n")

    # Start FastAPI server in a background thread
    server_thread = threading.Thread(target=start_uvicorn, daemon=True)
    server_thread.start()

    try:
        # Keep process alive while both tunnels and API are running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server and ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
