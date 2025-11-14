import subprocess, sys, time, atexit, platform
import webview

PORT = "8501"
URL  = f"http://127.0.0.1:{PORT}"

# Choose creation flags only if on Windows
creationflags = 0
if platform.system() == "Windows":
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

# Start Streamlit as a background process
proc = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "FINAL.py",
    "--server.address=127.0.0.1",
    f"--server.port={PORT}",
    "--browser.gatherUsageStats=false",
    "--server.headless=true"
], creationflags=creationflags)

def cleanup():
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        pass

atexit.register(cleanup)

# Give Streamlit a second to start
time.sleep(2)

window = webview.create_window("Levee Fault Detection", URL, width=1200, height=800, resizable=True)
webview.start()
