import os
import time
import hashlib
import subprocess
import signal
import logging

# ── CONFIGURATION ──────────────────────────────────────────────────────────
MAIN_SCRIPT = "main.py"
GITHUB_REPO_URL = "https://github.com/Ansh29754543/sedy-ai" # Update this!
CHECK_INTERVAL = 5  # Seconds between scans
LOG_FILE = "guardian_security.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [GUARDIAN] %(levelname)s: %(message)s"
)

def get_file_hash(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Capture the 'Clean' state of main.py at the moment the Guardian starts
INITIAL_HASH = get_file_hash(MAIN_SCRIPT)

def kill_main_server():
    """Finds and kills the uvicorn/fastapi process."""
    try:
        # Kills any process running main.py
        subprocess.run(["pkill", "-f", "main.py"], check=False)
        logging.info("Main server process terminated for safety.")
    except Exception as e:
        logging.error(f"Failed to kill process: {e}")

def self_heal():
    """The 'Fighting' mechanism: Pulls clean code from GitHub to overwrite viruses."""
    logging.warning("REVERTING TO GITHUB STATE: Overwriting local changes...")
    try:
        # 1. Discard local changes (the 'virus' or 'hack')
        subprocess.run(["git", "checkout", "main.py"], check=True)
        # 2. Pull the latest clean code from GitHub
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        logging.info("Self-heal successful. Code restored from GitHub.")
    except Exception as e:
        logging.error(f"Self-heal failed! Manual intervention required: {e}")

def restart_server():
    """Restarts the Sedy API after healing."""
    logging.info("Restarting Sedy API...")
    # This command starts main.py in the background
    subprocess.Popen(["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

def monitor():
    print("🛡️ Guardian Anti-Virus is watching GitHub & local files...")
    
    while True:
        current_hash = get_file_hash(MAIN_SCRIPT)
        
        # If the hash changed, someone (or a virus) edited the code
        if current_hash != INITIAL_HASH:
            logging.critical("CRITICAL: Unauthorized modification detected in main.py!")
            
            # STEP 1: Cut-off (Kill the server to stop the hack)
            kill_main_server()
            
            # STEP 2: Disconnect/Clear memory (Optional but safer)
            os.environ["GROQ_API_KEY"] = "LOCKED"
            
            # STEP 3: Fight/Heal (Restore from GitHub)
            self_heal()
            
            # STEP 4: Resume
            restart_server()
            
            # Update the hash so we don't loop forever
            global INITIAL_HASH
            INITIAL_HASH = get_file_hash(MAIN_SCRIPT)
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor()
