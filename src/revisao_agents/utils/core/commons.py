import os
from dotenv import load_dotenv

def get_clean_key(env_var):
    load_dotenv()
    raw = os.getenv(env_var, "")
    return raw.replace("export ", "").split("=")[-1].strip().strip("'").strip('"')
