import os

from dotenv import load_dotenv


def get_clean_key(env_var):
    """
    Load an environment variable and clean it from common export patterns.
    For example, if the variable is set as "export KEY=VALUE", it will return "VALUE".

    Args:
        env_var: The name of the environment variable to load.

    Returns:
        The cleaned value of the environment variable, or an empty string if not found.
    """
    load_dotenv()
    raw = os.getenv(env_var, "")
    return raw.replace("export ", "").split("=")[-1].strip().strip("'").strip('"')
