import os


def _is_development_environment() -> bool:
    try:
        is_dev = os.environ.get("DEVELOPMENT_ENVIRONMENT", False)
        return is_dev
    except:
        return False