import json
import os


def _is_development_environment() -> bool:
    return bool(os.environ.get("REINVENT_DEVELOPMENT_ENVIRONMENT"))
