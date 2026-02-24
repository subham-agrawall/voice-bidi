"""AWS credential helpers for local use.

This file provides a single function `load_aws_credentials_from_file()` which
will load credentials from a local AWS credentials file (default
`~/.aws/credentials`) and populate corresponding environment variables. The
function also returns a dict of the discovered credential values for
programmatic use.

Usage (example in `main.py`):

    from bidi_demo.app.aws_credentials import load_aws_credentials_from_file

    creds = load_aws_credentials_from_file()  # tries ~/.aws/credentials, profile 'adfs'

"""
from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_aws_credentials(
    credentials_path: Optional[str | Path] = None, *, profile: str = "adfs"
) -> Dict[str, str]:
    """Load AWS credentials from file and set environment variables.

    Args:
      credentials_path: Optional path to credentials file. If not provided,
                        defaults to ~/.aws/credentials.
      profile: The profile name to read (default: 'adfs').

    Returns:
      A dict containing any discovered credential keys (e.g.,
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
      AWS_DEFAULT_REGION). Empty dict if none found.
    """
    path = Path(credentials_path) if credentials_path else Path.home() / ".aws" / "credentials"
    path = path.expanduser()

    creds: Dict[str, str] = {}

    if not path.exists():
        logger.warning("Credentials file not found at %s", path)
        return creds

    config = configparser.ConfigParser()
    try:
        config.read(path)
    except Exception as e:
        logger.warning("Failed to read credentials file %s: %s", path, e)
        return creds

    if profile not in config:
        logger.warning("Profile %s not found in credentials file %s", profile, path)
        return creds

    section = config[profile]
    access_key = section.get("aws_access_key_id")
    secret_key = section.get("aws_secret_access_key")
    session_token = section.get("aws_session_token")
    region = section.get("region")

    if access_key:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
        creds["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
        creds["AWS_SECRET_ACCESS_KEY"] = secret_key
    if session_token:
        os.environ.setdefault("AWS_SESSION_TOKEN", session_token)
        creds["AWS_SESSION_TOKEN"] = session_token
    if region:
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        creds["AWS_DEFAULT_REGION"] = region
        
    logger.info("AWS credentials loaded from %s (profile=%s)", path, profile)
    return creds


__all__ = ["load_aws_credentials"]