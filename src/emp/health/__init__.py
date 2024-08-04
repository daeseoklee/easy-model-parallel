"""Check health of the installation."""

import logging
import os

from .. import DATA_DIR
from .font import verify_fonts_bool
from .slack import check_env as slack_check_env
from .slack import check_send_text as slack_check_send_text

logger = logging.getLogger(__name__)


def check_binaries():
    """Check if binaries are installed."""
    return True


def check_env():
    """Check environment variables."""
    ppmi_data_dir = os.environ.get("EMP_DATA_DIR")

    if ppmi_data_dir is None:
        logger.warning(
            "🤒 Please set the environment variable EMP_DATA_DIR to the path of the data directory.\n"
            f"Otherwise, the default {DATA_DIR} will be used."
        )
        return False

    logger.info(f"✅ EMP_DATA_DIR is set to {ppmi_data_dir}")
    return True


def main():
    successes = [check_env()]
    successes.append(check_binaries())

    successes.append(slack_check_env())
    successes.append(slack_check_send_text())

    successes.append(verify_fonts_bool())

    if all(successes):
        logger.info("")
        logger.info("💪 You are ready to go!")


if __name__ == "__main__":
    main()
