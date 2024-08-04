import logging
import os

from ..utils.slack.send_only import send_text

logger = logging.getLogger(__name__)


def check_one_env(env_var, secret=False, secret_show_first_n=0):
    value = os.environ.get(env_var)
    if value is None:
        logger.error(f"😡 Please set the environment variable {env_var}.")
        return False

    if secret:
        if secret_show_first_n > 0:
            logger.info(
                f"✅ {env_var} is set to {value[:secret_show_first_n]}{'*' * (len(value) - secret_show_first_n)}"
            )
        else:
            logger.info(f"✅ {env_var} is set.")
    else:
        logger.info(f"✅ {env_var} is set to {value}")
    return True


def check_many_env(env_vars, secret=False, secret_show_first_n=0):
    ret = True
    for env_var in env_vars:
        if not check_one_env(env_var, secret, secret_show_first_n):
            ret = False
    return ret


def check_env():
    secrets_checked = check_many_env(["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"], True, 5)
    normal_checked = check_many_env(["SLACK_CHANNEL_ID"])

    return secrets_checked and normal_checked


def check_send_text():
    logger.info("🚀 Sending a test message to the Slack channel...")
    response = send_text("💪 Checking health of the Slack bot.")

    if response is None:
        logger.error("😡 The response is None.")
        return False

    if response["ok"]:
        logger.info("✅ The message was sent successfully.")
        return True

    logger.error(f"😡 The message was not sent successfully. Response: {response}")
    return False
