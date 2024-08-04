"""
This module is used to start the Slack app in interactive mode.

Examples:
    >>> from emp.utils.slack import app
    >>> from emp.utils.slack.interactive import handler
    >>> @app.message("hello")  # doctest: +SKIP
    ... def hello_message(body: dict[str, Any], say):
    ...     logger.info("Received a hello command")
    ...     logger.info(f"body: {pformat(body)}")
    ...     say(text=f"Hi <@{body['event']['user']}>!")

    >>> handler.start()  # doctest: +SKIP
"""

import os

from slack_bolt.adapter.socket_mode import SocketModeHandler

from . import app, bot_token

app_token = os.environ.get("SLACK_APP_TOKEN", None)

if bot_token is None:
    handler = None
else:
    assert app_token is not None, "SLACK_APP_TOKEN environment variable is required"
    assert app is not None, "SLACK_BOT_TOKEN environment variable is required"
    handler = SocketModeHandler(app, app_token)
