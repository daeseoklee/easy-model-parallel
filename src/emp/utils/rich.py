import os
from collections.abc import Iterable
from typing import Any

from ansi2html import Ansi2HTMLConverter
from rich.console import Console
from rich.terminal_theme import MONOKAI
from rich.traceback import Traceback

CONSOLE_WIDTH = 130


def rich_print_to_string(objs: Iterable[Any]) -> str:
    """Convert a rich console print output to a string."""
    console = Console(
        width=CONSOLE_WIDTH,
        no_color=True,
        highlight=False,
        record=True,
        file=open(os.devnull, "w"),  # noqa: SIM115
    )

    if not isinstance(objs, Iterable):
        objs = [objs]
    for obj in objs:
        console.print(obj)

    return console.export_text()


def rich_print_to_html(objs: Iterable[Any], backend="ansi2html") -> str:
    """
    Convert a rich console print output to an html string.

    rich backend can be prettier with the following caveats:
    1. It produces a larger file size (2x)
    2. It isn't as easy to read as a plain HTML code due to alignment and many html tags.
    3. Slack doesn't understand the HTML code produced by `console.export_html()` and treat it as a binary file.

    Args:
        objs: The objects to print. Possibly but not limited to rich objects.
        backend: "ansi2html" or "rich". Defaults to "ansi2html".
    """
    assert backend in ("ansi2html", "rich")

    console = Console(width=CONSOLE_WIDTH, record=True, file=open(os.devnull, "w"))  # noqa: SIM115

    if not isinstance(objs, Iterable):
        objs = [objs]
    for obj in objs:
        console.print(obj)

    if backend == "rich":
        return console.export_html(theme=MONOKAI)

    tb_ansi = console.export_text(styles=True)  # text with ansi color codes
    return Ansi2HTMLConverter().convert(tb_ansi)


def rich_print_to_svg(objs: Iterable[Any], title: str) -> str:
    """Convert a rich console print to an svg string."""
    console = Console(width=CONSOLE_WIDTH, record=True, file=open(os.devnull, "w"))  # noqa: SIM115

    if not isinstance(objs, Iterable):
        objs = [objs]
    for obj in objs:
        console.print(obj)

    return console.export_svg(title=title)


def rich_traceback_to_string(tb: Traceback) -> str:
    """
    Convert a rich traceback to a string.

    Examples:
        >>> try:   # doctest: +ELLIPSIS
        ...    1 / 0
        ... except ZeroDivisionError:
        ...     import rich.traceback
        ...     tb = rich.traceback.Traceback()
        ...     print(rich_traceback_to_string(tb))
        ╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
        ...
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
        ZeroDivisionError: division by zero
        <BLANKLINE>
    """
    return rich_print_to_string([tb])


def rich_traceback_to_html(tb: Traceback, backend="ansi2html") -> str:
    """
    Convert a rich traceback to an html string.

    Examples:
        >>> try:   # doctest: +ELLIPSIS
        ...    1 / 0
        ... except ZeroDivisionError:
        ...     import rich.traceback
        ...     tb = rich.traceback.Traceback()
        ...     print(rich_traceback_to_html(tb))
        <!DOCTYPE HTML ...
        </html>
        <BLANKLINE>
    """
    return rich_print_to_html([tb], backend=backend)


def rich_traceback_to_svg(tb: Traceback, title: str) -> str:
    """
    Convert a rich traceback to an svg string.

    Examples:
        >>> try:   # doctest: +ELLIPSIS
        ...    1 / 0
        ... except ZeroDivisionError:
        ...     import rich.traceback
        ...     tb = rich.traceback.Traceback()
        ...     print(rich_traceback_to_svg(tb, "title"))
        <svg class="rich-terminal" ...
        </svg>
        <BLANKLINE>
    """
    return rich_print_to_svg([tb], title)
