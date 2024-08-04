import logging
import subprocess

logger = logging.getLogger(__name__)


class FontNotInstalledError(Exception):
    pass


def verify_fonts_installed():
    """
    This function verifies that the FiraCode font is installed because Rich Console.export_svg() uses it.

    Actually, the svg itself doesn't need the font but when we convert it to PDF using cairosvg, the font is needed.
    """
    cmd = ["fc-match", "FiraCode:style=Regular"]
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

    if not output.decode("utf-8").startswith("FiraCode-Regular"):
        raise FontNotInstalledError("FiraCode")

    cmd = ["fc-match", "FiraCode:style=Bold"]
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

    if not output.decode("utf-8").startswith("FiraCode-Bold"):
        raise FontNotInstalledError("FiraCode")

    # log.debug('Courier and Helvetica Neue were found to be installed.')


def verify_fonts_bool():
    try:
        verify_fonts_installed()
        logger.info("✅ FiraCode font is installed.")
        return True
    except FontNotInstalledError:
        logger.error("😡 FiraCode font is not installed.")
        return False


if __name__ == "__main__":
    verify_fonts_installed()
