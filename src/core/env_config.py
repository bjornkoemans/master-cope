from .display import print_colored, ColorPreset, RGBColor
from typing import Union, Any

DEBUG = False


def debug_print_colored(
    text: Any, color: Union[ColorPreset, RGBColor] = "white"
) -> None:
    """Print text in a specific color for debugging.

    Args:
        text (Any): Text to print
        color (Union[ColorPreset, RGBColor]): Either a color preset name or RGB tuple
    """
    if globals().get("DEBUG", False):
        print_colored(text, color)
