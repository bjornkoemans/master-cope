from typing import Literal, Union, Tuple


def display_indented_list(
    array: list,
    title: str,
    indent: int = 2,
) -> None:
    """Display a list of strings in an indented format.

    Args:
        array (list[str]): List of strings to display
        indent (int): Number of spaces to indent each line
    """
    print(f"{title}:")
    for item in array:
        print(" " * indent + str(item))


def _colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{}\033[0m".format(r, g, b, text)


# Color presets - optimized for white backgrounds

# Create type alias for color presets
ColorPreset = Literal[
    "green", "red", "blue", "yellow", "purple", "cyan", "white", "orange", "black"
]
RGBColor = Tuple[int, int, int]
COLOR_PRESETS: dict[ColorPreset, RGBColor] = {
    "green": (0, 128, 0),      # Dark green - readable on white
    "red": (178, 34, 34),       # Dark red - readable on white
    "blue": (0, 0, 139),        # Dark blue - readable on white
    "yellow": (184, 134, 11),   # Dark goldenrod - readable on white
    "purple": (75, 0, 130),     # Indigo - readable on white
    "cyan": (0, 139, 139),      # Dark cyan - readable on white
    "white": (100, 100, 100),   # Dark gray (white is invisible on white)
    "orange": (204, 85, 0),     # Dark orange - readable on white
    "black": (0, 0, 0),         # Black - for regular text
}


def print_colored(text: str, color: Union[ColorPreset, RGBColor] = "green") -> None:
    """Print text in a specific color.

    Args:
        text (str): Text to print
        color (Union[ColorPreset, RGBColor]): Either a color preset name or RGB tuple
    """
    if isinstance(color, str):
        if color not in COLOR_PRESETS:
            raise ValueError(
                f"Color preset '{color}' not found. Available presets: {list(COLOR_PRESETS.keys())}"
            )
        color = COLOR_PRESETS[color]
    """Print text in a specific RGB color.

    Args:
        text (str): Text to print
        color (tuple[int, int, int]): RGB color values
    """
    r, g, b = color
    print(_colored(r, g, b, text))
