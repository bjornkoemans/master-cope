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
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


# Color presets

# Create type alias for color presets
ColorPreset = Literal[
    "green", "red", "blue", "yellow", "purple", "cyan", "white", "orange"
]
RGBColor = Tuple[int, int, int]
COLOR_PRESETS: dict[ColorPreset, RGBColor] = {
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
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
