def wrap_text(txt: str, n_char: int = 50, separator: str = "\n"):
    """
    Wrap a text at the next space after n_char characters.

    Parameters:
    - txt (str): The input text to be wrapped.
    - n_char (int, optional): The maximum number of characters before wrapping.
    - separator (str): The separator added after each wrapped line

    Returns:
    - str: The wrapped text.
    """
    if len(txt) < n_char:
        return txt
    else:
        last_space_index = txt[:n_char].rfind(" ")
        wrapped_line = txt[:last_space_index].strip() + separator
        remaining_text = txt[last_space_index + 1 :]
    return wrapped_line + wrap_text(remaining_text, n_char)
