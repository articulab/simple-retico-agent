def strict_formatt(text, max_lenght):
    ftext = []
    lines = text.split("\n")
    for line in lines:
        # check indenr level
        flines = [
            line[i * max_lenght : (i + 1) * max_lenght]
            for i in range(len(line) // max_lenght + 1)
        ]
        ftext.extend(flines)
    return "\n".join(ftext)


import textwrap


def format_docstring(docstring: str, max_length: int = 88) -> str:
    """
    Format a docstring to have lines with a maximum length.

    Parameters:
    - docstring (str): The original docstring to be formatted.
    - max_length (int): The maximum line length. Default is 88 characters.

    Returns:
    - str: The formatted docstring with each line having at most `max_length` characters.
    """
    # Split the original docstring into paragraphs
    paragraphs = docstring.split("\n\n")

    # Wrap each paragraph to the specified line length
    formatted_paragraphs = [
        "\n".join(textwrap.wrap(paragraph, width=max_length))
        for paragraph in paragraphs
    ]

    # Join the formatted paragraphs with a blank line in between
    return "\n\n".join(formatted_paragraphs)


def format_docstring_2(docstring: str, max_length: int = 72, indent=4) -> str:
    """
    Format a docstring to have lines with a specified maximum length and align it to a specific style.

    Parameters:
    - docstring (str): The original docstring to be formatted.
    - max_length (int): The maximum line length. Default is 88 characters.

    Returns:
    - str: The formatted docstring with each line having at most `max_length` characters.
    """
    tab = "    "
    paragraphs = docstring.split("\n\n")

    formatted_paragraphs = []

    for paragraph in paragraphs:
        # Check if the paragraph is part of an argument list (starts with indentation)
        if (
            paragraph.strip().startswith("Args:")
            or paragraph.strip().startswith("Returns:")
            or paragraph.strip().startswith("Raises:")
        ):
            # Split the argument section by lines, and format the first line as the header
            lines = paragraph.split("\n")
            formatted_paragraph = [lines[0]]

            # Rewrap and indent subsequent lines of the argument description
            for line in lines[1:]:
                nb_indentations = (len(line) - len(line.lstrip())) // indent
                wrapped_lines = textwrap.wrap(
                    line.strip(),
                    width=max_length,
                    subsequent_indent=tab * (nb_indentations + 1),
                    initial_indent=tab * nb_indentations,
                )
                # formatted_paragraphs.append(wrapped_lines[0])
                formatted_paragraph.append("\n".join(wrapped_lines))
            formatted_paragraphs.append("\n".join(formatted_paragraph))
        else:
            # Format non-argument paragraphs normally
            lines = paragraph.splitlines()
            indentation = len(lines[0]) - len(lines[0].lstrip())
            indent_str = " " * indentation
            wrapped_lines = textwrap.wrap(
                paragraph, width=max_length, subsequent_indent=indent_str
            )
            wrapped_paragraph = "\n".join(wrapped_lines)
            formatted_paragraphs.append(wrapped_paragraph)

    # Join paragraphs with double newlines for consistent paragraph separation
    return "\n\n".join(formatted_paragraphs)


t = """lfwleflwfe

        fewfeffewfe

        '''Prepare run by instanciating the Thread that transcribes the user speech.

        Prepare run by instanciating the Thread that transcribes the user speech.
"""

# print(t)
# print("\nlala")
print(format_docstring_2(t))

# tab = "    "
# r = textwrap.wrap(
#     "language (string): language of the desired model, has to be contained in the constant LANGUAGE_MAPPING.",
#     width=88,
#     initial_indent=tab * 2,
#     subsequent_indent=tab * 3,
# )

# print(r)
# print("\n".join(r))
