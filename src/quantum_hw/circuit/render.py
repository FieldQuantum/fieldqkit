"""Lightweight circuit rendering helpers for terminal/notebook output."""

try:
    from IPython.display import display, HTML
except Exception:  # pragma: no cover - fallback for non-notebook envs
    display = None
    HTML = None


def _render_lines(lines: list[str]) -> None:
    """Render lines.

    Args:
        lines (*list[str]*): Lines (``list[str]``).
    """
    fline = "\n" + "\n".join(lines)
    if display is None or HTML is None:
        print(fline)
        return
    formatted_string = fline.replace("\n", "<br>").replace(" ", "&nbsp;")
    html_content = f'<div style="overflow-x: auto; white-space: nowrap; font-family: consolas;">{formatted_string}</div>'
    display(HTML(html_content))


def draw_circuit(lines: list[str]) -> None:
    """Draw circuit.

    Args:
        lines (*list[str]*): Lines (``list[str]``).
    """
    _render_lines(lines)


def draw_circuit_simply(lines: list[str], lines_use: list[int], nqubits: int) -> None:
    """Draw circuit simply.

    Args:
        lines (*list[str]*): Lines (``list[str]``).
        lines_use (*list[int]*): Lines use (``list[int]``).
        nqubits (*int*): Number of qubits.
    """
    lines_use_set = set(lines_use)
    selected = []
    for idx in range(2 * nqubits):
        if idx in lines_use_set:
            selected.append(lines[idx])
    for idx in range(2 * nqubits, len(lines)):
        selected.append(lines[idx])
    _render_lines(selected)
