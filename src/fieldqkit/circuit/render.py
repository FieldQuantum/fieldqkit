"""Lightweight circuit rendering helpers for terminal/notebook output.

SPDX-License-Identifier: Apache-2.0 AND MIT
Modified from quarkcircuit by FieldQuantum; this file has been altered from the
original and is redistributed as part of fieldqkit under the Apache License,
Version 2.0. The original work is licensed under the MIT License:

Copyright (c) 2024 XX Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

try:
    from IPython.display import display, HTML
except Exception:  # pragma: no cover - fallback for non-notebook envs
    display = None
    HTML = None


def _render_lines(lines: list[str]) -> None:
    """Display circuit diagram lines as HTML in Jupyter or plaintext in terminal.

    Args:
        lines (*list[str]*): List of pre-formatted circuit diagram line strings.
    """
    fline = "\n" + "\n".join(lines)
    if display is None or HTML is None:
        print(fline)
        return
    formatted_string = fline.replace("\n", "<br>").replace(" ", "&nbsp;")
    html_content = f'<div style="overflow-x: auto; white-space: nowrap; font-family: consolas;">{formatted_string}</div>'
    display(HTML(html_content))


def draw_circuit(lines: list[str]) -> None:
    """Display the complete circuit diagram with all qubit lines and gates.

    Args:
        lines (*list[str]*): List of pre-formatted circuit diagram line strings.
    """
    _render_lines(lines)


def draw_circuit_simply(lines: list[str], lines_use: list[int], nqubits: int) -> None:
    """Display a simplified circuit diagram showing only active qubit lines.

    Args:
        lines (*list[str]*): List of pre-formatted circuit diagram line strings.
        lines_use (*list[int]*): Indices of active qubit lines to display.
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
