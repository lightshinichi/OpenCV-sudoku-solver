# OpenCV Sudoku Solver

Point your camera at a sudoku puzzle and watch the solution appear overlaid on the board in real time. The project has two parts:

- **`solver.py`** — a 9x9 sudoku solver using a backtracking algorithm.
- **`camera_solver.py`** — an OpenCV pipeline that detects a sudoku board through the camera (or in a photo), reads the digits, solves the puzzle with `solver.py`, and projects the answer back onto the live frame.

## Setup

Requires Python 3.9+. No Tesseract or deep-learning framework is needed — digit recognition is done by template matching against your system fonts.

```bash
python3 -m venv .venv
.venv/bin/pip install opencv-python numpy pillow
```

## Usage

### Live camera

```bash
.venv/bin/python camera_solver.py            # default camera
.venv/bin/python camera_solver.py --camera 1 # a different camera
```

Hold the puzzle so the whole grid is visible, roughly face-on and evenly lit. Once the board is read successfully, the missing digits are drawn in green on top of the board and tracked as the camera moves.

Keys: **`r`** re-scan (new puzzle), **`s`** save a snapshot, **`q`** quit.

> On macOS, the first launch will ask you to grant camera access to your terminal (System Settings → Privacy & Security → Camera).

### Still image

```bash
.venv/bin/python camera_solver.py --image photo.jpg
```

Prints the detected and solved boards to the console and writes `photo_solved.png` with the overlay. Add `--no-show` to skip opening a window.

### Solver only

```bash
.venv/bin/python solver.py
```

Solves the example board hardcoded in the script. To solve your own puzzle, edit the `board` list (zeros are empty cells) or import the module:

```python
from solver import solve, print_board

board = [[7,8,0,4,0,0,1,2,0], ...]  # 9 rows of 9, 0 = empty
solve(board)      # fills the board in place, returns True if solvable
print_board(board)
```

## How the camera pipeline works

Each frame goes through five stages:

1. **Find the board** — grayscale, blur, and adaptive threshold, then take the largest 4-corner contour in the frame.
2. **Flatten it** — perspective-warp the board to a top-down 450x450 square.
3. **Read the digits** — split into 81 cells; in each cell, isolate the digit blob (discarding grid-line remnants), then classify it by cosine similarity against digits 1–9 rendered in common system fonts (Arial, Helvetica, Times, …), with OpenCV's Hershey fonts as a fallback. Low-confidence matches are treated as empty.
4. **Solve** — validate that the detected givens are consistent and that there are at least 17 of them, then run the backtracking solver from `solver.py`. If OCR misread a digit, the givens usually conflict and the frame is rejected rather than showing a wrong answer.
5. **Overlay** — draw only the filled-in digits on a flat canvas and inverse-warp it back into the camera frame, so the solution sits on the physical board.

The solved puzzle is cached, so after the first successful read the script only re-detects the board's position each frame — press `r` to scan a different puzzle.

## How the solver works

### The `solve` function

The main function that solves the puzzle. It works recursively by trying to fill empty cells with valid numbers from 1 to 9. If it reaches a position where no valid number fits, it backtracks to the previous step and tries a different number.

### The `check_if_valid` function

Checks whether a given number can be placed at a specified position, ensuring it doesn't already appear in the same row, column, or 3x3 subgrid.

### The `print_board` function

Prints the board in a user-friendly format, with lines separating each 3x3 subgrid.

### The `find_empty` function

Scans the board for the next empty spot (denoted by 0) and returns its (row, column), which `solve` uses to place the next number.
