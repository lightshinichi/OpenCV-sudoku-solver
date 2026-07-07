"""Live sudoku solver: point your camera at a sudoku board and see it solved.

Pipeline per frame:
  1. Find the board   -> largest 4-corner contour in the thresholded frame
  2. Flatten it       -> perspective warp to a 450x450 top-down view
  3. Read the digits  -> per-cell OCR by template matching against digits
                         rendered in common system fonts (no Tesseract needed)
  4. Solve            -> backtracking solver from solver.py in this repo
  5. Overlay          -> draw the solved digits and inverse-warp them back
                         onto the live frame

Usage:
    python camera_solver.py                 # default camera (index 0)
    python camera_solver.py --camera 1      # another camera
    python camera_solver.py --image pic.jpg # run on a still photo instead

Keys:  q = quit,  r = re-scan the board,  s = save a snapshot
"""

import argparse
import copy
import os

import cv2
import numpy as np

from solver import solve, check_if_valid

# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

WARP = 450          # side length of the flattened board
CELL = WARP // 9


def preprocess(frame):
    """Grayscale + blur + adaptive threshold; returns a binary image where
    grid lines and digits are white."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


def find_board_corners(thresh, min_area_frac=0.04):
    """Return the 4 corners of the largest quadrilateral contour, or None."""
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = thresh.shape[0] * thresh.shape[1]
    best, best_area = None, min_area_frac * frame_area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < best_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            best, best_area = approx.reshape(4, 2), area
    return best


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp_board(frame, corners):
    """Perspective-warp the board to a WARP x WARP square.
    Returns (warped_bgr, transform_matrix)."""
    src = order_corners(corners)
    dst = np.array([[0, 0], [WARP - 1, 0],
                    [WARP - 1, WARP - 1], [0, WARP - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (WARP, WARP))
    return warped, matrix


# ---------------------------------------------------------------------------
# Digit OCR via font-template matching
# ---------------------------------------------------------------------------

TEMPLATE_SIZE = 28

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
]


def normalize_glyph(binary):
    """Crop a white-on-black glyph to its bounding box and center it in a
    TEMPLATE_SIZE square, preserving aspect ratio. Returns float32 [0,1]."""
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return None
    glyph = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = glyph.shape
    scale = (TEMPLATE_SIZE - 8) / max(h, w)
    glyph = cv2.resize(glyph, (max(1, round(w * scale)),
                               max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)
    gh, gw = glyph.shape
    y0 = (TEMPLATE_SIZE - gh) // 2
    x0 = (TEMPLATE_SIZE - gw) // 2
    canvas[y0:y0 + gh, x0:x0 + gw] = glyph
    out = canvas.astype(np.float64) / 255.0
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def build_templates():
    """Render digits 1-9 in every available system font (plus OpenCV's
    Hershey fonts as a fallback) and return (stacked_templates, labels)."""
    templates, labels = [], []

    try:
        from PIL import Image, ImageDraw, ImageFont
        for path in FONT_CANDIDATES:
            if not os.path.exists(path):
                continue
            try:
                font = ImageFont.truetype(path, 64)
            except OSError:
                continue
            for digit in range(1, 10):
                img = Image.new("L", (96, 96), 0)
                draw = ImageDraw.Draw(img)
                draw.text((16, 8), str(digit), fill=255, font=font)
                arr = np.array(img)
                arr = (arr > 128).astype(np.uint8) * 255
                tpl = normalize_glyph(arr)
                if tpl is not None:
                    templates.append(tpl)
                    labels.append(digit)
    except ImportError:
        pass

    for hershey in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX):
        for thickness in (2, 4):
            for digit in range(1, 10):
                img = np.zeros((96, 96), dtype=np.uint8)
                cv2.putText(img, str(digit), (20, 76), hershey, 2.5,
                            255, thickness, cv2.LINE_AA)
                img = (img > 128).astype(np.uint8) * 255
                tpl = normalize_glyph(img)
                if tpl is not None:
                    templates.append(tpl)
                    labels.append(digit)

    return np.stack(templates).reshape(len(templates), -1), np.array(labels)


TEMPLATES, TEMPLATE_LABELS = None, None  # built lazily on first use


def classify_digit(cell_binary):
    """Return (digit, confidence) for a white-on-black cell glyph, or
    (0, 0.0) if it doesn't look like a digit."""
    global TEMPLATES, TEMPLATE_LABELS
    if TEMPLATES is None:
        TEMPLATES, TEMPLATE_LABELS = build_templates()
    tpl = normalize_glyph(cell_binary)
    if tpl is None:
        return 0, 0.0
    # cosine similarity (all rows unit-norm); errstate silences spurious
    # FP-flag warnings from Apple's Accelerate BLAS under NumPy 2.x
    with np.errstate(all="ignore"):
        scores = TEMPLATES @ tpl.ravel()
    best = int(np.argmax(scores))
    return int(TEMPLATE_LABELS[best]), float(scores[best])


def extract_cell_glyph(warped_thresh, row, col):
    """Cut one cell out of the thresholded warped board and isolate the
    digit blob, discarding grid-line remnants. Returns a binary image or
    None if the cell is empty."""
    pad = 3
    y0, x0 = row * CELL + pad, col * CELL + pad
    cell = warped_thresh[y0:y0 + CELL - 2 * pad, x0:x0 + CELL - 2 * pad]

    contours, _ = cv2.findContours(
        cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = cell.shape
    best, best_area = None, 0.02 * h * w  # ignore specks
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        cx, cy = x + cw / 2, y + ch / 2
        # a digit blob sits near the middle and isn't a thin sliver on an edge
        if not (0.2 * w < cx < 0.8 * w and 0.2 * h < cy < 0.8 * h):
            continue
        if ch < 0.25 * h or ch > 0.95 * h:
            continue
        area = cv2.contourArea(cnt)
        if area > best_area:
            best, best_area = cnt, area
    if best is None:
        return None

    mask = np.zeros_like(cell)
    cv2.drawContours(mask, [best], -1, 255, -1)
    return cv2.bitwise_and(cell, mask)


def read_board(warped_bgr):
    """OCR the flattened board. Returns (board 9x9 list, mean confidence)."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    board = [[0] * 9 for _ in range(9)]
    confidences = []
    for r in range(9):
        for c in range(9):
            glyph = extract_cell_glyph(thresh, r, c)
            if glyph is None:
                continue
            digit, conf = classify_digit(glyph)
            if digit and conf > 0.35:
                board[r][c] = digit
                confidences.append(conf)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return board, mean_conf


# ---------------------------------------------------------------------------
# Solving + overlay
# ---------------------------------------------------------------------------

def board_is_consistent(board):
    """True if no given digit conflicts with another (solver's own rules)."""
    for r in range(9):
        for c in range(9):
            digit = board[r][c]
            if digit == 0:
                continue
            board[r][c] = 0
            ok = check_if_valid(board, digit, (r, c))
            board[r][c] = digit
            if not ok:
                return False
    return True


def try_solve(board):
    """Solve a copy of the board. Returns the solved grid or None."""
    if sum(cell != 0 for row in board for cell in row) < 17:
        return None  # too few givens to be a real (unique) puzzle
    if not board_is_consistent(board):
        return None
    solved = copy.deepcopy(board)
    return solved if solve(solved) else None


def render_solution_overlay(givens, solved):
    """Draw only the filled-in digits on a transparent (black) canvas the
    size of the warped board."""
    canvas = np.zeros((WARP, WARP, 3), dtype=np.uint8)
    for r in range(9):
        for c in range(9):
            if givens[r][c] != 0:
                continue
            text = str(solved[r][c])
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            x = c * CELL + (CELL - tw) // 2
            y = r * CELL + (CELL + th) // 2
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (80, 255, 80), 2, cv2.LINE_AA)
    return canvas


def project_overlay(frame, overlay, matrix):
    """Inverse-warp the flat overlay back into the camera frame."""
    h, w = frame.shape[:2]
    back = cv2.warpPerspective(overlay, matrix, (w, h),
                               flags=cv2.WARP_INVERSE_MAP)
    mask = back.max(axis=2) > 0
    out = frame.copy()
    out[mask] = cv2.addWeighted(frame, 0.25, back, 0.75, 0)[mask]
    return out


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def process_frame(frame, cached):
    """Detect, read, solve and overlay. `cached` carries the last solved
    puzzle so we don't re-OCR every frame. Returns (display, cached)."""
    display = frame.copy()
    thresh = preprocess(frame)
    corners = find_board_corners(thresh)
    if corners is None:
        cv2.putText(display, "Looking for a sudoku board...", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return display, cached

    cv2.polylines(display, [corners.astype(int)], True, (0, 200, 255), 2)
    warped, matrix = warp_board(frame, corners)

    if cached is None:
        board, conf = read_board(warped)
        solved = try_solve(board)
        if solved is not None:
            cached = (board, solved)
            print(f"Board locked (OCR confidence {conf:.2f}). "
                  "Press 'r' to re-scan.")
        else:
            cv2.putText(display, "Board found - reading digits...", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return display, cached

    givens, solved = cached
    overlay = render_solution_overlay(givens, solved)
    display = project_overlay(display, overlay, matrix)
    cv2.putText(display, "Solved  (r = re-scan, q = quit)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
    return display, cached


def run_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {index}. On macOS, grant camera access "
            "to your terminal in System Settings > Privacy & Security.")
    print("Camera open. Hold the sudoku so the whole grid is visible.")
    cached, snap = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display, cached = process_frame(frame, cached)
        cv2.imshow("Sudoku Solver", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cached = None
            print("Re-scanning...")
        elif key == ord('s'):
            snap += 1
            path = f"snapshot_{snap}.png"
            cv2.imwrite(path, display)
            print(f"Saved {path}")
    cap.release()
    cv2.destroyAllWindows()


def run_image(path, show=True):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    display, cached = process_frame(frame, None)
    if cached is None:
        print("No solvable board detected in the image.")
    else:
        givens, solved = cached
        print("Detected board (0 = empty):")
        for row in givens:
            print(" ", row)
        print("Solved board:")
        for row in solved:
            print(" ", row)
    out = os.path.splitext(path)[0] + "_solved.png"
    cv2.imwrite(out, display)
    print(f"Wrote {out}")
    if show:
        cv2.imshow("Sudoku Solver", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index (default 0)")
    parser.add_argument("--image", help="solve a still image instead")
    parser.add_argument("--no-show", action="store_true",
                        help="with --image: don't open a window")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, show=not args.no_show)
    else:
        run_camera(args.camera)
