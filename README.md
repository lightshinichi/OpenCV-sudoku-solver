# Sudoku Solver: A Python Implementation Using Backtracking

This Python script is designed to solve Sudoku puzzles using a backtracking algorithm. It takes a partially filled 9x9 Sudoku board as input and outputs the completed board. The script defines several functions to accomplish this: `solve`, `check_if_valid`, `print_board`, and `find_empty`. Below is a detailed explanation of each component and how to use the script.

## Features

- **Solve any 9x9 Sudoku puzzle**: Given a valid Sudoku puzzle, this script can fill in the missing numbers to complete the puzzle.
- **Backtracking algorithm**: Utilizes a backtracking algorithm to efficiently find the solution.
- **Input validation**: Checks if a number can be legally placed in a given position according to Sudoku rules.
- **Pretty print**: Neatly prints the Sudoku board before and after solving, making it easy to visualize the puzzle and its solution.

## How It Works

### The `solve` Function

This is the main function that attempts to solve the Sudoku puzzle. It works recursively by trying to fill the board with valid numbers from 1 to 9. If it encounters a situation where no valid number can be placed in a position, it backtracks to the previous step and tries a different number.

### The `check_if_valid` Function

This function checks if a given number can be placed at a specified position on the board. It ensures that the number is not already present in the same row, column, or 3x3 subgrid according to Sudoku rules.

### The `print_board` Function

This function prints the Sudoku board in a user-friendly format, with lines separating each 3x3 subgrid for better readability.

### The `find_empty` Function

This function searches the board for an empty spot (denoted by 0) and returns the position (row and column) of the first empty spot found. This position is used by the `solve` function to try placing numbers.

## Usage

1. **Define the Sudoku puzzle**: Input the Sudoku puzzle as a 9x9 grid (list of lists) with zeros representing empty spots. An example puzzle is provided in the script.

2. **Run the script**: Execute the script in a Python environment. The script will first print the original Sudoku board, then attempt to solve the puzzle, and finally print the solved board.

3. **View the solution**: After the script finishes running, the solved Sudoku puzzle will be printed to the console.

## Example

Here's a simple example of how to define a Sudoku puzzle and solve it using the script:

```python
board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]

print_board(board)
solve(board)
print("-------------------------------------")
print("Solved board:")
print_board(board)
