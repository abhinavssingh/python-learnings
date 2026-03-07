# 🛡️ Adventure: The Hidden Relic

A feature-rich, text-based adventure game written in Python. This isn't just a "choose your path" script; it features a custom interactive menu system and cross-platform keyboard support.

## 🚀 Quick Start

1.  **Ensure you have Python 3.6+ installed.**
2.  **Run the game:**
    ```bash
    python adventure_game.py
    ```
3.  **(Optional) Enable Logging:** To save your adventure's transcript to a file:
    ```bash
    python adventure_game.py --log session.txt
    ```

## 🎮 How to Play

The game uses an interactive selector. 

| Input | Action |
| :--- | :--- |
| **UP / DOWN Arrows** | Move the selection cursor (`>`) |
| **ENTER** | Confirm your choice |
| **Letter Keys** | Jump to an option starting with that letter |
| **Manual Input** | If arrow keys aren't supported, type the keyword (e.g., "river") |

## 🛠️ Technical Deep Dive

### 1. Platform-Specific I/O
The script detects the Operating System using `os.name`. 
* **Windows:** Uses `msvcrt.getch()` to capture "raw" keystrokes without requiring the user to press Enter.
* **Linux/macOS:** Uses `termios` and `tty` to put the terminal into "raw mode" and read standard input characters.

### 2. The `Tee` Class
A clever implementation of `io.TextIOBase` that replaces `sys.stdout`. When the `--log` flag is provided, every `print()` statement is sent to both the screen and the specified log file, ensuring `isatty()` checks still pass for terminal detection.

### 3. Screen Refresh Logic
To prevent the terminal from becoming a wall of repetitive text, the `clear_lines` function uses **ANSI Escape Codes**:
* `\x1b[1A`: Moves the cursor up one line.
* `\x1b[2K`: Erases the current line.

## 🌲 Game Paths

> [!TIP]
> **Forest Path:** High risk, high reward. Focus on the river.
> **Cave Path:** Requires light. Don't go chasing glitter!