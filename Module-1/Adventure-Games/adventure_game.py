"""
adventure_game.py
-------------------
A simple text-based adventure game that demonstrates Python fundamentals:
variables, input handling, conditionals, loops, and functions.

Now enhanced with keyboard navigation: use UP/DOWN arrows and Enter
to pick options (falls back to text input if the environment doesn't
support raw key reads).
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from typing import Dict

#=========logging============================

def _maybe_enable_logging():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log", metavar="FILE", help="Capture all game output to FILE")
    args, _ = parser.parse_known_args()

    if not args.log:
        return  # nothing to do

    console = sys.stdout                   # the real console stream
    logf = open(args.log, "w", encoding="utf-8")

    class Tee(io.TextIOBase):
        """A stdout replacement that mirrors writes to console and a file,
        while still quacking like the original console (TTY)."""
        def __init__(self, console_stream, file_stream):
            self.console = console_stream
            self.file = file_stream
        # --- essential stream behavior ---
        def write(self, s):
            self.console.write(s); self.console.flush()
            self.file.write(s);    self.file.flush()
            return len(s)
        def flush(self):
            try: self.console.flush()
            finally: self.file.flush()
        def writable(self): return True
        # --- make this look like a real terminal ---
        def isatty(self):
            return getattr(self.console, "isatty", lambda: False)()
        def fileno(self):
            # Some TTY detection paths call fileno(); delegate to console.
            return getattr(self.console, "fileno")()
        # --- optional: pass-through encoding/errors if queried ---
        @property
        def encoding(self): return getattr(self.console, "encoding", "utf-8")
    sys.stdout = Tee(console, logf)  # replace stdout with TTY-like tee

    return

_maybe_enable_logging()


# ==== Keyboard helpers ======================================================

def _is_windows() -> bool:
    return os.name == 'nt'


def _read_key() -> str:
    """Read a single keypress from the terminal, returning a semantic token.

    Returns one of: 'UP', 'DOWN', 'ENTER', or a single character.
    Works on Windows (msvcrt) and POSIX (termios/tty). If raw mode isn't
    available, raises RuntimeError.
    """
    if _is_windows():
        try:
            import msvcrt  # type: ignore
        except Exception as e:
            raise RuntimeError("msvcrt not available") from e
        ch = msvcrt.getch()
        # Handle special keys
        if ch in (b'\r', b'\n'):
            return 'ENTER'
        if ch in (b'\x00', b'\xe0'):  # special prefix
            nxt = msvcrt.getch()
            code = nxt
            if code == b'H':
                return 'UP'
            if code == b'P':
                return 'DOWN'
            return ''
        try:
            return ch.decode('utf-8', errors='ignore')
        except Exception:
            return ''
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd) # type: ignore[attr-defined, reportAttributeAccessIssue]
        try:
            tty.setraw(fd) # type: ignore[attr-defined, reportAttributeAccessIssue]
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # escape sequence
                seq = sys.stdin.read(2)
                if seq == '[A':
                    return 'UP'
                if seq == '[B':
                    return 'DOWN'
                return ''
            if ch in ('\r', '\n'):
                return 'ENTER'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old) # type: ignore[attr-defined, reportAttributeAccessIssue]


def _supports_raw_keys() -> bool:
    # Basic heuristic: TTY plus platform modules importable
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    if _is_windows():
        try:
            import msvcrt  # noqa: F401
            return True
        except Exception:
            return False
    else:
        try:
            import termios  # noqa: F401
            import tty
            return True
        except Exception:
            return False


def select_with_arrows(prompt: str, options: Dict[str, str]) -> str:
    """Interactive selector using UP/DOWN + Enter. Returns the chosen key.

    Falls back to text input if raw key capture isn't supported.
    """
    if not _supports_raw_keys():
        # Fallback to simple input prompt
        return prompt_choice_text(prompt, options)

    keys = list(options.keys())
    idx = 0

    def render():
        os.system('')  # enable ANSI on newer Windows terminals
        sys.stdout.write('\n' + prompt + '\n')
        for i, k in enumerate(keys):
            prefix = '> ' if i == idx else '  '
            line = f"{prefix}{options[k]}  (key: {k})"
            sys.stdout.write(line + '\n')
        sys.stdout.write("\nUse UP/DOWN and Enter. Press letter/number keys to jump.\n")
        sys.stdout.flush()

    # Clear between renders by printing carriage returns and re-rendering
    # We keep it simple to avoid OS-specific clear calls.
    def clear_lines(count: int):
        for _ in range(count):
            sys.stdout.write('\x1b[1A\x1b[2K')  # up one line, erase line
        sys.stdout.flush()

    # Initial paint
    render()
    lines_rendered = len(options) + 3  # prompt + items + hint

    try:
        while True:
            key = _read_key()
            if key == 'UP':
                idx = (idx - 1) % len(keys)
            elif key == 'DOWN':
                idx = (idx + 1) % len(keys)
            elif key == 'ENTER':
                # Clear menu on select
                clear_lines(lines_rendered)
                return keys[idx]
            elif isinstance(key, str) and key:
                # Jump by first-character match if unique
                matches = [i for i, k in enumerate(keys) if k.startswith(key.lower())]
                if len(matches) == 1:
                    clear_lines(lines_rendered)
                    return keys[matches[0]]
            # Re-render
            clear_lines(lines_rendered)
            render()
    except Exception:
        # On any failure, fall back gracefully
        return prompt_choice_text(prompt, options)


# ==== Original prompt (text fallback) =======================================

def prompt_choice_text(prompt: str, options: Dict[str, str]) -> str:
    print(f"\n{prompt}")
    for key, label in options.items():
        print(f"  - {key}: {label}")
    while True:
        choice = input("> Your choice: ").strip().lower()
        if choice not in options:
            matches = [k for k in options if k.startswith(choice)] if choice else []
            if len(matches) == 1:
                return matches[0]
            print("  [!] Invalid choice. Please pick one of:", ", ".join(options.keys()))
            continue
        return choice


# ==== Game logic ============================================================

def forest_path(player: str) -> str:
    """Handle the forest scenario.

    Returns: 'win' | 'lose' | 'continue'
    """
    print(f"\n\U0001F332 The forest is dense and quiet, {player}. You hear water nearby.")
    choice = select_with_arrows(
        "Do you follow the sound of the river or climb a tall tree to scout?",
        {"river": "Follow the river", "tree": "Climb the tree"},
    )
    if choice == "river":
        print("\nYou follow the river and find a rickety wooden bridge with symbols carved into it.")
        sub = select_with_arrows(
            "Cross the bridge or search the riverbank?",
            {"bridge": "Cross the bridge", "bank": "Search the riverbank"},
        )
        if sub == "bridge":
            print("\nYou cross carefully. The symbols mark the path to a hidden glade—inside lies the TREASURE! \U0001F3C6")
            return "win"
        else:
            print("\nYou search along the bank but slip on wet stones and lose your trail. The forest swallows the path…")
            return "lose"
    else:
        print("\nFrom the treetop you spot smoke in the distance—likely a camp with clues. But climbing down, a branch snaps!")
        print("You land safely but your map tears beyond recognition. With no bearings, you wander in circles.")
        return "lose"


def cave_path(player: str) -> str:
    """Handle the cave scenario.

    Returns: 'win' | 'lose' | 'continue'
    """
    print(f"\n\U0001F573\uFE0F  The cave yawns before you, {player}. A chill wind carries faint echoes.")
    choice = select_with_arrows(
        "Do you light a torch or proceed in the dark?",
        {"torch": "Light a torch", "dark": "Proceed in the dark"},
    )
    if choice == "torch":
        print("\nWith the torch lit, you notice markings on the wall guiding you past hidden pits.")
        sub = select_with_arrows(
            "Follow the markings deeper or inspect a glittering crevice?",
            {"markings": "Follow the markings", "crevice": "Inspect the crevice"},
        )
        if sub == "markings":
            print("\nThe markings lead to a sealed chamber. A simple mechanism opens it, revealing the TREASURE! \U0001F3C6")
            return "win"
        else:
            print("\nA rockslide is triggered! You scramble out, but the passage collapses behind you—quest failed.")
            return "lose"
    else:
        print("\nIn the darkness you misstep into a pit. You're unharmed but trapped with no way forward.")
        return "lose"


def start_game() -> None:
    """Start the game and manage the main loop with replay option."""
    print("""
    ===============================
      ADVENTURE: The Hidden Relic
    ===============================
    """)
    player = input("What is your name, explorer? > ").strip() or "Explorer"
    print(f"\nWelcome, {player}! Your quest is to find the legendary treasure hidden in these lands.")

    while True:
        path = select_with_arrows(
            "Choose your path:",
            {"forest": "Explore the dark forest", "cave": "Enter the mysterious cave"},
        )
        outcome = forest_path(player) if path == "forest" else cave_path(player)

        if outcome == "win":
            print("\n\U0001F389 Congratulations! You found the treasure and completed your quest.")
        else:
            print("\n\u2620\uFE0F  Alas! Your choices led to failure this time.")

        again = select_with_arrows(
            "Would you like to play again?",
            {"yes": "Yes, restart" , "no": "No, quit"}
        )
        if again == "no":
            print("\nThanks for playing. Farewell!")
            break
        else:
            print("\nRestarting your adventure...\n")


if __name__ == "__main__":
    start_game()