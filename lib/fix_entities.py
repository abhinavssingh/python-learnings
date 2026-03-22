# fix_entities.py
from pathlib import Path
import html as html_mod

script_directory = Path(__file__).parent.resolve()
print("Script file directory:", script_directory)

p = Path(rf"{script_directory}\arrays_html.py")
src = p.read_text(encoding="utf-8")
fixed = html_mod.unescape(src)  # converts &amp;lt; -> <, &amp;gt; -> >, &amp;amp; -> &
p.write_text(fixed, encoding="utf-8")
print("unescaped:", p)