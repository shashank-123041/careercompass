# resume/ — files to drop in

The upload this restructure was built from contained these two files EMPTY, so
they were not recreated (nothing was invented). `resume_parser.py` degrades
gracefully without them (Gemini only, no dictionary fallback):

- `auto_extract.py`      — dictionary-based fallback extractor; must expose
                           `load_dictionary(path)` and `extract_row(text, dictionary)`
- `skill_dictionary.txt` — controlled skill vocabulary (~568 terms)

Copy your real versions into this folder to re-enable the fallback.
