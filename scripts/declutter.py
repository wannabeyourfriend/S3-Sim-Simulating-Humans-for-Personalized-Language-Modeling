"""Strip comments + collapse blank lines + run a formatter on Python files.

Pass any combination of file paths and directory paths. Directories are
walked recursively, picking up every `.py` not under a `__pycache__` /
`.venv` / `.git` directory.

Pipeline per file (in order):
  1. Tokenize and drop every `#`-prefixed comment token. Docstrings and
     string contents are untouched.
  2. Collapse three-or-more consecutive newlines down to two.
  3. Run an external formatter (default `ruff format`) on the result.
     Override or disable with `--formatter "<cmd>"` / `--no-format`.

Examples
--------

    # whole package, in place
    python scripts/declutter.py user_simulator/

    # one file, dry-run (prints to stdout, leaves the file alone)
    python scripts/declutter.py --dry-run user_simulator/oracle.py

    # use black instead of ruff
    python scripts/declutter.py --formatter "black -q -" tests/
"""

from __future__ import annotations

import argparse
import io
import logging
import shlex
import subprocess
import sys
import token as token_mod
import tokenize
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_SKIP_DIRS = {
    "__pycache__",
    ".venv",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
}


def _strip_comments(src: str) -> str:
    """Return `src` with every `#` comment token removed.

    Uses `tokenize.tokenize` + `tokenize.untokenize` so string literals
    (docstrings, f-strings, triple-quoted blocks) are preserved.
    """
    out: list[tokenize.TokenInfo] = []
    reader = io.BytesIO(src.encode("utf-8")).readline
    for tok in tokenize.tokenize(reader):
        if tok.type == token_mod.COMMENT:
            continue
        out.append(tok)
    return tokenize.untokenize(out).decode("utf-8")


def _collapse_blank_lines(src: str) -> str:
    """Collapse 3+ consecutive newlines down to exactly 2."""
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("\n")
        else:
            blank_run = 0
            out.append(ln)
    return "".join(out)


def _run_formatter(src: str, cmd: str) -> str:
    """Pipe `src` through `cmd` (read stdin, write stdout). Returns
    formatter output verbatim; raises if the formatter exits non-zero."""
    parts = shlex.split(cmd)
    proc = subprocess.run(
        parts,
        input=src,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"formatter {cmd!r} exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def _process_file(path: Path, formatter: str | None, dry_run: bool) -> tuple[bool, str]:
    """Apply the pipeline to one file. Returns (changed, error_or_empty)."""
    try:
        original = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return False, f"read failed: {e}"

    try:
        stripped = _strip_comments(original)
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        return False, f"tokenize failed: {e}"

    collapsed = _collapse_blank_lines(stripped)

    if formatter:
        try:
            final = _run_formatter(collapsed, formatter)
        except (RuntimeError, FileNotFoundError) as e:
            return False, f"format failed: {e}"
    else:
        final = collapsed

    if final == original:
        return False, ""

    if dry_run:
        sys.stdout.write(final)
    else:
        path.write_text(final, encoding="utf-8")
    return True, ""


def _iter_py_files(roots: list[Path]):
    for r in roots:
        if r.is_file() and r.suffix == ".py":
            yield r
            continue
        if r.is_dir():
            for p in sorted(r.rglob("*.py")):
                if any(part in _SKIP_DIRS for part in p.parts):
                    continue
                yield p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="File or directory paths to process")
    parser.add_argument(
        "--formatter",
        default="ruff format -q -",
        help="External formatter to pipe through (stdin → stdout). Default: 'ruff format -q -'",
    )
    parser.add_argument("--no-format", action="store_true", help="Skip the formatter step")
    parser.add_argument(
        "--dry-run", action="store_true", help="Write to stdout, leave files untouched"
    )
    args = parser.parse_args(argv)

    formatter = None if args.no_format else args.formatter

    n_changed = 0
    n_failed = 0
    n_total = 0
    for path in _iter_py_files(args.paths):
        n_total += 1
        changed, err = _process_file(path, formatter, args.dry_run)
        if err:
            n_failed += 1
            log.warning("FAIL  %s — %s", path, err)
        elif changed:
            n_changed += 1
            log.info("WRITE %s", path)
        else:
            log.debug("OK    %s", path)

    log.info("\n%d files scanned · %d changed · %d failed", n_total, n_changed, n_failed)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
