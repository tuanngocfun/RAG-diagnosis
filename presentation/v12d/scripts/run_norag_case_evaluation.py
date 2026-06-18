#!/usr/bin/env python3
"""Deprecated V12d helper.

The demo backend does not expose a reliable no-RAG switch. V12d therefore uses
official experiment-pipeline Gemma 4 no-RAG artifacts via
scripts/extract_gemma4_rag_norag_comparison.py.
"""

from __future__ import annotations


def main() -> int:
    print(
        "This helper is superseded. Use "
        "scripts/extract_gemma4_rag_norag_comparison.py for V12d."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
