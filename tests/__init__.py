"""Test package.

The Flask app runs with web_app/ as its import root, so its modules import each
other flatly (``import pdf_text_extractor``, ``from sentence_ids import ...``).
Tests import them as ``web_app.<module>``, so web_app/ has to be on sys.path for
those intra-module imports to resolve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web_app"))
