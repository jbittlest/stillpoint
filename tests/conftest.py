# Make the repo root importable so tests can use eval.footage (test-footage locations, env-configurable).
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
