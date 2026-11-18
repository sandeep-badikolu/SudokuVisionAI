"""Failures while preparing or matching a digit.

These are not solver failures and they are not grid-detection failures.
"""


class RecognitionError(Exception):
    """A cell image or a template set cannot be used for recognition."""
