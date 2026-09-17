"""Rules and codepoint filters for hostile text sanitization."""

import re

# Format characters (zero-width space, joiner, non-joiner, BOM, etc.)
ZERO_WIDTH_CHARS = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Byte order mark / zero-width no-break space
    "\u2060",  # Word joiner
}

# Bidirectional text override codepoints
BIDI_OVERRIDE_CHARS = {
    "\u202a",  # Left-to-Right Embedding
    "\u202b",  # Right-to-Left Embedding
    "\u202c",  # Pop Directional Formatting
    "\u202d",  # Left-to-Right Override
    "\u202e",  # Right-to-Left Override
    "\u2066",  # Left-to-Right Isolate
    "\u2067",  # Right-to-Left Isolate
    "\u2068",  # First Strong Isolate
    "\u2069",  # Pop Directional Isolate
}

# Control character pattern (keep standard whitespace: \t, \n, \r)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
