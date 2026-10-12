"""Bounded content detection before binary artifacts become text tool output."""

import codecs
from pathlib import Path

from fast_agent.core.exceptions import RecoverableToolError

_BINARY_HEADERS = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"%PDF-",
    b"PK\x03\x04",
    b"ID3",
    b"RIFF",
)


def require_text_file(path: Path) -> None:
    if not path.is_file():
        return  # Let the MCP report its normal missing-file/permission error.
    with path.open("rb") as source:
        sample = source.read(4096)
    binary = b"\x00" in sample or sample.startswith(_BINARY_HEADERS)
    if not binary:
        try:
            # A sample may end inside a valid UTF-8 character.
            codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        except UnicodeDecodeError:
            binary = True
    if binary:
        raise RecoverableToolError(
            "Binary artifact cannot be read as text. Use read_media_file for supported "
            "images/audio; use a suitable document/archive tool for PDF/ZIP. "
            "A file hash or size can be inspected without loading its content."
        )
