# SPDX-License-Identifier: Apache-2.0
"""Write a file so that it is either whole or untouched.

A file that is written in place is empty from the moment it is opened until the last byte is
out. A run that is stopped in between, or a disk that fills up, leaves half a file where the
whole one was. For a file that may be somebody's only copy of their labels that is not good
enough, so such files are written beside themselves and then moved into place.
"""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from pathlib import Path

# A file that is being written has this suffix, so that nothing takes it for the finished file.
PARTIAL_SUFFIX = ".partial"


def _new_file_mode() -> int:
    """The permissions a newly created file gets here."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` in one step.

    The text goes to a temporary file in the same folder, and that file then takes the place
    of ``path``. A write that fails or is interrupted leaves the file that was there exactly as
    it was, and leaves no temporary file behind when the failure can be caught. A file that is
    replaced keeps its permissions; a new file gets the permissions any new file gets. A
    symbolic link is written through, as a plain write would do, and a file that may not be
    written to is refused with ``PermissionError``, as a plain write would refuse it: moving a
    new file into its place would otherwise get round its protection.
    """
    path = Path(os.path.realpath(path))
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = _new_file_mode()
    else:
        if not os.access(path, os.W_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=PARTIAL_SUFFIX)
    try:
        with os.fdopen(handle, "w", encoding=encoding) as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


__all__ = ["PARTIAL_SUFFIX", "write_text_atomic"]
