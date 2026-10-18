# SPDX-License-Identifier: Apache-2.0
"""TLS for the commands that download: where the CA certificates come from, and what to say
when a server's certificate cannot be verified.

Some Python builds ship no CA certificates of their own; the macOS installer from python.org is
one. On such a build every https request fails with ``CERTIFICATE_VERIFY_FAILED``, whatever the
server, until a bundle of certificates is installed. The ``certifi`` package is such a bundle.
It comes with the ``download`` extra, and the commands that download use it when it is
installed. Verification is never turned off.
"""

from __future__ import annotations

import ssl

# What Python's ssl module, and the tools built on it, write when a certificate cannot be verified.
CERTIFICATE_FAILURE_SIGN = "CERTIFICATE_VERIFY_FAILED"

CERTIFICATE_HELP = (
    "the server's TLS certificate could not be verified (CERTIFICATE_VERIFY_FAILED), so nothing was "
    "downloaded. The usual cause is a Python that ships no CA certificates, such as the macOS installer from "
    'python.org. Install a bundle of them and run the command again: python -m pip install -e ".[download]" '
    "in the repository folder brings the certifi package; python -m pip install certifi does the same alone"
)


class CertificatesMissing(RuntimeError):
    """A download failed because the server's certificate could not be verified. The message
    says what to install."""


def tls_context() -> ssl.SSLContext:
    """The context for an https request: certifi's certificates when that package is installed,
    the Python build's own otherwise. Certificates are always verified."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def is_certificate_failure(error: BaseException | str) -> bool:
    """Whether an error, or an error's text, says that a certificate could not be verified."""
    if isinstance(error, str):
        return CERTIFICATE_FAILURE_SIGN in error
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ssl.SSLCertVerificationError) or CERTIFICATE_FAILURE_SIGN in str(error):
            return True
        error = getattr(error, "reason", None) if isinstance(getattr(error, "reason", None), BaseException) else error.__cause__
    return False
