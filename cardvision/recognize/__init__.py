# SPDX-License-Identifier: Apache-2.0
"""The card recognizer: crops, looks, the accept rule and a runner.

``crops`` cuts a box out of a frame, ``looks`` scores an embedding against the bank, ``accept``
decides whether a card is named and in which orientation, and ``Recognizer`` does all of it for
a list of crops.
"""

from cardvision.recognize.accept import (
    ACCEPT_KEYS,
    DEFAULT_DELTA,
    DEFAULT_TAU,
    NAMED,
    NOTHING,
    AcceptError,
    AcceptRule,
    Choice,
    accept_errors,
    choose_among,
    choose_look,
    load_accept,
    parse_accept,
)
from cardvision.recognize.crops import crop_edges, crop_size, crop_truncated, large_enough
from cardvision.recognize.looks import (
    NO_OTHER_IDENTITY,
    Look,
    bank_transposed,
    best_identities,
    best_rows,
    identity_columns,
    name_columns,
    score_look,
    similarities,
    top_and_other,
)
from cardvision.recognize.recognizer import Candidate, Recognition, Recognizer, declared_contract

__all__ = [
    "ACCEPT_KEYS",
    "DEFAULT_DELTA",
    "DEFAULT_TAU",
    "NAMED",
    "NOTHING",
    "NO_OTHER_IDENTITY",
    "AcceptError",
    "AcceptRule",
    "Candidate",
    "Choice",
    "Look",
    "Recognition",
    "Recognizer",
    "accept_errors",
    "bank_transposed",
    "best_identities",
    "best_rows",
    "choose_among",
    "choose_look",
    "crop_edges",
    "crop_size",
    "crop_truncated",
    "declared_contract",
    "identity_columns",
    "large_enough",
    "load_accept",
    "name_columns",
    "parse_accept",
    "score_look",
    "similarities",
    "top_and_other",
]
