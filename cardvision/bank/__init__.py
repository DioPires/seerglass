# SPDX-License-Identifier: Apache-2.0
"""Build the embedding bank and bind a release.

* ``views``: the views synthesized from one card image, which a bank row averages over.
* ``build``: card images to a bank file and its identity groups, from scratch or by extending
  an existing bank.
* ``refresh``: the check that a new bank only adds to an old one, apart from rows embedded
  again because their image changed.
* ``art``: the comparison of the image each row records with the image a card has now.
* ``release``: one document that pins every file and setting recognition runs with, and the
  check that the pinned files are present, unchanged and consistent with each other.
"""
