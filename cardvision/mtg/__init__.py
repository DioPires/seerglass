# SPDX-License-Identifier: Apache-2.0
"""Magic: The Gathering, for Seerglass: a catalog from Scryfall's bulk data, the card images,
and the measurements that choose a bank's scope.

Nothing in the rest of the package imports this subpackage, and it changes nothing there: it
writes the files the package reads (a catalog, card images) and measures the package's own
recognizer and bank format at Magic's scale. Run it as ``python -m cardvision.mtg COMMAND``;
``games/mtg/README.md`` lists the commands in the order they are run.

The modules:

``scryfall``
    Reading Scryfall: the bulk-data file, and polite HTTP for the image host.
``catalog``
    The catalog builder: one entry per printed face with an image, compact ``set:cn`` ids, the
    ``"printing"`` identity rule with one representative per card, and the list of what was
    left out and why.
``images``
    The image fetcher: ``normal`` JPEGs into a folder outside the repository, with a state file
    that remembers what was fetched and keeps tombstones for cards that left the catalog.
``looks``
    Look groups: printings that share an illustration and a frame treatment, and so cannot be
    told apart by their picture; statistics per format.
``crowding``
    Embedding every printing with a recognizer, and the crowding curve: how naming degrades as
    the bank grows.
``scorer``
    Bank scoring as one ONNX graph (matrix product, top rows, best row of another identity),
    for a GPU.
"""
