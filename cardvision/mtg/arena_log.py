# SPDX-License-Identifier: Apache-2.0
"""Read an MTG Arena ``Player.log`` into a table of the cards in play, with nothing personal in it.

It runs on the player's own computer, with Python 3.10 or later and nothing else installed::

    python cardvision/mtg/arena_log.py Player.log --out game-1.json

``python -m cardvision.mtg arena-log`` runs the same. Arena replaces ``Player.log`` each time it
starts (keeping one ``Player-prev.log``), so copy it after each recorded session.

What it reads
-------------
With *Detailed Logs (Plugin Support)* on (Arena's Options, Account), Arena writes the messages its
game server sends (``greToClientEvent``) into the log, as JSON. Each ``gameStateMessage`` holds
the game's objects (``gameObjects``: ``instanceId``, ``grpId``, which is the card's Arena id and
Scryfall's ``arena_id``, ``zoneId``, ``controllerSeatId``, ``isTapped``, ``type``) and zones
(``zones``: ``zoneId``, ``type``, ``objectInstanceIds``), whole (``GameStateType_Full``) or as the
changes since the last message (``GameStateType_Diff``, with ``diffDeletedInstanceIds``). The
log holds no screen positions: it says what is in each zone, not where on the screen it is.

What it writes
--------------
For each game, every moment at which what is in view changed, and what each zone held then::

    {"schema": "seerglass-arena-log/1", "client": "...", "games": [
      {"game": 1, "states": [
        {"t": 1792605153000, "logTime": "10/21/2026 7:12:33 PM",
         "zones": {"battlefield": [[87521, "you", false, "card"], ...], "hand": [...], ...},
         "hidden": {"hand": 7}}, ...]}]}

``t`` is the server's time in milliseconds since 1970 when the message carries one, and
``logTime`` the time Arena wrote on the log line before it, as written (its format follows the
computer's locale). A row is ``[grpId, seat, tapped, type]``: the seat is ``"you"`` or
``"opponent"``, the type ``card``, ``token``, ``ability`` (an ability on the stack, given by the
Arena id of the card it comes from) or ``emblem``. ``hidden`` counts the objects of a zone whose
card is not shown to you, such as the opponent's hand.

What it leaves out
------------------
Player names, account, match and session ids, chat, deck lists, file paths, and every hidden
card: the output is built field by field, and nothing is copied over whole. The raw log stays on
the player's computer; only this table is shared.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA = "seerglass-arena-log/1"
# Arena's zone types, and the name each kept zone has in the output. The library and Arena's
# internal zones (limbo, pending, suppressed) are not shown on screen and are left out.
ZONES = {
    "ZoneType_Battlefield": "battlefield",
    "ZoneType_Hand": "hand",
    "ZoneType_Stack": "stack",
    "ZoneType_Graveyard": "graveyard",
    "ZoneType_Exile": "exile",
    "ZoneType_Command": "command",
    "ZoneType_Revealed": "revealed",
}
OBJECT_TYPES = {
    "GameObjectType_Card": "card",
    "GameObjectType_Token": "token",
    "GameObjectType_Ability": "ability",
    "GameObjectType_Emblem": "emblem",
}
HEADER = re.compile(r"^\[UnityCrossThreadLogger\]\s*(\d[\d/.\- :]*\d(?:\s?[AaPp]\.?[Mm]\.?)?)")
CLIENT = re.compile(r'"clientVersion"\s*:\s*"([^"]{1,40})"|Client\s*Version:\s*([0-9][0-9.]{0,30})', re.IGNORECASE)
DOTNET_EPOCH_TICKS = 621_355_968_000_000_000  # 1970-01-01 in .NET ticks (100 ns since year 1)


def epoch_ms(value: Any) -> int | None:
    """A message's ``timestamp`` as milliseconds since 1970: Arena writes either that or .NET
    ticks, as a string or a number. ``None`` when it is neither."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number > 10**16:
        return (number - DOTNET_EPOCH_TICKS) // 10_000
    if 10**11 <= number < 10**14:
        return number
    return None


def json_objects(text: str) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Every JSON object the log holds that starts on a line, with the time of the last log
    header before it (``None`` before the first)."""
    decoder = json.JSONDecoder()
    log_time: str | None = None
    position = 0
    length = len(text)
    while position < length:
        end = text.find("\n", position)
        end = length if end < 0 else end
        line = text[position:end]
        header = HEADER.match(line)
        if header:
            log_time = header.group(1).strip()
        brace = line.find("{")
        if brace >= 0:
            try:
                value, stop = decoder.raw_decode(text, position + brace)
            except ValueError:
                value, stop = None, None
            if isinstance(value, dict):
                yield value, log_time
                next_line = text.find("\n", stop)
                position = length if next_line < 0 else next_line + 1
                continue
        position = end + 1


def gre_messages(value: Any) -> Iterator[tuple[dict[str, Any], int | None]]:
    """The ``greToClientMessages`` inside a decoded log object, with the object's time."""
    if not isinstance(value, dict):
        return
    event = value.get("greToClientEvent")
    if isinstance(event, dict):
        t = epoch_ms(value.get("timestamp"))
        for message in event.get("greToClientMessages") or ():
            if isinstance(message, dict):
                yield message, t
        return
    for child in value.values():  # some Arena versions wrap the event in another object
        if isinstance(child, dict):
            yield from gre_messages(child)


class _Game:
    def __init__(self, number: int, you: int | None) -> None:
        self.number = number
        self.you = you
        self.objects: dict[int, dict[str, Any]] = {}
        self.zones: dict[int, dict[str, Any]] = {}
        self.states: list[dict[str, Any]] = []
        self.last: Any = None

    def apply(self, state: dict[str, Any]) -> None:
        if state.get("type") == "GameStateType_Full":
            self.objects, self.zones = {}, {}
        for zone in state.get("zones") or ():
            if isinstance(zone, dict) and isinstance(zone.get("zoneId"), int):
                self.zones[zone["zoneId"]] = zone
        for obj in state.get("gameObjects") or ():
            if isinstance(obj, dict) and isinstance(obj.get("instanceId"), int):
                self.objects[obj["instanceId"]] = obj
        for gone in state.get("diffDeletedInstanceIds") or ():
            self.objects.pop(gone, None)

    def seat(self, number: Any) -> str:
        return "you" if self.you is not None and number == self.you else "opponent"

    def snapshot(self) -> tuple[dict[str, list[list[Any]]], dict[str, int]]:
        zones: dict[str, list[list[Any]]] = {}
        hidden: dict[str, int] = {}
        for zone in sorted(self.zones.values(), key=lambda z: z["zoneId"]):
            name = ZONES.get(zone.get("type"))
            if name is None:
                continue
            for instance in zone.get("objectInstanceIds") or ():
                obj = self.objects.get(instance)
                kind = OBJECT_TYPES.get(obj.get("type")) if obj else None
                card = None
                if obj and kind is not None:
                    card = obj.get("objectSourceGrpId") if kind == "ability" else obj.get("grpId")
                if not isinstance(card, int) or card <= 0 or (obj or {}).get("isFacedown"):
                    hidden[name] = hidden.get(name, 0) + 1
                    continue
                zones.setdefault(name, []).append(
                    [card, self.seat(obj.get("controllerSeatId", obj.get("ownerSeatId"))), bool(obj.get("isTapped")),
                     kind])
        return zones, hidden

    def record(self, t: int | None, log_time: str | None) -> None:
        zones, hidden = self.snapshot()
        if (zones, hidden) == self.last:
            return
        self.last = (zones, hidden)
        state: dict[str, Any] = {"t": t, "logTime": log_time, "zones": zones}
        if hidden:
            state["hidden"] = hidden
        self.states.append(state)


def read_log(text: str) -> dict[str, Any]:
    """The table of a ``Player.log``'s text (see the module's docstring)."""
    client = CLIENT.search(text[:200_000])
    games: list[_Game] = []
    current: _Game | None = None
    current_key: Any = None
    for value, log_time in json_objects(text):
        for message, t in gre_messages(value):
            if message.get("type") != "GREMessageType_GameStateMessage":
                continue
            state = message.get("gameStateMessage")
            if not isinstance(state, dict):
                continue
            info = state.get("gameInfo") or {}
            key = (info.get("matchID"), info.get("gameNumber")) if info.get("matchID") else current_key
            seats = message.get("systemSeatIds") or ()
            you = seats[0] if seats and isinstance(seats[0], int) else None
            if current is None or (key is not None and key != current_key):
                current = _Game(len(games) + 1, you)
                games.append(current)
                current_key = key
            if current.you is None:
                current.you = you
            current.apply(state)
            current.record(t, log_time)
    return {
        "schema": SCHEMA,
        "client": next((group for group in (client.groups() if client else ()) if group), None),
        "games": [{"game": game.number, "states": game.states} for game in games if game.states],
    }


def arena_ids(table: dict[str, Any]) -> list[int]:
    """Every card id the table names, once each: a recording's candidate list."""
    seen: set[int] = set()
    for game in table.get("games", ()):
        for state in game.get("states", ()):
            for rows in state.get("zones", {}).values():
                seen.update(row[0] for row in rows)
    return sorted(seen)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read an MTG Arena Player.log into a table of the cards in play "
                                                 "(card ids, zones, seats and times only).")
    parser.add_argument("log", help="a Player.log (or Player-prev.log) copied after the session")
    parser.add_argument("--out", required=True, help="the table, JSON")
    args = parser.parse_args(None if argv is None else list(argv))
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    table = read_log(text)
    out = Path(args.out)
    out.write_text(json.dumps(table, separators=(",", ":")) + "\n", encoding="utf-8")
    states = sum(len(game["states"]) for game in table["games"])
    print(f"{out}: {len(table['games'])} games, {states} states, {len(arena_ids(table))} different cards")
    if not table["games"]:
        print("No game found: is Detailed Logs (Plugin Support) on in Arena's Options, Account?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
