# SPDX-License-Identifier: Apache-2.0
"""The Arena log reader (``cardvision.mtg.arena_log``), on a made-up log in Arena's format: the
zones it reports, the seats, the times, and that nothing personal comes through."""

from __future__ import annotations

import json

from cardvision.mtg import arena_log as L

TICKS = 638_651_000_000_000_000  # .NET ticks, as Arena writes some timestamps


def _event(state, seat=1, timestamp=None):
    message = {"type": "GREMessageType_GameStateMessage", "systemSeatIds": [seat], "gameStateMessage": state}
    event = {"transactionId": "x", "greToClientEvent": {"greToClientMessages": [message]}}
    if timestamp is not None:
        event["timestamp"] = str(timestamp)
    return event


def _log():
    full = {
        "type": "GameStateType_Full",
        "gameInfo": {"matchID": "match-secret", "gameNumber": 1, "stage": "GameStage_Play"},
        "zones": [
            {"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [101, 102]},
            {"zoneId": 31, "type": "ZoneType_Hand", "ownerSeatId": 1, "objectInstanceIds": [103]},
            {"zoneId": 35, "type": "ZoneType_Hand", "ownerSeatId": 2, "objectInstanceIds": [201, 202]},
            {"zoneId": 32, "type": "ZoneType_Library", "objectInstanceIds": [104]},
        ],
        "gameObjects": [
            {"instanceId": 101, "grpId": 87521, "type": "GameObjectType_Card", "zoneId": 28, "controllerSeatId": 1},
            {"instanceId": 102, "grpId": 90001, "type": "GameObjectType_Token", "zoneId": 28, "controllerSeatId": 2,
             "isTapped": True},
            {"instanceId": 103, "grpId": 77777, "type": "GameObjectType_Card", "zoneId": 31, "controllerSeatId": 1},
            {"instanceId": 104, "grpId": 55555, "type": "GameObjectType_Card", "zoneId": 32, "controllerSeatId": 1},
        ],
        "players": [{"systemSeatNumber": 1, "playerName": "SomePlayer#12345", "userId": "ABCDEF"}],
    }
    diff = {
        "type": "GameStateType_Diff",
        "zones": [{"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [101, 105]},
                  {"zoneId": 27, "type": "ZoneType_Stack", "objectInstanceIds": [106]}],
        "gameObjects": [
            {"instanceId": 105, "grpId": 77777, "type": "GameObjectType_Card", "zoneId": 28, "controllerSeatId": 1},
            {"instanceId": 106, "grpId": 1234, "objectSourceGrpId": 87521, "type": "GameObjectType_Ability",
             "zoneId": 27, "controllerSeatId": 1},
        ],
        "diffDeletedInstanceIds": [102],
    }
    second_game = {**full, "gameInfo": {"matchID": "match-secret", "gameNumber": 2}}
    lines = [
        "[UnityCrossThreadLogger]Client Version: 2026.55.10.1234",
        "[UnityCrossThreadLogger]10/21/2026 7:12:33 PM: Match to SomePlayer#12345: GreToClientEvent",
        json.dumps(_event(full, timestamp=TICKS), indent=2),  # spread over lines, as Arena sometimes writes it
        "[UnityCrossThreadLogger]10/21/2026 7:12:40 PM: Match to SomePlayer#12345: GreToClientEvent",
        json.dumps(_event(diff, timestamp=1792605160000)),
        json.dumps(_event(diff)),  # the same state again: no new row
        "some {unbalanced text from another logger",
        json.dumps({"chat": "hello SomePlayer#12345"}),
        json.dumps(_event(second_game)),
    ]
    return "\n".join(lines) + "\n"


def test_zones_seats_and_times():
    table = L.read_log(_log())
    assert table["schema"] == L.SCHEMA and table["client"] == "2026.55.10.1234"
    assert [game["game"] for game in table["games"]] == [1, 2]
    first, second = table["games"][0]["states"]
    assert first["zones"]["battlefield"] == [[87521, "you", False, "card"], [90001, "opponent", True, "token"]]
    assert first["zones"]["hand"] == [[77777, "you", False, "card"]]
    assert "library" not in first["zones"]
    assert first["t"] == (TICKS - L.DOTNET_EPOCH_TICKS) // 10_000
    assert first["logTime"] == "10/21/2026 7:12:33 PM"
    assert second["zones"]["battlefield"] == [[87521, "you", False, "card"], [77777, "you", False, "card"]]
    assert second["zones"]["stack"] == [[87521, "you", False, "ability"]]
    assert second["t"] == 1792605160000
    assert len(table["games"][0]["states"]) == 2  # the repeated diff added nothing


def test_the_opponents_hidden_hand_is_counted_not_named():
    first = L.read_log(_log())["games"][0]["states"][0]
    assert first["hidden"] == {"hand": 2}


def test_nothing_personal_comes_through():
    text = json.dumps(L.read_log(_log()))
    for secret in ("SomePlayer", "ABCDEF", "match-secret", "hello"):
        assert secret not in text


def test_candidate_list_and_command_line(tmp_path):
    log = tmp_path / "Player.log"
    log.write_text(_log(), encoding="utf-8")
    out = tmp_path / "game.json"
    assert L.main([str(log), "--out", str(out)]) == 0
    table = json.loads(out.read_text(encoding="utf-8"))
    assert L.arena_ids(table) == [77777, 87521, 90001]


def test_a_log_without_games(tmp_path):
    log = tmp_path / "Player.log"
    log.write_text("[UnityCrossThreadLogger]nothing here\n", encoding="utf-8")
    assert L.main([str(log), "--out", str(tmp_path / "out.json")]) == 1


def test_timestamps():
    assert L.epoch_ms("1792605160000") == 1792605160000
    assert L.epoch_ms(L.DOTNET_EPOCH_TICKS) == 0
    assert L.epoch_ms("not a time") is None and L.epoch_ms(42) is None
