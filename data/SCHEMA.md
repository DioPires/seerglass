<!-- SPDX-License-Identifier: Apache-2.0 -->

# The label manifest format

Version 1.

A label manifest says, for single frames of public videos, where the cards are and which cards
they are. It holds text only: a row names a video and a time, never the picture. The frame is
obtained from the video under the [extraction contract](#the-extraction-contract) below. A
manifest you keep for yourself may also name [a recording of your own](#recordings-of-your-own).

- A manifest is a UTF-8 text file with the suffix `.jsonl`: one JSON object per line
  ([JSON Lines](https://jsonlines.org)), one line per labeled frame. Blank lines are skipped.
- `manifest.schema.json`, beside this file, is the JSON Schema of one row. It is a copy of
  `cardvision.data.schema.ROW_SCHEMA`, and a test keeps the two equal. In its patterns `$` is
  the end of the text, as JSON Schema defines it. Some regular-expression libraries let `$`
  match before a final line break as well, and a checker built on one lets an id that ends in
  a line break through. `cardvision validate-manifest` refuses such an id.
- `example.manifest.jsonl` is a small made-up manifest that shows every kind of label.
- `cardvision validate-manifest FILE` checks a manifest against everything on this page.

## Coordinates

**Every coordinate is a fraction of the frame**, never a number of pixels: `x` is a fraction of
the frame's width and `y` of its height, the origin is the frame's top-left corner, and `y`
grows downwards. Multiply by the size of the frame at hand to get pixels; the same row then
serves the frame at any resolution of the same shape.

A point always lies inside the frame, so its `x` and `y` are between 0 and 1. The corner of a
box may lie outside the frame when a card runs past the frame's edge, so a corner's coordinates
may range from -1 to 2; the box must still overlap the frame.

## Row fields

Every field in the first table is required.

| Field | Type | Meaning and constraints |
| --- | --- | --- |
| `manifest_version` | integer | The version of this format. Always `1`. |
| `platform` | string | Where the video is hosted: `youtube` or `twitch`. Or `local`: [a recording of your own](#recordings-of-your-own), which is on no platform. |
| `video_id` | string | The platform's id of the video: 11 characters for `youtube`, digits only (6 to 12) for `twitch`. **A video id identifies the video's channel.** For `local`, a name you choose for the recording. |
| `t` | number | A time in seconds from the start of the video at which the labeled frame is on screen; 0 or more, three decimals at most. See [which frame](#which-frame). |
| `frame_width`, `frame_height` | integer | The size in pixels of the frame the labels were made on (16 to 16384). Used to check that a fetched frame has the same shape, and wherever a measure needs real pixels (a card's proportions, for instance). |
| `source_type` | string | The kind of footage: `physical` or `digital`, optionally with one detail after a slash. See [source types](#source-types). |
| `group` | string | The pseudonymous id of the frame's *recording group*: 2 to 40 characters, lower-case letters, digits and hyphens. See [groups and roles](#groups-and-roles). |
| `role` | string | `train`, `evaluation` or `extra`. See [groups and roles](#groups-and-roles). |
| `trained_on` | boolean | `true` only when this very frame was among the frames the released models were trained on. Only a `train` row can be `true`; a `train` row can be `false`. |
| `completeness` | string | `exhaustive-boxes`: every card in the frame has a box, so anything unboxed is background. `points`: only the listed points are labeled and nothing is known about the rest of the frame. |
| `credit` | string | The credit chosen by whoever contributed the labels: 1 to 80 characters, no line breaks, no space at either end. |
| `reachable` | boolean or null | Whether the video could be found on its platform when last checked; `null` when it was never checked. Always `null` on a `local` row. |
| `reachable_checked` | string or null | The date of that check as `YYYY-MM-DD`; `null` exactly when `reachable` is `null`. |

Optional fields. An optional field is left out when it has nothing to say; it is never `null`
and never an empty object.

| Field | Type | Meaning and constraints |
| --- | --- | --- |
| `boxes` | array | The frame's [boxes](#boxes). Required when `completeness` is `exhaustive-boxes`, where an empty array means a frame with no card in it. Absent or empty on a `points` row. At most 2000. |
| `points` | array | The frame's labeled [points](#points). Required, with at least one point, when `completeness` is `points`; allowed beside the boxes of an `exhaustive-boxes` row. At most 2000. |
| `prefill_offered` | boolean | Whether a model's proposed boxes were shown to the labeler on this frame, to be corrected, before the labels were finished. |
| `attributes` | object | Further facts about the frame. See [attributes](#attributes). |

A row is identified by `platform`, `video_id` and `t` together: no two rows of a set of
manifests may share all three.

### Recordings of your own

A phone or screen recording of your own play is on no platform. A row names it with
`platform` set to `local`:

- **`video_id` is a name you choose** for the recording, for example `kitchen-table-01`: 1 to
  64 letters, digits, `_` and `-`, the first a letter or a digit. No space, no dot, no slash.
  The name identifies no file by itself. You say which file it is each time frames are wanted:
  `cardvision fetch --video-file kitchen-table-01=/path/to/the/recording.mp4`.
- **It is never downloaded.** `cardvision fetch --download` does not apply to a `local` row. A
  `local` row with no file given is left without a frame, like any row without a video.
- **`reachable` and `reachable_checked` stay `null`.** Nobody else can look for the video, so
  there is no check to record. A `local` row with either field set is refused.
- **It cannot be offered to this repository.** Nobody else can obtain the video, so nobody
  else could use the labels. `cardvision validate-manifest --contribution` refuses a `local`
  row, and the manifests under `data/` hold none. A manifest you keep elsewhere may use
  `local` freely, and every tool here reads it: the validator, `cardvision fetch`,
  `cardvision evaluate`, the trainers and the scoring page of `browser/`.

Every other field means what it means for any row. Recordings made at one table, with one
camera, belong in one `group`.

## Boxes

A box marks one card, or one region to leave out.

| Field | Required | Type | Meaning and constraints |
| --- | --- | --- | --- |
| `kind` | yes | string | What the box holds: `face` (a card face), `back` (a card back), `unreadable` (it cannot be told what this is), `not-card` (a person judged that this is not a card; such a box usually began as a model's proposal). |
| `ignore` | yes | boolean | `true` makes the box an *ignore region*: training neither rewards nor penalizes a detection there and scoring does not count one. Used for a card that is there but not worth learning from: mostly covered, cut off by the frame, blurred. `kind` then only records what was marked. |
| `quad` | yes | array | The outline: four corners, each `[x, y]`, **clockwise as seen on screen**. The corners must form a convex outline with every edge at least one pixel long on the labeled frame. |
| `oriented` | yes | boolean | `true` when the first corner is the card's own top-left, so the first edge is the card's top edge whichever way the card is turned. `false` when the corner order says nothing about which way is up. |
| `shape` | yes | string | `rotated-box`: a rectangle fitted to the card, turned to match it. A card seen at an angle is covered by the best-fitting rectangle. `quad`: four corners placed one by one, which follow perspective. |
| `card_id` | faces only | string or null | The catalog id of the card. A string exactly when `identity` is `named`, else `null`. An id is 1 to 64 characters, with no space, no line break and no other whitespace or control character anywhere in it. |
| `identity` | faces only | string | `named`: `card_id` says which card. `unknown`: a person looked and could not name it. `unlabeled`: nobody was asked to name it. |
| `readable` | faces only | string or null | Whether the card can be named **from this frame alone**: `yes`, `maybe` or `no`; `null` when that was not recorded. A card named by reasoning from the game, or from a sliver of its art, is `named` and `no`. |
| `id` | no | string | A name for the box, unique among the boxes of its row (1 to 32 letters, digits, `.`, `_`, `-`). |
| `origin` | no | string | How the box came about, as a short lower-case tag, for example `manual` (drawn by hand) or `prefill` (a model's proposal that a person kept or corrected). |
| `attributes` | no | object | Further facts about the box. See [attributes](#attributes). |

`card_id`, `identity` and `readable` are all three present on a `face`, ignore region or not,
and all three absent on any other kind.

What a box covers:

- A box that is not an ignore region covers **the whole card**, including any part hidden under
  a small object (a die, a counter, a fingertip, the edge of the card above). It follows the
  card's outer border, not the art frame inside it and not a sleeve or case around it.
- An ignore region covers only what is visible, and one region may cover several cards lying
  together (the strips of a stack, the hidden cards of a fanned hand). Where an ignore region
  overlaps an ordinary box, the ordinary box still counts in full.
- A neat pile of backs (a deck) has one box, the outline of its top card. A spread pile of
  backs has one box around the pile. Backs are never named.
- A card with half or less of it in frame is an ignore region around the visible part. A card
  mostly in frame is an ordinary box around the whole card, which may run past the frame's edge.

Two boxes of one row may not be identical.

## Points

A point says what is under one position of the frame. Points are how a set is labeled for the
hover measure: what should be shown when the pointer rests there.

| Field | Required | Type | Meaning and constraints |
| --- | --- | --- | --- |
| `x`, `y` | yes | number | The position, 0 to 1. A point on a card's edge counts as on the card. |
| `kind` | yes | string | What is under the point: `face`, `back`, `unreadable`, or `not-card` (the table, an overlay, the client's interface: nothing should be shown there). |
| `card_id`, `identity`, `readable` | faces only | | As for a box. |
| `id` | no | string | A name for the point, unique among the points of its row. |
| `origin` | no | string | How the point was chosen, as a short lower-case tag, for example `seeded` (drawn at random over the whole frame from a seed) or `center` (placed by a person on the center of a card). |
| `attributes` | no | object | Further facts about the point. See [attributes](#attributes). |

Points drawn at random fall on and off cards alike, so a `not-card` point is a real negative: a
tool that shows a card there is wrong. Two points of one row may not share a position.

The labeling tool's census of card centers is written as points with `origin` `center`, one on
the center of every card a person clicked, and its seeded points with `origin` `seeded`; a
frame labeled both ways is one row holding both. A center is named afterwards from a crop
around it, so the point's `readable` says whether the name was legible in that crop, and the
point carries `attributes.readable_from` `crop` to say so. A point named with no model's
proposal shown carries `attributes.blind` true (`labeler/FORMAT.md`, "Exports").

The points of the project's own evaluation set also carry `attributes.verified_by`, how the
name was given: `accept-both` (two models proposed the same card and the person accepted it),
`accept-one` (the person accepted one of two models' names), `picked` (the person chose the
card) or `unknown`. They record `blind` false where proposals were shown. The labeling tool
writes neither; they come from the project's own labeling.

## Attributes

`attributes`, on a row, a box or a point, holds further facts as short machine-readable values.
It is not a place for notes: there is no free text anywhere in a manifest.

- 1 to 16 entries.
- A key is lower-case letters, digits and underscores, starting with a letter (32 at most).
- A value is a number, `true` or `false`, or a string of 1 to 48 letters, digits, `.`, `_`
  and `-`.

Two keys are read by the tools:

| On | Key | Value | Meaning |
| --- | --- | --- | --- |
| row | `frame_dhash` | 16 lower-case hexadecimal digits | The [fingerprint](#the-frame-fingerprint) of the frame the labels were made on. Any other form (wrong length, upper case, a character outside `0-9a-f`, a number) is refused. When it is present, `cardvision fetch` finds the frame by content instead of by time alone. |
| box | `visible_share` | `half` | The card is drawn showing only about half of itself (a digital client's hand row, for instance), and that half is what a viewer points at. Such a box is kept as a face although its proportions are not a whole card's. |

Any other key is carried through untouched.

## Source types

`source_type` says what kind of footage a frame is. Two families are fixed:

- `physical`: cards on a table, filmed by a camera.
- `digital`: a game client's screen.

A value is a family alone or a family with one detail after a slash. These are built in:

| Value | Meaning |
| --- | --- |
| `physical` | Physical play, no further detail. |
| `physical/tournament` | An event's own camera over a table. |
| `physical/webcam` | A player's webcam or desk camera. |
| `physical/phone` | A hand-held or propped-up phone. |
| `digital` | A digital client, no further detail. |

A game adds its own, typically one per digital client, in a *sources file*:

```json
{
  "sourceTypes": ["digital/client-a", "digital/client-b"],
  "handRow": {"sourceTypes": ["digital/client-b"], "minCenterY": 0.85}
}
```

`sourceTypes` extends the built-in list; a detail is lower-case letters, digits and hyphens.
`handRow` is optional: it names the sources whose client draws the cards in hand cut off along
the bottom of the screen, and how far down the frame (measured at a box's center, as a
fraction of the height) that row begins. The label conversions use it to keep those cut-off
cards as faces. Pass the file to the validator with `--sources`.

## Groups and roles

Videos that share an origin, such as one channel or one event, form a **recording group**.
The same table, camera and sleeves appear throughout a group, so a model that has trained on
one of its videos has in effect seen the others.

- `group` is a pseudonymous id. It says which rows belong together and nothing else.
- A video belongs to exactly one group.
- A group has exactly one `role`, shared by every row of every video in it. **A group is never
  split across roles.**

The roles:

| Role | Meaning |
| --- | --- |
| `train` | Published as training data. |
| `evaluation` | Part of the public evaluation set. Never trained on. |
| `extra` | Published, and in neither set: labels that were used in some other way, and labels that have been offered but not yet given a role. |

**Nobody chooses a role.** A new group gets `train` or `evaluation` from a seeded shuffle
(`cardvision assign-roles`, `cardvision.data.roles`): the groups are put in a fixed order,
shuffled with a seed that is published with the result, and dealt out so that each source type
ends up with a set share of its groups in `evaluation`. The rule looks at no frame and no
model output. A group that already has a role keeps it, and a group with a frame that has
trained a released model stays in `train`.

Offered labels therefore arrive with `role` set to `extra`, `trained_on` set to `false` and
`reachable` set to `null`; `cardvision validate-manifest --contribution` checks exactly that,
and that no row is of [a recording of your own](#recordings-of-your-own).

`role` and `trained_on` answer different questions. `role` is where the row is published.
`trained_on` is a fact about the released models, taken from the lists they were trained on:
a `train` row can be `false` (a frame published for training that the released models did not
use), and no other role can be `true`.

## How reachability is recorded

`reachable` and `reachable_checked` say whether a row's video could be found on its platform,
and when that was checked. The check is a `cardvision fetch --download` run, and the record of
the run is the index it writes ([the frame folder](#the-frame-folder)). One command copies the
outcome into the manifest:

```
cardvision record-reachable MANIFEST --index DIR/index.jsonl [--date YYYY-MM-DD]
```

Each row is judged by its last line in the index, and only on evidence about the video itself:

| The index says | The row gets |
| --- | --- |
| `ok`, with the `source` `download`: the row's frame was obtained from a download | `reachable` true, and the date |
| `unusable`, with the `source` `download`: the video was downloaded, and it holds no frame near `t` that matches the row's fingerprint, or it has another shape | `reachable` false, and the date |
| `failed`, with the `source` `download` and the `cause` `unavailable`: the platform said the video is private, removed or does not exist | `reachable` false, and the date |
| `failed`, with the `source` `download` and any other `cause`, or none: the video is restricted (`restricted`), or the download failed for a reason that says nothing about the video (`error`: no network, the program that cuts a section, a timeout, a file that could not be read) | Nothing: both fields stay as they were. The command lists the row |
| Any line with the `source` `kept`: the frame came, or did not come, from a file kept from an earlier download | Nothing. A kept file says nothing about the platform today |
| Any line with the `source` `local`: the file was one you gave | Nothing |
| `no-video`, or no line at all: the row had no video, was not selected, or the run stopped before it | Nothing |

- **A failed download proves nothing by itself.** A run made with no network fails every row,
  and every video is still there. Only the platform's own statement that a video is gone
  (`unavailable`) turns a row to false.
- The command prints every row it sets to false, and every row it leaves alone after a failed
  download, each with its reason, and then the counts.
- **The date is the date, in UTC, on which the row's index line was written** (`written`), so
  the command may be run any time after the fetch. `--date` records one date for every row in
  place of that. An index written by an older version of `cardvision fetch` records no time on
  its lines: the command then asks for `--date`, the date of that fetch run, and does not
  guess one.
- A `local` row is never set ([recordings of your own](#recordings-of-your-own)).
- A download can fail for a passing reason. Run `cardvision fetch` again before you record:
  it retries the rows without a frame, and the last line of a row is the one that counts.
- The manifest is rewritten in place in the canonical row form: one compact JSON object per
  line, with the fields in the order in which `cardvision assign-roles` and the labeling
  tool write them. In a manifest already in that form, the two fields are the only bytes that
  change. The command opens no network connection.

Whoever offers labels does not run this. The check is made when labels are accepted.

## What the validator checks

`cardvision validate-manifest FILE [FILE ...] [--catalog C.json] [--sources S.json] [--contribution]`

It reads files and nothing else: no network, no image decoding. Given several files, or a
folder, it checks them as one set.

| Code | Refused |
| --- | --- |
| `json` | A line that is not a JSON value (`NaN` and `Infinity` included). |
| `schema` | A row that does not fit the JSON Schema: a missing or unknown field, a wrong type, a value outside its list or range; boxes missing from an `exhaustive-boxes` row; a `points` row without points or with boxes; identity fields missing from a face or present on anything else; `card_id` that disagrees with `identity`, or that has whitespace or a control character in it; `trained_on` true outside `train`; `reachable` and `reachable_checked` not both set or both null, or set on a `local` row; a video id of the wrong form for its platform; a manifest with no row. |
| `row` | A time with more than three decimals; a date that is not a calendar date; a quad that is not convex, runs counter-clockwise, has an edge under one pixel or lies wholly outside the frame; a box or a point given twice in one row; a label id used twice in one row. |
| `duplicate-row` | Two rows with the same platform, video id and time, in one file or across files. |
| `group` | A group with two roles; a video in two groups. |
| `source-type` | A source type that is neither built in nor listed in the sources file. |
| `unknown-card` | With `--catalog`: a `card_id` the catalog does not hold. |
| `contribution` | With `--contribution`: a role other than `extra`, `trained_on` true, `reachable` set, or a row whose `platform` is `local`. |
| `file` | Any file in a manifest's folder, at any depth, that is not plain UTF-8 text: an image or video by its name or by its first bytes, a file with a zero byte, a symbolic link, or a text file that embeds an image as `data:image/...`. **A manifest folder holds no frame and no picture of any kind.** |

## The extraction contract

Labels are fractions of a frame, so they land on the right pixels only when the frame at hand
is the frame that was labeled. This is how that frame is obtained. `cardvision.data.extract`
holds the same rules as code.

### Which stream

The best video-only stream that is at most 1080 pixels high, as MP4 when the platform offers
one. In the downloader's format-selection syntax:

```
bv[height<=1080][ext=mp4]/bv[height<=1080]/b[height<=1080]/bv/b
```

That is: the best video-only MP4 stream within the limit; failing that, the best video-only
stream within the limit in any container; failing that, the best combined stream within the
limit; failing that, whatever there is.

### Which frame

`t` is a time at which the labeled frame is on screen. **The frame for `t` is the last frame
whose own timestamp is at or before `t`.** In a video of constant frame rate `r` that is frame
number `floor(t * r)`, counting from 0.

Published rows put `t` at the middle of the frame's time on screen, `(n + 0.5) / r` for frame
number `n`, rounded to the millisecond. A time in the middle of a frame cannot be moved onto a
neighboring frame by rounding, or by the small differences in how tools report timestamps.

The rule is stated in timestamps because a frame number depends on the exact frame rate:
30000/1001 frames a second is not 29.97, and a rounded rate is a whole frame out after a few
hours of video. And a video need not have one frame rate: a phone or a screen recorder often
writes a file of variable frame rate, in which one frame stays on screen for a sixtieth of a
second and another for seconds. There the frame for `t` has no number that could be worked
out, and the rule holds as it stands.

**A frame's timestamp is its presentation timestamp, as the video's container gives it.**
`cardvision fetch` reads it from the decoder for each frame it decodes, and tells frames apart
by these timestamps alone, at a constant frame rate and at a variable one. No frame is placed
by a frame rate.

Seeking must be exact to the frame. A seek that lands on the nearest keyframe gives a frame
that can be seconds away, and a wrong frame makes every label of the row wrong. So a seek is
treated as a guess, and checked:

1. The tool asks the decoder for a place about a second before the time it wants. The decoder
   finds that place by the one frame rate the file reports for itself, an average or a
   nominal rate, so in a file of variable frame rate it can land seconds early or late.
2. The first frame decoded there must have a timestamp at or before the time wanted. If it
   has not, the tool asks again further back, by at least twice as much each time, and in the
   end reads the file from its start.
3. From a place known to be early enough, it steps forward frame by frame, reading each
   frame's timestamp, to the last frame at or before the time wanted.

`cardvision fetch --exact-seek` leaves out the seek: it reads each file from its start, frame
by frame, by the same timestamps. That is slow on a long file, and it is there for files that
do not seek well. Both ways give the same frames.

No timestamp says how long the last frame of a file stays on screen. It is taken to stay for
one frame's time at the rate the file reports, and a time up to half a frame's time later
still counts as on it. A time after that is past the end of the file: a row without a
fingerprint whose `t` lies there is `failed`, and its reason says that the video ends before
`t`.

### What size

The frame is used whole, at the size it decodes to. Nothing is cropped, padded or stretched.

Its **shape must match** the row's: the frame's width over its height must equal
`frame_width` over `frame_height` to within 1 % of that ratio. A frame that fails is reported
as **unusable** for the row and is not written; no tool applies labels to it. The size may
differ freely (a 1280 x 720 frame serves a row labeled at 1920 x 1080), because fractions
scale. Such a frame is never used silently: `cardvision fetch` counts the frames it wrote at
another size than their row records, and marks each in the index (`other_size`).

### The frame fingerprint

A row may carry `attributes.frame_dhash`: a 64-bit fingerprint of the frame the labels were made
on, written as 16 lower-case hexadecimal digits. It lets a fetch confirm that the frame it found
is the labeled one, and find it when the video it has is a little out of step with the video
that was labeled. The fingerprint is a *difference hash*, defined exactly so that any program
can compute it from the frame's pixels (`cardvision.data.dhash.dhash` is the reference code):

1. **Grayscale.** Each pixel of the RGB frame becomes `round(0.299 R + 0.587 G + 0.114 B)`,
   the sum in double precision and the rounding to the nearest whole number, halves to the
   even one.
2. **A grid of 9 columns by 8 rows.** The pixel at column `x` and row `y` of a frame `W` pixels
   wide and `H` high lies in cell column `min(8, floor(9x / W))` and cell row
   `min(7, floor(8y / H))`. Each cell takes the mean gray value of its pixels (0 for a cell
   with none, which only happens in frames smaller than the grid).
3. **Bits.** Going through the 8 cell rows from the top, and in each the first 8 cells from the
   left, the bit is 1 when the cell to its right is brighter than it and 0 otherwise. That
   gives 64 bits; the first is the most significant, and the number is written as 16
   lower-case hexadecimal digits.

The distance between two fingerprints is the number of bits in which they differ (the Hamming
distance, 0 to 64). Two decodes of one frame, at the same or different resolutions, and with
the loss of JPEG or video compression, differ by a few bits at most; two different frames of a
video usually differ by many more. Fingerprints **at most 6 bits apart** (`SAME_FRAME_MAX_HAMMING`)
are taken to be of the same frame.

A fingerprint is not the frame. It records only which of 72 coarse regions are brighter than the
region on their right, so the picture cannot be rebuilt from it, and it reveals nothing beyond
that coarse brightness pattern.

### Which frame, when the row has a fingerprint

A downloaded stretch of video is cut with only frame precision at best, and a platform may
serve a version of a video that starts a frame or a second earlier or later than the one that
was labeled. So for a row with `frame_dhash`, `cardvision fetch` does not trust `t` alone:

1. It reads every decoded frame on screen from 1 second before `t` to 1 second after it
   (the frame on screen at `t - 1` included), about 60 frames at 30 frames a second.
2. It fingerprints each of them, and keeps the one at the smallest Hamming distance from the
   row's fingerprint. A tie goes to the frame nearest `t` in time, then to the earlier one.
3. If that distance is at most 6, it writes that frame under the row's usual file name and
   records, in the index, the distance and how far in time the frame was from `t`.
4. Otherwise the row is **unusable** with reason `fingerprint`: the video does not hold the
   labeled frame near `t` (a different version of the video, or a different cut). No frame is
   written for it, and the index records the smallest distance found.

The shape check under [what size](#what-size) still applies to the frame kept. A row without
`frame_dhash` is handled as before: the frame for `t`, by the rule under
[which frame](#which-frame).

**At the two ends of a file** the second either side of `t` is cut short, and the frames that
exist are read:

- A row whose `t` lies past the end of the file, by no more than a second, still has the
  second before it. Its frames, up to the file's last, are fingerprinted as in step 2, and the
  row is kept or is `unusable` as in steps 3 and 4. A row with `frame_dhash` is therefore not
  failed merely because its `t` is past the end.
- A row whose whole second before `t` lies past the end has no frame to read. It is `failed`,
  and its reason says that the video ends before `t - 1`, where the row's window starts. The
  file's last frame is not compared with the row's fingerprint, though it may look like the
  labeled frame.
- The end of the file is where its last frame leaves the screen, as
  [which frame](#which-frame) says: one frame's time after that frame's timestamp, and half a
  frame's time of allowance.

The scoring page of `browser/` handles the two cases in the same way. It takes the end of the
file from the length the browser reports for it. That can differ from the above: by a
fraction of a frame's time in most files, and by more where the last frame of a file stays on
screen for long.

### A fetched frame is not the labeled frame, byte for byte

A frame obtained today is **not byte-identical** to the frame that was labeled, and cannot be
made so:

- the platform re-encodes its streams from time to time and may no longer offer the stream the
  labels were made on, so today's best stream may differ in resolution or codec;
- different decoders round differently;
- the labeled frames were themselves stored as JPEG, a further lossy step.

The picture is the same picture and the labels land in the same places, but pixel values
differ slightly. Figures computed on re-fetched frames can therefore differ a little from
published ones, and a published figure states the tolerance it expects. The shape check above
is the test the tools apply; they do not compare pixels.

### How the frames were first sampled

Unless the notes published with a set say otherwise, the frames that were labeled were chosen
from frames sampled as follows.

- **Fixed-rate sampling.** One frame per second was taken from the stream above: from the
  whole video when it ran 45 minutes or less, and otherwise from three ten-minute stretches
  starting at 25 %, 50 % and 75 % of its length. The sampler's n-th frame, counting from 1, is
  the last frame before `n - 0.5` seconds of the stretch (the frame nearest ahead of each
  half second), so it shows the video at about `n - 0.5` seconds after the stretch began.
  `cardvision.data.extract.sampled_frame_time` turns such a frame number into a row's `t`.
- **Single frames.** Some frames were taken one at a time at a chosen whole second, as the
  first frame at or after it (`frame_time_at_or_after`).

Either way the published `t` names the frame itself, by the rule under
[which frame](#which-frame); the sampling is described so that the numbers can be traced, not
because a reader has to repeat it.

## The frame folder

`cardvision fetch MANIFEST --out DIR` writes:

- `DIR/frames/<platform>-<video id>-t<milliseconds, ten digits>.jpg` (or `.png`), one file per
  row that has a usable frame, for example `youtube-EXAMPLE0001-t0000754517.jpg`;
- `DIR/index.jsonl`, one line per row attempted, appended as the run goes. When a row appears
  more than once the last line counts.

An index line:

| Field | Meaning |
| --- | --- |
| `platform`, `video_id`, `t` | The row. |
| `status` | `ok`; `unusable` (a frame was read and failed the shape check, or no frame matched the row's fingerprint); `no-video` (no file was given and no download was asked for, or no file was given for a `local` row); `failed` (the download failed, the file could not be read, or it has no frame at `t`: for a row with a `frame_dhash`, no frame in the second before `t`). |
| `file` | The frame, relative to `DIR`, when `status` is `ok`. |
| `width`, `height` | The size of the frame read. |
| `frame_time` | The timestamp of the frame read, in seconds of the video. |
| `scale` | The frame's width over the row's `frame_width`. |
| `other_size` | `true` when the frame written is not `frame_width` by `frame_height` pixels. Left out when the size is the row's. |
| `source` | `local` (a file you gave), `download`, or `kept` (a file kept from an earlier download). This says where the file came from, whatever the row's `platform`. |
| `attempts` | Only for a download: how many times the row's part of the video was asked for. 1, or more when a failed download was tried again. A download that failed with the cause `unavailable` or `restricted` is not tried again. 0 for a part that was not asked for at all, because an earlier part of the same video had that answer. |
| `reason` | Why, when `status` is not `ok`. `fingerprint` means no frame within a second of `t` was within 6 bits of the row's `frame_dhash`. |
| `cause` | Only when `status` is `failed` and `source` is `download`: why, in one word. `unavailable`: the platform said the video is private, removed or does not exist. `restricted`: the video is there and this request may not have it (an age gate, a video for a channel's members, a video kept out of a country). `error`: anything else, which says nothing about the video (no network, the program that cuts a section, a timeout, a downloaded file that could not be read or that ends before `t`, a whole video longer than the limit set with `--max-whole-minutes`, which is then not downloaded). `refused` is a fourth cause the code knows and never writes: a refusal stops the run and writes no line. |
| `distance` | Only for a row with a `frame_dhash`: the distance in bits between the fingerprint and the frame kept, or, for reason `fingerprint`, the nearest frame found. |
| `offset` | Only for a row with a `frame_dhash` and a frame kept: `frame_time` minus `t`, in seconds, to the millisecond. A negative value means the labeled frame was found earlier than `t`. |
| `written` | When the line was written: UTC, ISO 8601, to the second, for example `2026-01-31T21:15:00Z`. An index written by an older version has no such field. |

Running the command again skips rows that already have a frame and retries the rest.
`DIR` must lie outside the manifest's folder, which holds text only.

### When a download fails, and when the platform refuses

- **A failed download is tried once more, unless the platform has answered for the video.**
  The `cause` of a failure is read from the downloader's whole error
  (`cardvision.data.fetch.failure_cause`). A part of a video whose download fails with the
  cause `error` is asked for again after the usual pause, once, or as often as `--retries N`
  says. A failure with the cause `unavailable` or `restricted` is the platform's own statement
  about that video, and asking again would get the same answer: the part is not asked for
  again, and its `attempts` is 1. Either way the rows of a part that failed are `failed`,
  with the downloader's error as the `reason`: the start of the error's first line, where the
  cause is stated, and its end. The later parts of the same video in the same run are not asked
  for either: their rows are `failed` with the same cause, `attempts` 0 and a `reason` that says
  an earlier part of the video was answered so. `record-reachable` judges those rows as it
  judges the part that was asked for.
- **A very long whole video can be refused.** With `--whole-video` (or when sections cannot be
  cut), `--max-whole-minutes N` stops a video that the downloader reports as longer than `N`
  minutes from being downloaded. The length comes from what the downloader reads about the
  video before it downloads, so the check makes no request of its own. The rows of such a video
  are `failed` with the cause `error` and a `reason` that gives the length and the limit. It is
  not asked for again and does not count towards the failures in a row. Without the option
  there is no limit.
- **A refusal stops the run.** When the downloader's error says the platform refused the
  request, nothing more is asked for. The signs are an HTTP 429 ("Too Many Requests"), the
  downloader's words for a rate-limited session, for a captcha and for a blocked address, the
  words "not a bot", and a redirect to a "sorry" page
  (`cardvision.data.fetch.REFUSAL_PATTERNS`). The words "sign in to confirm" are no sign by
  themselves: "sign in to confirm your age" is the age gate of one video, and its rows are
  `failed` with the cause `restricted`. The rows of the refused part, and every row after
  them, get **no line** in the index: they were not tried, which is not the same as failed.
  The command says how many, and exits with code 3.
- **Several failures in a row stop the run too.** A section is cut by a helper program, whose
  errors on the platform's media server come back as an exit code and nothing more, so a rate
  limit there cannot be read from the error. When 3 parts in a row have failed, each after its
  retries, with the cause `error`, nothing more is asked for (`--stop-after-failures N`; `0`
  never stops). A part that downloads sets the count back to nothing. A failure with the cause
  `unavailable` or `restricted` neither counts nor sets the count back. The parts that failed
  keep their `failed` lines; every row after them gets no line. The command says how many rows
  were not tried, and exits with code 3 here as well: **3 means the run stopped early and rows
  were not tried.**
- **A certificate that cannot be verified stops the run too**, with a message that says what
  to install ([data/README.md](README.md), "Get the frames"). No row gets a line.

### Kept video files

`cardvision fetch --download` deletes each downloaded file once its frames are read. With
`--keep-video DIR` it moves the file to `DIR` instead. `DIR` must lie outside the manifest's
folder too. Files given with `--video-file` are yours already and are not copied.

**The name of a kept file** says which video it is, and where the file starts in that video:

- `<platform>-<video id>+<start>.<ext>` for a section of a video, for example
  `youtube-EXAMPLE0001+77.mp4`;
- `<platform>-<video id>.<ext>` for a whole video, for example `youtube-EXAMPLE0001.mp4`.

`<start>` is the time in the video, in seconds, at which the file starts. A whole number of
seconds is written without a decimal point (`77`, `0`). Any other time is written to the
millisecond at most (`12.5`). `<ext>` is the extension the downloader gave the file.
`cardvision.data.fetch.kept_video_name` is the reference code. The scoring page of `browser/`
reads the same form ([browser/README.md](../browser/README.md), "The scoring page").

**The index**, `DIR/video-files.json`, lists the kept files:

```
{"kind": "cardvision-video-files", "version": 1, "files": [
  {"file": "youtube-EXAMPLE0001+77.mp4", "platform": "youtube", "video_id": "EXAMPLE0001",
   "start": 77.0, "end": 84.0, "rows": [80.017]}
]}
```

| Field | Meaning |
| --- | --- |
| `file` | The file's name in `DIR`. |
| `platform`, `video_id` | The video. |
| `start` | The time in the video at which the file starts, in seconds. `null` for a whole video. |
| `end` | The time in the video up to which the section was asked for. `null` for a whole video. |
| `rows` | The times `t` of the rows the file serves, in ascending order. |

`start` is the offset `cardvision fetch` itself read the file with: time `x` of the file is
time `start + x` of the video. A section is cut only as precisely as a frame, so the file's
first frame is the first frame of the video at or after `start`, and a time read from a
section can be up to one frame early. The fingerprint allows for that. A row without a
fingerprint can land one frame off in a section.

A file downloaded again replaces the kept file of the same name. Running the command again
adds to the index and keeps the entries of files it did not download again.

**Kept files are read again.** With `--keep-video DIR`, with or without `--download`, a row
that a file in `DIR` covers gets its frame from that file and is not downloaded. Its index
line has the `source` `kept`.

- A whole-video file covers every row of its video.
- A section listed in `video-files.json` covers the rows whose times are in its `rows`. A row
  whose time merely falls inside the section is not covered: the section was cut for its own
  rows, and may lack the second either side of another row that the fingerprint rule reads.
- A file the index does not list is known by its name. A whole-video name covers every row of
  the video. A section name covers the rows at or after its start, up to the next such
  section; whether the file reaches a row is found out when it is read.
- With `--whole-video`, only whole-video files are used.
- If a kept file does not give a row's frame, the row is downloaded when `--download` is
  given, and is otherwise reported with its own status. One case differs: a section known by
  its name alone that does not reach a row never covered that row, so without `--download`
  the row is `no-video`, not `failed`.

So a file given the name of a kept file and put in `DIR` by hand is read as one. Nothing in
`DIR` changes when a kept file is read.
