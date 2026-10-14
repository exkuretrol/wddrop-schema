"""
Wire format shared by the client and the server.

UNIT OF ANALYSIS = one chest open, with its FULL content list.

The recording target is everything the game reports after a chest is opened — which in
practice is mostly `Item::Junk` and `Item::SaleOnly`, plus equipment when it drops. The
whole content list is kept together, because "what did this chest give me" is the
observation; splitting it into separate rows would lose the fact that they came from one
roll.

THE GAME IS IN JAPANESE. Since client 0.5.0 the recogniser reads one language, because it
is the only one whose face renders from the files the game already installed — so every
name on this wire is a Japanese name and `locale` is `ja` on every event.

Confirmed against both sources, which agree exactly:
  * API  `received_contents: [{content_id, content_type, quantity, boost_id}]`
  * UI   one message line per item, `DungeonTreasure@DropItem` + `Common@NameAndQuantity`:
    e.g. 「蒼雫の鉱石×3を手に入れた!!」 -> content_id 20000001, Item::SaleOnly
    Mining announces the same acquisition differently, `Common@GetItem` 「{0} を入手した」,
    which is why it took its own reader.

Two provenances stay distinguishable because they have different published rate tables
and different confounders:

    chest_direct   contents straight out of a treasure chest
    junk_reversal  equipment produced by reversing a junk item

DESIGN RULES
------------
1. The client never sends account identity. It sends a locally-generated `install_id`
   (random UUID4, persisted in the client config). The server HMACs that with a
   server-only salt to get `player_id`. Player name, code, gold, characters, party and
   friends are never collected in either mode. See DISCLAIMER.md.

2. The client never decides the calendar day. It sends UTC + its own tz offset; the
   server derives the JST day bucket (the game's own timezone). A client with a wrong
   clock can be detected and quarantined rather than silently poisoning a day bucket.

3. Empty chests count. `contents` may legitimately be an empty list; see the field docs.

4. WHAT IS SENT IS WHAT IS STORED. Every field here is filled by the shipping client, and
   the server's tables carry these columns and no others. A column nothing fills is not a
   placeholder for a better client — it is a column of NULLs that reads, in a year, as a
   measurement that was taken and came back empty.

KNOWN LIMIT, AND IT IS NOT SOLVED BY THIS FORMAT
------------------------------------------------
The recogniser matches a CLOSED vocabulary and refuses anything under threshold, so a line
it cannot place is never emitted at all. Unreadable drops are therefore silently absent,
which biases the sample toward whatever renders cleanly — the very kind of bias this study
measures. `match_confidence` bounds it from above (how close the accepted readings run to
the gate); the local review queue holds near-miss lines for a human. Neither is a record of
the drop that got away.

This docstring used to claim the opposite — that unmatched OCR was transmitted with its
`raw_text` for later repair. That was true of `capture/ocr.py:resolve_line()`, which is not
on the live path and never was: both emit paths reconstruct their text from the name they
already recognised, so `raw_text` restated `item_name` through a template and no unmatched
line ever reached it.

IDS ARE STABLE ACROSS A GAME UPDATE. NAMES ARE NOT. (measured 2026-08-13)
-------------------------------------------------------------------------
Comparing the game's own item and equipment tables at three dates a week apart — the last
comparison spanning a content update — says which half of a record can still be trusted to
mean the same thing next month:

    items         2,582 entries throughout
                  ids added 0, removed 0, RETYPED 0 ... and 6 RENAMED
                  993000571  【ジラートの幻影】のクリスタル
                          -> 希望のクリスタル【ジラートの幻影】   (and five siblings)
    equipment     identification 823 -> 823, zero renames
                  the per-row id 3,719 -> 3,720, i.e. one new roll variant

So `item_id` and `equipment_identification` are the durable identity, and `item_name` is a
DATED OBSERVATION: six of them are already wrong against today's tables. That cuts both ways
and both ways are worth knowing —

  * as IDENTITY the name is unusable, and it was never used that way here;
  * as EVIDENCE it is the only record of what the player was actually shown, because
    `item_reference` is replaced wholesale on every load and so always reads back today's
    name for a drop recorded weeks ago.

Dropping `item_name` from this wire is therefore a real option and not merely a saving (it
is ~34 bytes a line, 2.7 MB at 80,000). It costs the evidence above, and it makes an
unresolved line — which the client cannot currently produce, since it never emits a name it
could not place — into a row with no identity at all. Decide it deliberately; it is a wire
break, so it needs a SCHEMA_VERSION bump and the `client_policy` floor raised to the build
that stops sending names.

REMOVED, DELIBERATELY (2026-08-13), so that re-adding one is a decision and not a rediscovery
--------------------------------------------------------------------------------------------
    raw_text                     reconstructed from item_name; never the pixels read
    item_type                    a pure function of item_id in the vocabulary — join for it
    equipment_name               likewise, of equipment_identification
    equipment_id                 the exact row is never resolvable from a display name
    quality, level               nothing reads ★ or 等級 off the screen yet
    source_junk_name/_id         no capture emits `junk_reversal`
    game_version                 no screen the recogniser reads shows it. It remains a
                                 COLUMN on the server, stamped over a date range as each
                                 build's live date is learned (`wddrop_server/versions.py`)
Each comes back the day something fills it, with the capture that fills it.
"""
from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

# 2 = the ja-only trim: raw_text, item_type, equipment_name/_id, quality, level,
# source_junk_name/_id and game_version left the wire because nothing filled them. Bumped
# rather than changed quietly, so a batch that predates the trim is identifiable as one.
# 3 adds the two self-reported covariates on CaptureInfo. ADDITIVE and optional, so a client
# that never sends them stays valid and the ingest floor does not move for it.
# 4 adds the optional `mining` block. ADDITIVE and optional on the same terms as 3: a client
# that only opens chests never sends it and stays valid, and the ingest floor does not move.
SCHEMA_VERSION = 4


class CaptureMode(str, enum.Enum):
    """How the record was obtained.

    The field existed while it had ONE value, because it is what lets the analysis split on
    error profile — a second source with different error behaviour could not be mixed in
    without it, and there WAS a second before it was removed for the ban risk it carried.
    That bet has now paid: AUTOPILOT is the second source arriving, and no column had to be
    added for it.

    THE TWO ARE NOT THE SAME POPULATION, and the difference is the study's own confound.
    A human stops playing for reasons that may depend on what they just found — quitting
    after a good drop or a bad streak correlates session end with drop quality and
    manufactures the very pattern this study tests for. A bot stops because its route ended
    or its dive limit was reached, which is independent of the contents. So autopilot rows
    are the arm with that confound removed, and user rows are the arm with it present:
    pooling them silently would waste the comparison, and splitting on this makes it.
    """

    OCR = "ocr"          # reads the game's own result screens off the framebuffer
    # Read by wd-autopilot, which plays unattended and decides WHEN to read from its own
    # screen classification rather than from a HUD-absence episode. Same recogniser, same
    # vocabulary, different sampling: it sees every chest it opens, where a human client
    # sees the ones its capture loop happened to catch.
    AUTOPILOT = "autopilot"


class Provenance(str, enum.Enum):
    CHEST_DIRECT = "chest_direct"
    JUNK_REVERSAL = "junk_reversal"
    # Mining an ore vein with a pickaxe. A separate stream for the same question — players
    # ask whether mining yields degrade with farming time exactly as they ask it of chests —
    # and it MUST stay distinguishable: the two have different item pools, different costs
    # and different rates, so pooling them would answer neither question.
    MINING = "mining"


class LabelSource(str, enum.Enum):
    """Where the dungeon/floor labels came from.

    In OCR mode the player picks the dungeon, so labels are a claim, not a measurement.
    They are read off the screen, so they carry recognition error. Analysis must be able to
    split on this, both to measure the mislabel rate and to avoid treating a user claim as
    ground truth.
    """

    USER_DECLARED = "user_declared"
    API = "api"
    # The floor the client READ off the screen itself, rather than one a human typed into a
    # dropdown and may have left stale. A different error profile, not a better one: it
    # carries recognition error instead of attention error, and the mislabel rate of each
    # is measurable only while they stay apart.
    SCREEN_READ = "screen_read"


class StopReason(str, enum.Enum):
    """How a session ended.

    Recorded because manual start/stop puts the session boundary under the player's
    control, and stopping may be OUTCOME-DEPENDENT — quitting right after a good drop, or
    after a bad streak, makes session-end correlate with drop quality and can manufacture
    exactly the "quality falls with farming time" pattern being tested. Without this field
    that bias is invisible; with it, the analysis can compare user-stopped sessions against
    idle/closed ones.
    """

    USER_STOP = "user_stop"
    IDLE_TIMEOUT = "idle_timeout"
    GAME_CLOSED = "game_closed"
    APP_CLOSED = "app_closed"
    # The bot's route finished, or its dive limit was reached. THE REASON THIS ENUM EXISTS,
    # answered in the negative: a stop decided by a plan written before the dive began
    # cannot depend on what the dive produced. It is the control the other values are
    # compared against.
    ROUTE_END = "route_end"


class ReceivedItem(BaseModel):
    """One line of a chest's contents — junk, sale-only, valuable, or equipment.

    Identity is an ID, not a string: `item_id` for an item, `equipment_identification` for a
    piece of equipment, exactly one of them, and neither when the name did not resolve. The
    item's TYPE (Item::Junk / Item::SaleOnly / ...) is not carried here because it is a
    function of `item_id` in the same vocabulary the client matched against — the analysis
    joins for it rather than trusting a copy that can disagree.
    """

    item_name: str = Field(
        description=(
            "The Japanese name as displayed. Kept beside the id because it is what was "
            "actually on screen, and because it is the only identity an unresolved line has."
        ),
    )
    item_id: int | None = Field(
        default=None, description="The game's own item id."
    )
    quantity: int = Field(default=1, ge=1)
    # WHETHER THAT NUMBER WAS OBSERVED OR INFERRED. The game prints no "×N" for a single
    # item, for equipment, or while a drop boost is active, so the client records 1 and
    # flags it rather than guessing — "never fabricate a quantity" is one of its hard rules.
    # Carrying the flag on the wire is what keeps that rule true past the client: without
    # it an inferred 1 and an observed 1 are the same row. Measured on the first real spool,
    # 40 of 94 content lines were inferred, and NONE of the observed lines was a 1 — so
    # server-side "quantity = 1" would have been 100% inference presented as measurement.
    quantity_unknown: bool = Field(
        default=False,
        # `qty_unknown` is what the capture client has always written into its spool, and
        # those files are the player's only copy of unsent data. Accepting both spellings
        # means a spool written before this field existed still uploads correctly, instead
        # of silently losing the flag it was carrying all along.
        validation_alias=AliasChoices("quantity_unknown", "qty_unknown"),
        description="True when the game printed no number and the 1 is an assumption.",
    )
    # HOW THE NAME WAS CHOSEN, and it changes what `match_confidence` means.
    #
    # Junk comes in families differing by ONE character in seventeen
    # (北穿の幽霊城の常なる/妖なる四鱗のガラクタ), so the identical rest of the line drowns the
    # difference: the whole-line scores sit within 0.03 and the ambiguity gate correctly
    # refuses both. Re-scored over only the columns where the two candidates differ, the same
    # frame separates at a margin of 0.44.
    #
    # The flag has to be here because the confidence beside it is the WHOLE-LINE score, which
    # is low BY CONSTRUCTION for these — 0.83 where an unambiguous name reads 0.89. Analysis
    # ranking on confidence alone would rate the most deliberate readings worst, and could
    # not tell a thin whole-line margin that was resolved from one that was merely accepted.
    # Measured: this tier read ZERO times in 195 bot-read lines before the tie-break reached
    # the embedded reader.
    tie_broken: bool = Field(
        default=False,
        description=("True when two near-identical candidates were separated on the columns "
                     "where they differ, rather than on the whole line."),
    )

    # -- equipment-only field (None for junk / sale-only) ---------------------------
    #
    # A displayed name does NOT identify a single equipment row. In the game's own table
    # 826 distinct names span 3,719 rows; rows sharing a name differ only by
    # grade_lottery_id / rarity_lottery_id and share one `identification` value. So the
    # name resolves to a FAMILY, and it is the observed quality/level that would pin down
    # the variant — which nothing reads off the screen today. Sending a guessed
    # `equipment_id` would be fabricated precision, so the family is the only key here.
    equipment_identification: int | None = Field(
        default=None,
        description="The equipment family key, which is stable where the display name is not.",
    )

    # HOW CLOSE THIS READING RAN TO THE GATE. Every line here was already accepted, so this
    # does not separate right from wrong — it is the distribution that says whether a
    # player's machine is reading comfortably or scraping the threshold, on hardware nobody
    # here can inspect. It is the only recognition-quality signal that survives the client.
    match_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Render-and-compare match score for this line, where the reader has one.",
    )

    @model_validator(mode="after")
    def _one_identity_at_most(self) -> "ReceivedItem":
        """A line is an item, or a piece of equipment, or unresolved — never two of those.

        The capture client cannot produce both (`ItemIndex.identify` returns one key or
        neither), so this is about anything else that speaks this API. Refused HERE, where it
        costs a 422 naming the field, rather than only at the database's own constraint:
        that one fires at COMMIT, which is after every other event in the batch has been
        written, so it would turn one malformed line into a 500 that loses the whole batch.
        """
        if self.item_id is not None and self.equipment_identification is not None:
            raise ValueError("a line cannot be both an item and a piece of equipment")
        return self


class SpotSource(str, enum.Enum):
    """What a vein's `spot` is anchored to, because the two do not age the same way.

    ROUTE_STEP is anchored to the bot's own route — "step 3 of north_veins". It identifies a
    spot only for as long as that route is unedited: insert a waypoint and every historical
    row silently re-attributes to a different place, with nothing looking wrong. CELL is
    anchored to the dungeon itself and survives any route change.

    Recorded rather than assumed because route-anchored is all that is available today. The
    tag is what stops a later, better value being pooled with it as though they were the same
    measurement — which would quietly make the stable one as unreliable as the fragile one.
    """

    ROUTE_STEP = "route_step"
    CELL = "cell"


class VeinEnd(str, enum.Enum):
    """Why mining at a spot stopped. `truncated` says WHETHER the list is short; this says why.

    MINED_OUT and NOT_SPAWNED are complete observations. NO_PICKAXES and GAVE_UP are not: the
    vein had more to give and we stopped, so those set `truncated` as well. Keeping both means
    analysis can filter uniformly on `truncated` across chests and veins without re-encoding
    this table, and still ask the sharper question when it wants to.
    """

    MINED_OUT = "mined_out"
    NOT_SPAWNED = "not_spawned"
    NO_PICKAXES = "no_pickaxes"
    GAVE_UP = "gave_up"


class VeinSpot(BaseModel):
    """Where a vein was. See SpotSource for why the anchor is recorded with it."""

    source: SpotSource
    # ROUTE_STEP: which wall, which route, which step along it.
    facing: str | None = Field(default=None, max_length=8)
    route: str | None = Field(default=None, max_length=64)
    step: int | None = Field(default=None, ge=0)
    # CELL: the dungeon's own grid. Null until the bot can localise the party.
    cell_x: int | None = None
    cell_y: int | None = None


class MiningContext(BaseModel):
    """Mining-only fields. One record per SWING, plus one closing record per vein.

    A swing is the unit because a swing is the ROLL — the same reason a chest is one record
    rather than one per line. A vein is a SEQUENCE of independent rolls, so summing them into
    one row would lose exactly the structure that rule exists to protect.

    Which means the obvious aggregates are DERIVED, not stored, and cannot drift from the rows
    they summarise:

        taps            = count of swing records for the vein
        pickaxes_broken = how many of them set `pickaxe_broken`

    `pickaxe_item` is the item NAME and not an enum of the two known pickaxes, because the
    vocabulary this project matches against went stale the moment the game shipped new items —
    an enum would be that same bet made in a type.
    """

    vein_index_in_dive: int | None = Field(default=None, ge=0)
    spot: VeinSpot | None = None

    # ON A SWING ROW. `contents` carries what that swing yielded, and may be empty: a swing
    # that gave nothing is an observation, not an absence, and dropping it would compute every
    # rate over a population selected for having dropped something.
    pickaxe_broken: bool = False
    pickaxe_item: str | None = Field(default=None, max_length=64)

    # ON THE CLOSING ROW, which has no contents. `vein_spawned=False` is a SPAWN observation,
    # not a yield of zero — a spot where nothing spawned was never a trial, and folding it in
    # with the yields would mix spawn rate into yield per vein. Filter on it.
    vein_spawned: bool | None = None
    end_reason: VeinEnd | None = None


class DiveContext(BaseModel):
    """The independent variable lives here.

    `elapsed_seconds` is measured from entering the floor/dungeon to the moment of the
    acquisition. `chest_index_in_dive` is recorded because it is the main RIVAL
    explanation: if quality falls with elapsed time, it may simply be falling with the
    number of chests already opened. Only recording time would make the two
    indistinguishable.
    """

    dive_id: UUID
    started_at: datetime
    elapsed_seconds: int = Field(ge=0)

    # Required in practice: the client gates its Start button on a dungeon being chosen.
    dungeon_id: int | None = None
    # OPTIONAL and often null by design. Players will not keep a floor dropdown current, and
    # a stale label is worse than an honest null — so the client makes it optional and
    # sticky rather than mandatory. Note drop tables are keyed per floor, so a null floor
    # means the dungeon stratum pools several distributions; a source that gets the
    # true floor for free) is what calibrates whether that pooling is safe.
    floor_id: int | None = None

    label_source: LabelSource = LabelSource.USER_DECLARED
    # Known only when the session ends, so the uploader backfills it across the dive.
    stop_reason: StopReason | None = None

    chest_index_in_dive: int | None = Field(default=None, ge=0)


class CaptureInfo(BaseModel):
    mode: CaptureMode
    client_version: str
    # THE GAME'S LANGUAGE, NOT THE WINDOW'S. Fixed at "ja" since client 0.5.0 — the window
    # still speaks six languages (`ui_locale`, which is never sent), but the recogniser
    # reads one, so this describes which vocabulary produced the names in `contents`.
    #
    # The default was "zh_tw" until 2026-08-13, which was a live way to mislabel data: the
    # shipping client always sends this field, so the default only fired for an event that
    # omitted it — an old build's batch, a hand-rolled client — and those were stored, in a
    # NOT NULL column, as a language this client can no longer read a single name in.
    #
    # Left as `str` rather than Literal["ja"] on purpose: a stricter type would REJECT a
    # 0.4.x client's batch outright, and losing a player's records is worse than knowing
    # they came from an older vocabulary.
    locale: str = Field(default="ja", description="Game language the names were read in.")
    # Free-form QC signals (screen resolution, detector version, template set hash...).
    qc: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # WHAT THE PLAYER'S OWN GAME LOOKED LIKE WHEN THIS WAS READ. Both are self-reported and
    # neither is a measurement — see `progress_conditions.json` for what each bit means.
    #
    # ON THE EVENT, NOT ON THE PLAYER. A player-level record would hold only their latest
    # answer, and every row collected before they finished a chapter would be silently
    # re-attributed to progress they did not have at the time. The covariate has to be
    # observed WITH the reading, exactly like `client_version` beside it.
    #
    # `progress` is a flags integer: bit N is the Nth condition, OR to combine — the same
    # encoding a permissions field uses. Unlike one of those, an unset bit is not a "no": a
    # condition that did not exist when the player answered means nobody asked them, and
    # `client_version` together with the reference table is what tells the two apart.
    #
    # `character_grade` is the game's own grade id, a ladder rather than a set, so it is a
    # number rather than more bits. None means unanswered; grade 1 is a real rung.
    progress: int | None = Field(default=None, ge=0)
    character_grade: int | None = Field(default=None, ge=1)


class DropEvent(BaseModel):
    """One submission. `event_id` is the idempotency key — resubmitting is safe."""

    schema_version: int = SCHEMA_VERSION
    event_id: UUID
    install_id: UUID
    occurred_at: datetime = Field(description="UTC, client clock.")
    tz_offset_minutes: int = Field(
        ge=-24 * 60, le=24 * 60,
        description="Client local offset from UTC, for clock-sanity QC only.",
    )
    provenance: Provenance
    contents: list[ReceivedItem] = Field(
        default_factory=list,
        description=(
            "Every line the chest reported — junk, sale-only, valuable, equipment. "
            "MAY BE EMPTY: the game has its own empty-chest message "
            "(DungeonTreasure@DropEmpty, 「但是裡面什麼都沒有……」). An empty chest is a real "
            "observation and the WORST outcome, so it must be recordable — dropping such "
            "events would delete the bottom of the distribution and inflate every measured "
            "drop rate."
        ),
    )
    # The session ended while this chest was still open, so its content list may be short.
    # Recorded rather than dropped, because a chest that was never recorded and a chest that
    # never happened are indistinguishable afterwards — but the analysis must be able to
    # exclude it on evidence, which it cannot do if the flag stops at the client.
    truncated: bool = False
    dive: DiveContext | None = None
    # Present only on `mining` rows; None for every chest. See MiningContext.
    mining: MiningContext | None = None
    capture: CaptureInfo

    @field_validator("occurred_at")
    @classmethod
    def _must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware (send UTC)")
        return v

    @model_validator(mode="after")
    def _mining_block_matches_provenance(self) -> "DropEvent":
        """A mining block on a chest row, or a mining row without one, is a bug upstream.

        Rejected rather than ignored. A silently dropped block is how `label_source` spent
        months on this wire without ever reaching a column, and the cost was not the field: it
        was that every row already stored had a label whose provenance was unanswerable
        afterwards. A 422 is loud on the day it happens; a dropped field is discovered later,
        if ever.
        """
        if self.mining is not None and self.provenance is not Provenance.MINING:
            raise ValueError(
                f"mining block on a {self.provenance.value} event; it belongs to "
                f"{Provenance.MINING.value} rows only")
        if self.provenance is Provenance.MINING and self.mining is None:
            raise ValueError("a mining event must carry its mining block")
        return self

    @model_validator(mode="after")
    def _truncated_agrees_with_the_end_reason(self) -> "DropEvent":
        """`truncated` and `end_reason` must not contradict each other.

        Both are kept — `truncated` is the flag analysis applies uniformly across chests and
        veins without knowing what ended either, `end_reason` is why — and keeping both means
        they CAN disagree. A row saying the vein was mined out and also that the list is short
        is not a row anyone can interpret, and it would be interpreted anyway: every filter on
        `truncated` and every filter on `end_reason` would silently select different sets.

        Checked only on a closing row. A swing row has no `end_reason`, and can legitimately
        be truncated on its own — the session can end while a banner is still being read.
        """
        end = self.mining.end_reason if self.mining else None
        if end is None:
            return self
        incomplete = end in (VeinEnd.NO_PICKAXES, VeinEnd.GAVE_UP)
        if self.truncated != incomplete:
            raise ValueError(
                f"end_reason={end.value} means the content list is "
                f"{'short' if incomplete else 'complete'}, so truncated must be "
                f"{incomplete}, not {self.truncated}")
        return self


class IngestBatch(BaseModel):
    """Clients batch events so a dive can be uploaded in one request."""

    events: list[DropEvent] = Field(min_length=1, max_length=500)


class IngestResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    errors: list[str] = Field(default_factory=list)
    # HOW LONG THESE ROWS CAN STILL BE TAKEN BACK, said on the way in rather than discovered
    # on the way out. The client offers a Delete button on readings it was unsure of, and
    # that button has to know when it stops being able to keep its promise — otherwise a
    # player deletes a record, loses their own copy, and is told afterwards that the study
    # kept its one. Learning it from the response means the client cannot drift out of step
    # with a server that changes the setting.
    #
    # Optional, so an older server that does not send it is not an error: the client falls
    # back to its own default. See `wddrop_client.config.REMOVAL_WINDOW_SECONDS`.
    removal_window_seconds: int | None = None


class DiveClose(BaseModel):
    """How a dive ended, sent after the events it produced.

    `stop_reason` is known only when the session ends. While a dive's events are still in
    the spool the client stamps them in place, but per-record sending — the default — has
    already uploaded them by then, and they are stored with a null. This backfills those
    rows, so the outcome-dependent-stopping check StopReason exists for is possible on the
    mode most players will actually use.

    `install_id`, not `player_id`: the client cannot compute the pseudonym, and it scopes
    the update to the caller's own rows so knowing a dive_id is not enough to write to
    someone else's. It travels in the BODY for the same reason erasure does.
    """

    install_id: UUID
    dive_id: UUID
    stop_reason: StopReason


class DiveCloseResult(BaseModel):
    updated: int


class EventDelete(BaseModel):
    """Take one record back, because the client itself was unsure of the reading.

    NOT erasure, and the difference is the whole reason it is a separate call. Erasure is
    "forget me", covers everything a player ever sent, and is reversible for a week because
    the id is the only credential there is. This is one row, named by the player who recorded
    it, on the strength of having been the one looking at the screen — so it is a hard delete
    inside a bounded window rather than a mark inside a longer one.

    Which records a player is offered this on is the client's judgement and is deliberately
    narrow (`removal.why_imprecise`): an inferred quantity, an unplaceable name, a panel that
    was not finished, junk that names another dungeon. An EMPTY chest is never one of them —
    it is the worst outcome and a real observation, and deleting those one at a time would
    inflate every rate the study measures.

    `install_id`, not `player_id`: the client cannot compute the pseudonym, and it scopes the
    delete to the caller's own rows. It travels in the BODY, like erasure's does, so it never
    reaches an access log.
    """

    install_id: UUID
    event_id: UUID


class EventDeleteResult(BaseModel):
    removed: int
    window_seconds: int
