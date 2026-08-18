# wddrop-schema

The wire format shared by the **wddrop** client and server, and the one implementation of
how their versions compare.

Both sides of a drop-rate study have to agree on what a record IS. Keeping that agreement in
one installable package rather than in two copies is the whole point of this repository: two
copies drift, and a wire format that has drifted still parses — it just means something
different on each side, which no error reports.

## What is in here

`wddrop_schema.models` — the pydantic models a client uploads and a server accepts:
`IngestBatch`, `DropEvent`, `ReceivedItem`, `DiveContext`, `DiveClose`, and the enums that go
with them (`CaptureMode`, `LabelSource`, `StopReason`, `Provenance`). `SCHEMA_VERSION` says
which shape this is.

`wddrop_schema.versioning` — version comparison, used by the server to decide whether a build
is below the floor and refused, and by the client to decide whether a newer release is worth
telling the player about. A client that thinks it is current while the server thinks it is
stale shows a player a waiting count that never falls under a window that says everything is
fine, so both questions are answered by the same fifteen lines.

## Using it

```toml
[project]
dependencies = ["wddrop-schema"]

[tool.uv.sources]
wddrop-schema = { git = "https://github.com/exkuretrol/wddrop-schema", tag = "v0.1.0" }
```

Pin a tag. The version each side speaks is then a line someone changed on purpose, rather
than whatever happened to be in a sibling directory.

## Releasing

Bump `version` in `pyproject.toml`, commit, and tag it `vX.Y.Z` to match. Consumers move when
they change their pinned tag, which is a visible commit in each of them.
