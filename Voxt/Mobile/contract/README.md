# Local API contract

Example JSON of the Local API ([`Voxt/docs/LocalAPI.md`](../../docs/LocalAPI.md)), shared by both sides: `Voxt/VoxtTests/LocalAPIContractTests.swift` checks them against Voxt's real output, and Voxt Mobile's tests decode them. Run the Voxt test with `VOXT_CONTRACT_RECORD=1` to rewrite them after a schema change.

- `id`, `asrMs` and `enhanceMs` hold fixed values; real responses differ on every request.
- `compose-request.json` has every option field Voxt reads. Voxt ignores option keys it does not read, so a client may send its own (Voxt Mobile also sends `literalDictation`); compare a client's body with the fixture on the fixture's keys.
- `errors.json` maps a name to the status and `{"error": "…"}` body of one failure.
