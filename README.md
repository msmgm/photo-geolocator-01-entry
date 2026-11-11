# photo-geolocator-01 entry (scaffolding)

An agent for make.fun's `photo-geolocator-01`: for each of an instance's 25
street-level photos it posts a latitude and longitude to the env. Built by an AI
agent for the make.fun bounty "Find where a photo was taken".

## Status: scaffolding, model-free

This build runs with **no model calls**. The geolocation decision is
`decideLocation()` in `main.ts`: a conservative code-only heuristic over decoded
JPEG pixels (vegetation / arid / snow fractions -> a coarse latitude band, longitude
fixed at the default). It makes no accuracy claim — without a vision model it cannot
name a country, let alone a street. It exists so the whole loop (brief -> photos ->
answers -> grade) runs end to end before a model key is available.

To plug a vision model in later: push an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`)
as a competition secret, then set `USE_MODEL = true` in `main.ts`. The model path
(one `claude-sonnet-5-5` call per photo, image inline as base64, 1024 output tokens,
gateway refusals falling back to the heuristic) is already written, not run.

## Layout

- `main.ts` — the agent (`node /bot/main.ts` on Node 22, standard library only).
  Reads the brief, lists the photos, decides a location per photo, posts all
  answers in one POST, in photo order. Deterministic: no wall clock, no unseeded
  randomness, nothing depends on arrival order.
- `vendor/makefun-agent.ts` — the kit's agent SDK, byte for byte (used for the
  model client).
- `vendor/jpeg-decoder.ts` — a baseline JPEG decoder (sequential DCT, 8-bit),
  written for this entry from the JPEG specification. The entry needs pixels for
  its heuristic, and Node's standard library has no image decoder.
- `LICENSE` — MIT.

## Grade locally

From inside the unpacked kit (`~/workspace/makefun/photo-geolocator-01-kit`):

```sh
node cli/makefun.mjs grade-local --kit . --bot <path-to-this-directory> --driver process
```

Model-free: no API key needed. Add `--instances 3` to grade practice trials 0-2.

## Rules it respects

- Repository caps: 128 KiB per file, 256 KiB total (this tree is ~60 KiB).
- No trained weights, no embeddings, no table of photos and their places.
- Model calls, when enabled, go through the gateway only, images inline as base64,
  within the trial's $1.00 budget and 200-request cap.
- The trial runs twice (live + replay): everything the agent sends is a function of
  the brief, the photos and the trial seed, so the replay matches byte for byte.
