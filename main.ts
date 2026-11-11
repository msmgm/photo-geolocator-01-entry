// photo-geolocator-01 entry (scaffolding): a model-free agent with a clearly-marked
// geolocation placeholder.
//
// What it does, end to end:
//   1. reads the brief from MAKEFUN_PROMPT_FILE (the one file the trial hands it);
//   2. lists the 25 photos from the env at MAKEFUN_ENV_URL (GET /photos);
//   3. fetches each photo (GET /photos/<id>.jpg, four at a time) and decides a latitude
//      and longitude for it with decideLocation() below;
//   4. posts every answer in one POST /answers, in photo order, and exits.
//
// Determinism (RULES.txt: the trial runs twice, live then replay, and the replay must
// leave the env byte-for-byte as the live pass left it):
//   - no wall clock anywhere in a request (no Date.now(), no new Date() in a payload);
//   - no unseeded randomness (no Math.random()); the trial seed is read and kept for any
//     future seeded choice, but this version needs none: the heuristic is a pure function
//     of the photo's pixels;
//   - photos are fetched in parallel, but every answer is slotted by photo index and the
//     single POST goes out in photo order, so nothing depends on arrival order;
//   - a photo that fails to fetch or decode still gets an answer (the default guess):
//     an unanswered photo counts the maximum error (20,015 km), a default guess is
//     always better than that.
//
// Model calls: the agent MAY call an allowlisted vision model through the gateway, but
// this build does not. USE_MODEL is false and decideLocation() runs the code-only
// heuristic. Flipping it on requires a provider key pushed to the competition as a
// secret (ANTHROPIC_API_KEY / OPENAI_API_KEY) — the house pays for none of an entry's
// model calls, at most $1.00 a trial. The model path below is written, not run: one
// claude-sonnet-5-5 call per photo, the image inline as base64 (never by URL), at most
// 1024 output tokens, gateway refusals final (fall back to the heuristic), transient
// errors retried twice with a fixed backoff. It is the same shape as the kit's starter,
// so it fits the trial's budget and request cap when a key exists.
//
// What this entry does NOT carry, per the rules: no trained weights, no embeddings, no
// table of photos and their places, no network beyond the env and the gateway.
//
// Run as the grader runs it: `node /bot/main.ts` on Node 22, standard library only
// (plus the kit's agent SDK vendored under vendor/, byte for byte).

import { readFileSync } from 'node:fs';
import { Anthropic } from './vendor/makefun-agent.ts';
import { decodeJpeg } from './vendor/jpeg-decoder.ts';

// ---------------------------------------------------------------- configuration

/** PLACEHOLDER SWITCH: keep false until a provider key is pushed as a secret. */
const USE_MODEL = false;
const MODEL = 'claude-sonnet-5-5';
const MODEL_MAX_TOKENS = 1024;
const MODEL_PROMPT =
  'Where was this photo taken? Give your best estimate of the latitude and longitude, as one line of JSON: {"lat": <degrees north>, "lon": <degrees east>}.';
/** How many photos to fetch / ask about at once (the starter uses 4). */
const CONCURRENCY = 4;
/** A gateway refusal (4xx) is final; only these statuses are retried, twice. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 2;

/** One photo's answer. */
interface Guess {
  id: string;
  lat: number;
  lon: number;
}

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const round6 = (v: number): number => Math.round(v * 1e6) / 1e6;
const clampLat = (v: number): number => Math.min(90, Math.max(-90, v));
const clampLon = (v: number): number => Math.min(180, Math.max(-180, v));

// ------------------------------------------------- THE GEOLOCATION DECISION
// PLACEHOLDER — this is the function the final agent improves. Its contract:
//   decideLocation(jpegBytes, photoId) -> { lat, lon }, both in decimal degrees.
// The current implementation is a conservative code-only heuristic (see below). It is
// deliberately weak and makes no accuracy claim: without a vision model it can only read
// coarse pixel statistics, and longitude has no pixel signal at all, so it stays fixed.
// A model version keeps this exact signature: same inputs, same output, same
// determinism (every request a function of the photo bytes and the fixed prompt).

/**
 * The default guess, used when a photo cannot be decoded and as the base of the
 * heuristic: the land / population centroid of the Old World (30N, 45E), the middle of
 * the landmass where most OpenStreetView-5M photos come from. One fixed point is the
 * honest floor: any single guess is thousands of kilometres off most photos, and this
 * one at least sits in the middle of the map instead of an ocean.
 */
const DEFAULT_GUESS: { lat: number; lon: number } = { lat: 30, lon: 45 };

/**
 * The code-only heuristic. It decodes the JPEG and reads four coarse pixel fractions
 * over a deterministic downsample (every pixel at a fixed stride, so the result depends
 * only on the bytes):
 *   - green: vegetation (G clearly above R and B) — tropics and temperate zones;
 *   - dry: tan/brown soil and dry grass (R above G above B) — arid belts;
 *   - snow: near-white pixels — snow cover, high latitudes or alpine ground;
 *   - bright: mean brightness, to tell a bright vegetated scene (tropics) from a dim
 *     one (temperate).
 * The latitude follows, in this priority order (first match wins): snow -> 58,
 * arid -> 23, vegetated and bright -> 8, vegetated and dim -> 45, anything else ->
 * the default 30. Longitude has no pixel signal, so it stays at the default 45.
 * These bands are coarse climate belts, not places: the heuristic cannot name a
 * country, let alone a street, and its error will be in the thousands of kilometres.
 */
function decideLocationHeuristic(jpeg: Buffer): { lat: number; lon: number } {
  let img;
  try {
    img = decodeJpeg(new Uint8Array(jpeg));
  } catch (err) {
    process.stderr.write(`entry: cannot decode a photo, using the default guess: ${(err as Error).message}\n`);
    return { ...DEFAULT_GUESS };
  }
  const { width, height, data } = img;
  const stepX = Math.max(1, Math.floor(width / 64));
  const stepY = Math.max(1, Math.floor(height / 36));
  let n = 0;
  let green = 0;
  let dry = 0;
  let snow = 0;
  let brightSum = 0;
  for (let y = 0; y < height; y += stepY) {
    for (let x = 0; x < width; x += stepX) {
      const o = (y * width + x) * 3;
      const r = data[o]!;
      const g = data[o + 1]!;
      const b = data[o + 2]!;
      n += 1;
      if (g > r + 12 && g > b + 12 && g > 70) green += 1;
      if (r > g + 8 && g >= b && r - b > 25) dry += 1;
      if (r > 205 && g > 205 && b > 205) snow += 1;
      brightSum += (r + g + b) / 3;
    }
  }
  if (n === 0) return { ...DEFAULT_GUESS };
  const greenFrac = green / n;
  const dryFrac = dry / n;
  const snowFrac = snow / n;
  const meanBright = brightSum / n;
  let lat: number;
  if (snowFrac >= 0.2) lat = 58;
  else if (dryFrac >= 0.4 && greenFrac < 0.15) lat = 23;
  else if (greenFrac >= 0.35) lat = meanBright >= 120 ? 8 : 45;
  else lat = DEFAULT_GUESS.lat;
  return { lat, lon: DEFAULT_GUESS.lon };
}

// ------------------------------------------------------------------ model path
// Not run while USE_MODEL is false. Same shape as the kit's starter, using the kit's
// vendored SDK client: one call per photo, the image inline as base64, 1024 output
// tokens, coordinates parsed the same way, any failure falling back to the heuristic.

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat/lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next object, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** One model call for one photo; null when the gateway refused it or it gave no coordinates (the heuristic runs instead). */
async function decideLocationWithModel(jpeg: Buffer, id: string): Promise<{ lat: number; lon: number } | null> {
  const baseUrl = process.env.ANTHROPIC_BASE_URL;
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (baseUrl === undefined || baseUrl === '' || apiKey === undefined || apiKey === '') {
    process.stderr.write(`entry: no Anthropic gateway key for ${id}, using the heuristic\n`);
    return null;
  }
  const llm = new Anthropic({ baseUrl: baseUrl.replace(/\/+$/, ''), apiKey });
  for (let attempt = 0; ; attempt += 1) {
    try {
      const msg = await llm.messages({
        model: MODEL,
        max_tokens: MODEL_MAX_TOKENS,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
              { type: 'text', text: MODEL_PROMPT },
            ],
          },
        ],
      });
      const text = (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
      const c = coordinates(text);
      if (c === null) {
        process.stderr.write(`entry: no coordinates in the model answer for ${id}, using the heuristic\n`);
        return null;
      }
      return c;
    } catch (err) {
      const status = (err as { status?: number }).status;
      // A gateway refusal (LlmRefusalError) or a non-retryable status is final.
      if (err instanceof Error && err.name === 'LlmRefusalError') {
        process.stderr.write(`entry: the model call was refused for ${id} (${err.message}), using the heuristic\n`);
        return null;
      }
      if (status !== undefined && !RETRY_STATUSES.includes(status)) {
        process.stderr.write(`entry: the model call failed for ${id} (HTTP ${status}), using the heuristic\n`);
        return null;
      }
      if (attempt >= RETRIES) {
        process.stderr.write(`entry: the model call failed for ${id} after ${RETRIES} retries, using the heuristic\n`);
        return null;
      }
      await sleep(2000 * (attempt + 1));
    }
  }
}

/** The geolocation decision for one photo: the model when enabled, else the heuristic. */
async function decideLocation(jpeg: Buffer, id: string): Promise<{ lat: number; lon: number }> {
  if (USE_MODEL) {
    const fromModel = await decideLocationWithModel(jpeg, id);
    if (fromModel !== null) return fromModel;
  }
  return decideLocationHeuristic(jpeg);
}

// ------------------------------------------------------------------------ main

async function getJson(url: string): Promise<unknown> {
  const res = await fetch(url, { headers: { accept: 'application/json' } });
  if (!res.ok) throw new Error(`GET ${url} answered HTTP ${res.status}`);
  return (await res.json()) as unknown;
}

async function main(): Promise<number> {
  // The brief: the one file the trial hands the entry. Read, not parsed for decisions
  // here (the model prompt above is fixed); its first line is logged for transparency.
  const promptFile = need('MAKEFUN_PROMPT_FILE');
  const prompt = readFileSync(promptFile, 'utf-8');
  process.stderr.write(`entry: brief ${promptFile}: ${(prompt.split('\n')[0] ?? '').slice(0, 120)}\n`);
  const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
  // The trial seed: read for the record. This version draws no randomness from it; any
  // future seeded choice must use it (never Math.random() or a clock).
  const seed = process.env.MAKEFUN_TRIAL_SEED ?? 'photo-geolocator-01';
  process.stderr.write(`entry: trial seed ${seed.slice(0, 16)}… (no randomness drawn from it in this build)\n`);

  const listed = (await getJson(`${envUrl}/photos`)) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      let jpeg: Buffer;
      try {
        const res = await fetch(`${envUrl}${p.url}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        jpeg = Buffer.from(await res.arrayBuffer());
      } catch (err) {
        process.stderr.write(`entry: cannot fetch ${p.id}, using the default guess: ${(err as Error).message}\n`);
        guesses[i] = { id: p.id, ...DEFAULT_GUESS };
        continue;
      }
      const c = await decideLocation(jpeg, p.id);
      guesses[i] = { id: p.id, lat: round6(clampLat(c.lat)), lon: round6(clampLon(c.lon)) };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) {
    process.stderr.write('entry: no answers to post\n');
    return 1;
  }
  // One POST, in photo order. A 409 means the answers are already recorded.
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${envUrl}/answers`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ answers }),
      });
    } catch (err) {
      if (attempt >= 1) throw err;
      await sleep(1000);
      continue;
    }
    const text = await res.text();
    process.stdout.write(`entry: ${answers.length} of ${photos.length} photos answered: HTTP ${res.status} ${text}\n`);
    if (res.ok || res.status === 409) return 0;
    if (res.status >= 500 && attempt < 1) {
      await sleep(1000);
      continue;
    }
    return 1;
  }
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`entry: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
