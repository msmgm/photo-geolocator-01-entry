// ============================================================================
// PHOTO-GEOLOCATOR-01 ENTRY: COMPREHENSIVE ARCHITECTURAL OVERVIEW
// ============================================================================
// This file implements a model-free geolocation agent scaffolding for the
// make.fun photo-geolocator competition entry, featuring a clearly-marked
// heuristic fallback and an optional model-based pipeline.
//
// 1. END-TO-END EXECUTION FLOW
//    - Reads the trial brief from the MAKEFUN_PROMPT_FILE environment variable.
//    - Lists all available photos from the environment URL at MAKEFUN_ENV_URL (/photos).
//    - Filters unanswered photos and sorts them deterministically by identifier.
//    - Fetches each photo in parallel (four concurrent workers) and computes a
//      latitude and longitude estimate using decideLocation().
//    - Submits all computed answers in a single batch via POST /answers in photo order.
//
// 2. DETERMINISM AND COMPLIANCE RULES
//    - Adheres strictly to RULES.txt (live and replay runs must match byte-for-byte).
//    - Eliminates wall-clock usage entirely (no Date.now() or new Date() in payloads).
//    - Avoids unseeded randomness (no Math.random()); the trial seed is read and retained
//      for potential future seeded choices.
//    - Ensures deterministic ordering: answers are slotted by photo index and posted
//      in strict photo order regardless of asynchronous arrival times.
//    - Guarantees complete coverage: failed fetches or decoding errors receive a robust
//      default guess rather than being omitted, minimizing potential error penalties.
//
// 3. MODEL PATH CONFIGURATION & STATUS
//    - The agent is capable of calling an allowlisted vision model through the gateway,
//      but USE_MODEL is currently false, executing the code-only pixel heuristic instead.
//    - Enabling the model path requires a provider key pushed as a competition secret
//      (ANTHROPIC_API_KEY / OPENAI_API_KEY), staying well within the $1.00 trial budget.
//    - When enabled, it executes one claude-sonnet-5-5 call per photo with base64 inline
//      images, 1024 max tokens, gateway refusal finality, and 2 retries with backoff.
//
// 4. CONCURRENCY MODEL
//    - Utilizes a pooled worker pattern governed by the CONCURRENCY constant (set to 4).
//    - Workers pull tasks dynamically from an index pointer over the sorted photo list,
//      balancing throughput while respecting upstream rate limits.
//
// 5. EXCLUSIONS AND CONSTRAINTS
//    - Per competition rules, this entry carries no trained weights, no embeddings,
//      no offline lookup tables of photos and locations, and no external network access
//      beyond the test environment and gateway.
//    - Executed under the grader environment: Node 22 (`node /bot/main.ts`), using standard
//      library modules alongside the vendored kit agent SDK under vendor/.
// ============================================================================

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

/**
 * Normalizes a latitude and longitude pair by clamping to valid ranges and rounding to 6 decimals.
 */
function normalizeGuess(latOrCoord: number | { lat: number; lon: number }, lon?: number): { lat: number; lon: number } {
  const lat = typeof latOrCoord === 'object' ? latOrCoord.lat : latOrCoord;
  const l = typeof latOrCoord === 'object' ? latOrCoord.lon : (lon ?? 0);
  return {
    lat: round6(clampLat(lat)),
    lon: round6(clampLon(l)),
  };
}

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

/** Summary statistics of coarse pixel fractions across the downsampled image. */
interface PixelFractions {
  /** Fraction of sampled pixels identified as vegetation (green). */
  greenFrac: number;
  /** Fraction of sampled pixels identified as dry soil or grass. */
  dryFrac: number;
  /** Fraction of sampled pixels identified as snow cover. */
  snowFrac: number;
  /** Average brightness across sampled pixels. */
  meanBright: number;
  /** Total number of pixels sampled. */
  n: number;
}

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
/**
 * Measures coarse pixel fractions and statistics over a deterministic downsample of the decoded image.
 */
function measurePixelFractions(img: { width: number; height: number; data: Uint8Array | Uint8ClampedArray }): PixelFractions {
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
  if (n === 0) {
    return { greenFrac: 0, dryFrac: 0, snowFrac: 0, meanBright: 0, n: 0 };
  }
  return {
    greenFrac: green / n,
    dryFrac: dry / n,
    snowFrac: snow / n,
    meanBright: brightSum / n,
    n,
  };
}

/**
 * Decides the latitude band based on pixel fractions and brightness statistics.
 */
function decideLatitude(fractions: PixelFractions): number {
  const { greenFrac, dryFrac, snowFrac, meanBright } = fractions;
  if (snowFrac >= 0.2) return 58;
  if (dryFrac >= 0.4 && greenFrac < 0.15) return 23;
  if (greenFrac >= 0.35) return meanBright >= 120 ? 8 : 45;
  return DEFAULT_GUESS.lat;
}

/**
 * The code-only heuristic. It decodes the JPEG, measures pixel fractions,
 * decides the latitude band, and returns latitude and longitude guesses.
 */
function decideLocationHeuristic(jpeg: Buffer): { lat: number; lon: number } {
  let img;
  try {
    img = decodeJpeg(new Uint8Array(jpeg));
  } catch (err) {
    process.stderr.write(`entry: cannot decode a photo, using the default guess: ${(err as Error).message}\n`);
    return { ...DEFAULT_GUESS };
  }
  const fractions = measurePixelFractions(img);
  if (fractions.n === 0) return { ...DEFAULT_GUESS };
  const lat = decideLatitude(fractions);
  return { lat, lon: DEFAULT_GUESS.lon };
}

// ------------------------------------------------------------------ model path
// Not run while USE_MODEL is false. Same shape as the kit's starter, using the kit's
// vendored SDK client: one call per photo, the image inline as base64, 1024 output
// tokens, coordinates parsed the same way, any failure falling back to the heuristic.

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/**
 * Parses the first valid latitude and longitude found in a JSON object within the model text, or null.
 */
function parseCoordinatesFromJson(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: try the next object.
    }
  }
  return null;
}

/**
 * Parses the first pair of numbers in valid latitude and longitude range within the model text, or null.
 */
function parseCoordinatesFromNumbers(text: string): { lat: number; lon: number } | null {
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** The latitude and longitude an answer gives: its first JSON object with lat/lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  return parseCoordinatesFromJson(text) ?? parseCoordinatesFromNumbers(text);
}

/**
 * Builds the messages array containing the image inline as base64 and the fixed prompt text for the model.
 */
function buildModelMessages(jpeg: Buffer): Array<{ role: 'user'; content: Array<{ type: string; source?: { type: string; media_type: string; data: string }; text?: string }> }> {
  return [
    {
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
        { type: 'text', text: MODEL_PROMPT },
      ],
    },
  ];
}

/**
 * Extracts and joins text blocks from a model reply with newlines.
 */
function extractModelText(msg: { content?: Array<{ type: string; text?: string }> }): string {
  return (msg.content ?? [])
    .filter((b) => b.type === 'text')
    .map((b) => b.text ?? '')
    .join('\n');
}

/**
 * Executes a vision model call for a single photo through the Anthropic gateway.
 * 
 * Requirements & Policy:
 * - Requires ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY environment variables to be set;
 *   if missing, logs a message and falls back to the heuristic immediately.
 * - Sends the image as inline base64 data alongside the standardized prompt.
 * - Parses the response for latitude and longitude coordinates.
 * - Implements a robust retry policy: retries transient errors (such as network timeouts
 *   or rate limits defined in RETRY_STATUSES) up to a maximum of RETRIES with backoff,
 *   while gateway refusals (LlmRefusalError) and non-retryable HTTP statuses fail
 *   immediately without retrying.
 * - Returns null if the model call fails, is refused, or returns no valid coordinates,
 *   triggering a seamless fallback to the code-only heuristic.
 */
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
        messages: buildModelMessages(jpeg),
      });
      const text = extractModelText(msg);
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

/**
 * Fetches the JPEG buffer for a given photo from the environment URL.
 * Throws an error if the HTTP response is not ok.
 */
async function fetchPhoto(envUrl: string, p: Photo): Promise<Buffer> {
  const res = await fetch(`${envUrl}${p.url}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

/**
 * Filters the photo list to keep only unanswered photos and sorts them by id.
 */
function sortPhotos(photoList: Photo[]): Photo[] {
  return photoList.filter((p) => !p.answered).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
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
  const photos = sortPhotos(listed.photos);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      let jpeg: Buffer;
      try {
        jpeg = await fetchPhoto(envUrl, p);
      } catch (err) {
        process.stderr.write(`entry: cannot fetch ${p.id}, using the default guess: ${(err as Error).message}\n`);
        guesses[i] = { id: p.id, ...DEFAULT_GUESS };
        continue;
      }
      const c = await decideLocation(jpeg, p.id);
      guesses[i] = { id: p.id, ...normalizeGuess(c) };
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
