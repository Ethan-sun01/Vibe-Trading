/**
 * Options Lab — API contract types, preset strategy catalog, and pure
 * display-only helpers.
 *
 * All numeric finance math (payoff curves, Black-Scholes pricing, Greeks,
 * scenario grids) is computed by the backend (`POST /options/payoff`).
 * The ONLY client-side math living here is display-only: strike rounding,
 * percent <-> fraction conversion of user input, grid downsampling, and
 * bid/ask spread width.
 */

// ---------------------------------------------------------------------------
// API contract — POST /options/payoff
// ---------------------------------------------------------------------------

export type OptionType = "call" | "put";

/** One option leg. `qty` is a signed integer: positive = long, negative = short. */
export interface OptionLeg {
  option_type: OptionType;
  strike: number;
  qty: number;
  /** Optional per-share premium; when omitted the backend prices via Black-Scholes. */
  premium?: number;
}

export interface OptionsPayoffRequest {
  legs: OptionLeg[];
  entry_spot: number;
  expiry_days: number;
  /** Annual fraction, default 0.05. */
  risk_free_rate?: number;
  /** Annualized IV fraction, default 0.3. */
  volatility?: number;
  /** Contract multiplier, default 1. */
  multiplier?: number;
  /** Fraction in [0, 1), default 0.001. */
  commission_rate?: number;
  spot_min?: number;
  spot_max?: number;
  /** Integer 21-501, default 121. */
  spot_points?: number;
  /** IV scenarios as annualized fractions (max 9). */
  scenario_iv_values?: number[];
}

export interface BreakevenInterval {
  start: number;
  end: number | null;
}

export interface OptionsPayoffSummary {
  net_premium: number;
  entry_commission: number;
  entry_cost: number;
  entry_side: "debit" | "credit" | "flat";
  breakevens: number[];
  breakeven_intervals: BreakevenInterval[];
  max_profit: number | null;
  max_loss: number | null;
  profit_unbounded: boolean;
  loss_unbounded: boolean;
}

export interface OptionsGreeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export interface ExpiryCurve {
  spot: number[];
  pnl: number[];
}

export interface ScenarioGrid {
  iv_values: number[];
  spot: number[];
  /** One row per iv_values entry, each row aligned with spot. */
  pnl: number[][];
}

export interface OptionsPayoffResponse {
  status: "ok";
  inputs: Record<string, unknown>;
  summary: OptionsPayoffSummary;
  greeks: OptionsGreeks;
  expiry_curve: ExpiryCurve;
  scenario_grid: ScenarioGrid;
  limitations: string[];
}

// ---------------------------------------------------------------------------
// API contract — GET /options/chain
// ---------------------------------------------------------------------------

/** One contract row; any market field may be null (render "–"). */
export interface OptionsContractRow {
  contract_symbol: string;
  strike: number;
  last_price: number | null;
  bid: number | null;
  ask: number | null;
  volume: number | null;
  open_interest: number | null;
  implied_volatility: number | null;
  in_the_money: boolean;
  expiration: number;
}

export interface OptionsChainData {
  ticker: string;
  expiration: number;
  /** Epoch seconds. */
  expirations: number[];
  calls_count: number;
  puts_count: number;
  calls: OptionsContractRow[];
  puts: OptionsContractRow[];
}

export interface OptionsChainResponse {
  ok: boolean;
  market: string;
  source: string;
  data: OptionsChainData;
  /** Present on `{ok: false}` error envelopes (HTTP 400/502 throw as ApiError). */
  error?: string;
}

// ---------------------------------------------------------------------------
// UI parameter model (percent convention)
// ---------------------------------------------------------------------------

/**
 * Builder parameters as the USER types them. ONE convention, kept consistent
 * across the page: volatility, risk-free rate and commission rate are entered
 * as PERCENTS (30 means 30%) and converted to fractions only when the request
 * body is built.
 */
export interface OptionsLabParams {
  entry_spot: number;
  expiry_days: number;
  /** Percent: 30 = 30% annualized IV. Sent as 0.30. */
  volatility_pct: number;
  /** Percent: 5 = 5%. Sent as 0.05. */
  risk_free_rate_pct: number;
  /** Contract multiplier; 100 = standard US equity contract. */
  multiplier: number;
  /** Percent: 0.1 = 0.1%. Sent as 0.001. */
  commission_rate_pct: number;
}

export const DEFAULT_PARAMS: OptionsLabParams = {
  entry_spot: 100,
  expiry_days: 30,
  volatility_pct: 30,
  risk_free_rate_pct: 5,
  multiplier: 100,
  commission_rate_pct: 0.1,
};

export function pctToFraction(pct: number): number {
  return pct / 100;
}

export function fractionToPct(fraction: number): number {
  return fraction * 100;
}

/**
 * Validate inputs against the backend contract and build the request body.
 * Returns null when any input is out of contract bounds (the page then skips
 * the debounced call). Display-only guard — the backend stays authoritative.
 */
export function buildPayoffRequest(
  legs: OptionLeg[],
  p: OptionsLabParams,
): OptionsPayoffRequest | null {
  if (!Number.isFinite(p.entry_spot) || p.entry_spot <= 0) return null;
  if (!Number.isFinite(p.expiry_days) || p.expiry_days < 0) return null;
  if (!Number.isFinite(p.volatility_pct) || p.volatility_pct <= 0) return null;
  if (!Number.isFinite(p.risk_free_rate_pct)) return null;
  if (!Number.isFinite(p.multiplier) || p.multiplier <= 0) return null;
  if (!Number.isFinite(p.commission_rate_pct) || p.commission_rate_pct < 0 || p.commission_rate_pct >= 100) {
    return null;
  }
  if (legs.length === 0) return null;
  for (const leg of legs) {
    if (leg.option_type !== "call" && leg.option_type !== "put") return null;
    if (!Number.isFinite(leg.strike) || leg.strike <= 0) return null;
    if (!Number.isInteger(leg.qty) || leg.qty === 0) return null;
    if (leg.premium !== undefined && (!Number.isFinite(leg.premium) || leg.premium < 0)) return null;
  }
  return {
    legs: legs.map((l) => ({
      option_type: l.option_type,
      strike: l.strike,
      qty: l.qty,
      ...(l.premium !== undefined ? { premium: l.premium } : {}),
    })),
    entry_spot: p.entry_spot,
    expiry_days: p.expiry_days,
    risk_free_rate: pctToFraction(p.risk_free_rate_pct),
    volatility: pctToFraction(p.volatility_pct),
    multiplier: p.multiplier,
    commission_rate: pctToFraction(p.commission_rate_pct),
  };
}

// ---------------------------------------------------------------------------
// Preset strategy catalog
// ---------------------------------------------------------------------------

export interface StrategyPreset {
  id: string;
  /** i18n key of the display name (options.presets.*). */
  nameKey: string;
  /** Generate legs parameterized by the underlying spot S. */
  buildLegs: (spot: number) => OptionLeg[];
}

/** Strike grid step by price level — keeps preset strikes on listed-like increments. */
export function niceStrikeStep(price: number): number {
  if (price >= 200) return 5;
  if (price >= 50) return 2.5;
  if (price >= 20) return 1;
  if (price >= 5) return 0.5;
  return 0.25;
}

/** Round a raw price to a "nice" listed-style strike. Returns 0 for non-positive input. */
export function roundToNiceStrike(price: number): number {
  if (!Number.isFinite(price) || price <= 0) return 0;
  const step = niceStrikeStep(price);
  const rounded = Math.max(step, Math.round(price / step) * step);
  return Number(rounded.toFixed(4));
}

function k(spot: number, factor: number): number {
  return roundToNiceStrike(spot * factor);
}

/**
 * Twelve canonical presets, parameterized by spot S. Strike ratios follow the
 * S x {0.9, 0.95, 1.0, 1.05, 1.1} grid (rounded to nice strikes).
 */
export const STRATEGY_PRESETS: StrategyPreset[] = [
  {
    id: "long_call",
    nameKey: "options.presets.longCall",
    buildLegs: (s) => [{ option_type: "call", strike: k(s, 1), qty: 1 }],
  },
  {
    id: "long_put",
    nameKey: "options.presets.longPut",
    buildLegs: (s) => [{ option_type: "put", strike: k(s, 1), qty: 1 }],
  },
  {
    id: "bull_call_spread",
    nameKey: "options.presets.bullCallSpread",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 0.95), qty: 1 },
      { option_type: "call", strike: k(s, 1.05), qty: -1 },
    ],
  },
  {
    id: "bear_put_spread",
    nameKey: "options.presets.bearPutSpread",
    buildLegs: (s) => [
      { option_type: "put", strike: k(s, 1.05), qty: 1 },
      { option_type: "put", strike: k(s, 0.95), qty: -1 },
    ],
  },
  {
    id: "bull_put_spread",
    nameKey: "options.presets.bullPutSpread",
    buildLegs: (s) => [
      { option_type: "put", strike: k(s, 1.05), qty: -1 },
      { option_type: "put", strike: k(s, 0.95), qty: 1 },
    ],
  },
  {
    id: "bear_call_spread",
    nameKey: "options.presets.bearCallSpread",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 0.95), qty: -1 },
      { option_type: "call", strike: k(s, 1.05), qty: 1 },
    ],
  },
  {
    id: "long_straddle",
    nameKey: "options.presets.longStraddle",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 1), qty: 1 },
      { option_type: "put", strike: k(s, 1), qty: 1 },
    ],
  },
  {
    id: "long_strangle",
    nameKey: "options.presets.longStrangle",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 1.05), qty: 1 },
      { option_type: "put", strike: k(s, 0.95), qty: 1 },
    ],
  },
  {
    id: "iron_condor",
    nameKey: "options.presets.ironCondor",
    buildLegs: (s) => [
      { option_type: "put", strike: k(s, 0.9), qty: 1 },
      { option_type: "put", strike: k(s, 0.95), qty: -1 },
      { option_type: "call", strike: k(s, 1.05), qty: -1 },
      { option_type: "call", strike: k(s, 1.1), qty: 1 },
    ],
  },
  {
    id: "iron_butterfly",
    nameKey: "options.presets.ironButterfly",
    buildLegs: (s) => [
      { option_type: "put", strike: k(s, 1), qty: -1 },
      { option_type: "call", strike: k(s, 1), qty: -1 },
      { option_type: "put", strike: k(s, 0.9), qty: 1 },
      { option_type: "call", strike: k(s, 1.1), qty: 1 },
    ],
  },
  {
    id: "call_butterfly",
    nameKey: "options.presets.callButterfly",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 0.9), qty: 1 },
      { option_type: "call", strike: k(s, 1), qty: -2 },
      { option_type: "call", strike: k(s, 1.1), qty: 1 },
    ],
  },
  {
    id: "ratio_call_spread",
    nameKey: "options.presets.ratioCallSpread",
    buildLegs: (s) => [
      { option_type: "call", strike: k(s, 1), qty: 1 },
      { option_type: "call", strike: k(s, 1.1), qty: -2 },
    ],
  },
];

export function getPreset(id: string): StrategyPreset | undefined {
  return STRATEGY_PRESETS.find((p) => p.id === id);
}

export function buildPresetLegs(presetId: string, spot: number): OptionLeg[] | null {
  const preset = getPreset(presetId);
  if (!preset || !Number.isFinite(spot) || spot <= 0) return null;
  return preset.buildLegs(spot);
}

// ---------------------------------------------------------------------------
// Display-only helpers (chain table, heatmap downsampling)
// ---------------------------------------------------------------------------

/**
 * Even-stride index selection that ALWAYS keeps the first and last element.
 * Used to downsample the scenario spot grid for heatmap display.
 */
export function downsampleIndices(total: number, maxPoints: number): number[] {
  if (total <= 0) return [];
  if (total <= maxPoints || maxPoints < 2) {
    return Array.from({ length: total }, (_, i) => i);
  }
  const stride = (total - 1) / (maxPoints - 1);
  const indices: number[] = [];
  for (let i = 0; i < maxPoints; i++) {
    indices.push(Math.round(i * stride));
  }
  indices[indices.length - 1] = total - 1;
  return indices;
}

/** Bid/ask spread width as a percent of mid. Null when uncomputable. */
export function bidAskSpreadPct(bid: number | null, ask: number | null): number | null {
  if (bid === null || ask === null || !Number.isFinite(bid) || !Number.isFinite(ask)) return null;
  const mid = (bid + ask) / 2;
  if (mid <= 0) return null;
  return ((ask - bid) / mid) * 100;
}

/** Index of the row whose strike is nearest to the reference spot (-1 when empty). */
export function nearestStrikeIndex(rows: OptionsContractRow[], spot: number): number {
  if (rows.length === 0 || !Number.isFinite(spot)) return -1;
  let best = 0;
  let bestDist = Math.abs(rows[0].strike - spot);
  for (let i = 1; i < rows.length; i++) {
    const dist = Math.abs(rows[i].strike - spot);
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
}

/** Locale date label for an epoch-seconds expiration (e.g. "Aug 15, 2026"). */
export function expirationLabel(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Format an IV fraction (0.30) as a percent label ("30%"). */
export function ivLabel(iv: number): string {
  const pct = fractionToPct(iv);
  return `${Number(pct.toFixed(2)).toLocaleString()}%`;
}

// ---------------------------------------------------------------------------
// API contract — vol surface (POST /options/surface, /options/surface/synthetic)
// ---------------------------------------------------------------------------

/**
 * One observed (strike, expiry, IV) point fed to the surface fitter.
 * `implied_volatility` is an annualised FRACTION (0.25 = 25%).
 */
export interface SurfaceQuote {
  strike: number;
  expiry_days: number;
  implied_volatility: number;
}

export interface SurfaceSviParams {
  a: number;
  b: number;
  rho: number;
  m0: number;
  sigma: number;
}

/** Fit diagnostics for one expiry of the surface. */
export interface SurfaceExpiryFit {
  expiry_days: number;
  n_quotes: number;
  rmse: number;
  max_residual: number;
  wings_ok: boolean;
  params: SurfaceSviParams;
}

export interface VolSurfaceGrid {
  strikes: number[];
  /** Expiration days, one per row of `iv`. */
  expiries: number[];
  /** One row per expiry, each aligned with `strikes`. Annualised fractions. */
  iv: number[][];
}

/** Surface-consistent call Greeks across the strike grid for one expiry. */
export interface SurfaceGreeksGrid {
  expiry_days: number;
  strikes: number[];
  delta: number[];
  gamma: number[];
  theta: number[];
  vega: number[];
  rho: number[];
}

export interface VolSurfaceResponse {
  status: "ok";
  spot: number;
  risk_free_rate: number;
  dividend_yield: number;
  expiries: SurfaceExpiryFit[];
  grid: VolSurfaceGrid;
  greeks: SurfaceGreeksGrid | null;
  limitations: string[];
}

export interface VolSurfaceQuotesRequest {
  spot: number;
  quotes: SurfaceQuote[];
  risk_free_rate: number;
  dividend_yield: number;
  strike_min?: number;
  strike_max?: number;
  strike_points: number;
  expiries?: number[];
  greeks_expiry_days?: number;
}

export interface SyntheticSurfaceRequest {
  spot: number;
  expiry_days: number[];
  strike_min: number;
  strike_max: number;
  strike_points: number;
  atm_iv: number;
  /** Log-moneyness linear coefficient (fraction, e.g. -0.2). */
  skew: number;
  /** Log-moneyness quadratic coefficient (fraction, e.g. 0.1). */
  curvature: number;
  risk_free_rate: number;
  dividend_yield: number;
  greeks_expiry_days?: number;
}

// ---------------------------------------------------------------------------
// Vol-surface UI parameter model (percent convention)
// ---------------------------------------------------------------------------

/**
 * Surface builder parameters as the USER types them: ATM IV, skew, curvature
 * and strike range are entered as PERCENTS and converted to fractions only
 * when the request body is built, matching the payoff builder convention.
 */
export interface SurfaceLabParams {
  spot: number;
  /** Percent: 25 = 25% annualised ATM IV. Sent as 0.25. */
  atm_iv_pct: number;
  /** Percent: -20 = -0.2 log-moneyness linear coefficient. */
  skew_pct: number;
  /** Percent: 10 = 0.1 log-moneyness quadratic coefficient. */
  curvature_pct: number;
  /** Percent band around spot covered by the strike grid (±30 = 70…130). */
  strike_range_pct: number;
  /** Expiration days of the fitted smiles. */
  expiries: number[];
  /** Expiry (days) whose Greeks curves are computed; null disables them. */
  greeks_expiry_days: number | null;
  /** Strike-grid resolution. */
  strike_points: number;
}

export const DEFAULT_SURFACE_PARAMS: SurfaceLabParams = {
  spot: 100,
  atm_iv_pct: 25,
  skew_pct: -20,
  curvature_pct: 10,
  strike_range_pct: 30,
  expiries: [30, 60, 90, 180],
  greeks_expiry_days: 60,
  strike_points: 41,
};

/**
 * Validate surface inputs against the backend contract and build the
 * synthetic request. Returns null when any input is out of bounds (the page
 * then skips the debounced call). Display-only guard — the backend stays
 * authoritative.
 */
export function buildSyntheticSurfaceRequest(
  p: SurfaceLabParams,
): SyntheticSurfaceRequest | null {
  if (!Number.isFinite(p.spot) || p.spot <= 0) return null;
  if (!Number.isFinite(p.atm_iv_pct) || p.atm_iv_pct <= 0) return null;
  if (!Number.isFinite(p.strike_range_pct) || p.strike_range_pct <= 0 || p.strike_range_pct >= 100) {
    return null;
  }
  if (!Number.isFinite(p.skew_pct)) return null;
  if (!Number.isFinite(p.curvature_pct)) return null;
  if (!Number.isInteger(p.strike_points) || p.strike_points < 5 || p.strike_points > 101) return null;
  if (p.expiries.length === 0) return null;
  for (const d of p.expiries) {
    if (!Number.isFinite(d) || d <= 0) return null;
  }
  if (p.greeks_expiry_days !== null && (!Number.isFinite(p.greeks_expiry_days) || p.greeks_expiry_days <= 0)) {
    return null;
  }
  const band = p.strike_range_pct / 100;
  return {
    spot: p.spot,
    expiry_days: [...p.expiries].sort((a, b) => a - b),
    strike_min: Number((p.spot * (1 - band)).toFixed(4)),
    strike_max: Number((p.spot * (1 + band)).toFixed(4)),
    strike_points: p.strike_points,
    atm_iv: pctToFraction(p.atm_iv_pct),
    skew: pctToFraction(p.skew_pct),
    curvature: pctToFraction(p.curvature_pct),
    risk_free_rate: 0.05,
    dividend_yield: 0,
    ...(p.greeks_expiry_days !== null ? { greeks_expiry_days: p.greeks_expiry_days } : {}),
  };
}

/**
 * Build the quotes-mode request. `quotes` must already be validated
 * (see {@link surfaceQuotesValid}); returns null on a bad spot or empty set.
 */
export function buildSurfaceQuotesRequest(
  spot: number,
  quotes: SurfaceQuote[],
  opts: { greeks_expiry_days?: number | null; strike_points?: number } = {},
): VolSurfaceQuotesRequest | null {
  if (!Number.isFinite(spot) || spot <= 0) return null;
  if (quotes.length < 4) return null;
  const strike_points = opts.strike_points ?? 41;
  if (!Number.isInteger(strike_points) || strike_points < 5 || strike_points > 101) return null;
  return {
    spot,
    quotes,
    risk_free_rate: 0.05,
    dividend_yield: 0,
    strike_points,
    ...(opts.greeks_expiry_days !== null && opts.greeks_expiry_days !== undefined
      ? { greeks_expiry_days: opts.greeks_expiry_days }
      : {}),
  };
}

/** True when a quote set has enough valid points to fit a surface. */
export function surfaceQuotesValid(quotes: SurfaceQuote[]): boolean {
  return (
    quotes.length >= 4 &&
    quotes.every(
      (q) =>
        Number.isFinite(q.strike) && q.strike > 0 &&
        Number.isFinite(q.expiry_days) && q.expiry_days > 0 &&
        Number.isFinite(q.implied_volatility) && q.implied_volatility > 0,
    )
  );
}

export interface SurfaceCsvParseResult {
  quotes: SurfaceQuote[];
  errors: string[];
}

/**
 * Parse "strike,expiry_days,iv_fraction" CSV lines (one per row) into surface
 * quotes. Blank lines and lines starting with `#` are ignored; a malformed row
 * is recorded in `errors` and skipped rather than failing the whole paste, so
 * one bad line cannot kill a 50-row chain export.
 */
export function parseSurfaceQuotesCsv(text: string): SurfaceCsvParseResult {
  const quotes: SurfaceQuote[] = [];
  const errors: string[] = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (line === "" || line.startsWith("#")) return;
    const parts = line.split(",").map((s) => s.trim());
    if (parts.length !== 3) {
      errors.push(`${i + 1}: expected "strike,expiry_days,iv", got ${parts.length} column(s)`);
      return;
    }
    const strike = Number(parts[0]);
    const expiryDays = Number(parts[1]);
    const iv = Number(parts[2]);
    if (!Number.isFinite(strike) || strike <= 0) {
      errors.push(`${i + 1}: strike must be a positive number`);
      return;
    }
    if (!Number.isFinite(expiryDays) || expiryDays <= 0) {
      errors.push(`${i + 1}: expiry_days must be a positive number`);
      return;
    }
    if (!Number.isFinite(iv) || iv <= 0) {
      errors.push(`${i + 1}: implied_volatility must be a positive fraction like 0.25`);
      return;
    }
    quotes.push({ strike, expiry_days: expiryDays, implied_volatility: iv });
  });
  return { quotes, errors };
}

/** Default expiry choices offered by the surface panel (calendar days). */
export const SURFACE_EXPIRY_OPTIONS = [30, 60, 90, 180, 365];
