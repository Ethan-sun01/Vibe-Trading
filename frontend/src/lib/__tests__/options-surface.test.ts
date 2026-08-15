import { describe, expect, it } from "vitest";
import {
  buildSurfaceQuotesRequest,
  buildSyntheticSurfaceRequest,
  DEFAULT_SURFACE_PARAMS,
  parseSurfaceQuotesCsv,
  surfaceQuotesValid,
  type SurfaceLabParams,
} from "@/lib/options";

const VALID_PARAMS: SurfaceLabParams = {
  spot: 100,
  atm_iv_pct: 25,
  skew_pct: -20,
  curvature_pct: 10,
  strike_range_pct: 30,
  expiries: [90, 30, 60],
  greeks_expiry_days: 60,
  strike_points: 41,
};

describe("buildSyntheticSurfaceRequest", () => {
  it("converts percent inputs to fractions and sorts expiries", () => {
    const req = buildSyntheticSurfaceRequest(VALID_PARAMS);
    expect(req).not.toBeNull();
    expect(req!.spot).toBe(100);
    expect(req!.atm_iv).toBeCloseTo(0.25);
    expect(req!.skew).toBeCloseTo(-0.2);
    expect(req!.curvature).toBeCloseTo(0.1);
    expect(req!.expiry_days).toEqual([30, 60, 90]);
    expect(req!.strike_min).toBeCloseTo(70);
    expect(req!.strike_max).toBeCloseTo(130);
    expect(req!.greeks_expiry_days).toBe(60);
    expect(req!.risk_free_rate).toBe(0.05);
  });

  it("omits greeks_expiry_days when disabled", () => {
    const req = buildSyntheticSurfaceRequest({
      ...VALID_PARAMS,
      greeks_expiry_days: null,
    });
    expect(req).not.toBeNull();
    expect(req!.greeks_expiry_days).toBeUndefined();
  });

  it("accepts a non-30% strike band", () => {
    const req = buildSyntheticSurfaceRequest({
      ...VALID_PARAMS,
      strike_range_pct: 20,
    });
    expect(req!.strike_min).toBeCloseTo(80);
    expect(req!.strike_max).toBeCloseTo(120);
  });

  it.each([
    ["non-positive spot", { ...VALID_PARAMS, spot: 0 }],
    ["non-positive ATM IV", { ...VALID_PARAMS, atm_iv_pct: 0 }],
    ["strike band too wide", { ...VALID_PARAMS, strike_range_pct: 100 }],
    ["strike points below floor", { ...VALID_PARAMS, strike_points: 4 }],
    ["strike points above cap", { ...VALID_PARAMS, strike_points: 102 }],
    ["empty expiries", { ...VALID_PARAMS, expiries: [] }],
    ["negative expiry", { ...VALID_PARAMS, expiries: [30, -5] }],
    ["non-positive greeks expiry", { ...VALID_PARAMS, greeks_expiry_days: 0 }],
  ])("rejects %s", (_label, params) => {
    expect(buildSyntheticSurfaceRequest(params as SurfaceLabParams)).toBeNull();
  });

  it("keeps defaults buildable", () => {
    expect(buildSyntheticSurfaceRequest(DEFAULT_SURFACE_PARAMS)).not.toBeNull();
  });
});

describe("parseSurfaceQuotesCsv", () => {
  it("parses valid rows and ignores blanks and comments", () => {
    const text = [
      "# header comment",
      "85,30,0.31",
      "",
      "  100 , 30 , 0.25  ",
      "115,30,0.29",
    ].join("\n");
    const { quotes, errors } = parseSurfaceQuotesCsv(text);
    expect(errors).toEqual([]);
    expect(quotes).toEqual([
      { strike: 85, expiry_days: 30, implied_volatility: 0.31 },
      { strike: 100, expiry_days: 30, implied_volatility: 0.25 },
      { strike: 115, expiry_days: 30, implied_volatility: 0.29 },
    ]);
  });

  it("records per-line errors and keeps the valid rows", () => {
    const { quotes, errors } = parseSurfaceQuotesCsv(
      ["85,30,0.31", "garbage", "100,30", "115,30,0.29", "0,30,0.2"].join("\n"),
    );
    expect(quotes.length).toBe(2);
    expect(errors.length).toBe(3);
    // Row 2 is "garbage" (1 column); row 3 is "100,30" (2 columns).
    expect(errors[0]).toContain("2:");
    expect(errors[0]).toContain("1 column");
    expect(errors[1]).toContain("3:");
    expect(errors[2]).toContain("5:");
  });

  it("rejects non-finite and non-positive values", () => {
    const { quotes, errors } = parseSurfaceQuotesCsv(
      ["85,30,NaN", "-5,30,0.2", "100,-30,0.2", "100,30,-0.2"].join("\n"),
    );
    expect(quotes).toEqual([]);
    expect(errors.length).toBe(4);
  });

  it("returns empty result for empty text", () => {
    const { quotes, errors } = parseSurfaceQuotesCsv("   \n# only a comment\n");
    expect(quotes).toEqual([]);
    expect(errors).toEqual([]);
  });
});

describe("surfaceQuotesValid / buildSurfaceQuotesRequest", () => {
  const QUOTES = [
    { strike: 85, expiry_days: 30, implied_volatility: 0.31 },
    { strike: 100, expiry_days: 30, implied_volatility: 0.25 },
    { strike: 115, expiry_days: 30, implied_volatility: 0.29 },
    { strike: 85, expiry_days: 60, implied_volatility: 0.33 },
  ];

  it("accepts a four-quote set", () => {
    expect(surfaceQuotesValid(QUOTES)).toBe(true);
    const req = buildSurfaceQuotesRequest(100, QUOTES, { greeks_expiry_days: 30 });
    expect(req).not.toBeNull();
    expect(req!.quotes).toEqual(QUOTES);
    expect(req!.greeks_expiry_days).toBe(30);
  });

  it("rejects fewer than four quotes", () => {
    expect(surfaceQuotesValid(QUOTES.slice(0, 3))).toBe(false);
    expect(buildSurfaceQuotesRequest(100, QUOTES.slice(0, 3))).toBeNull();
  });

  it("rejects a non-finite IV", () => {
    const bad = [
      ...QUOTES.slice(0, 3),
      { strike: 100, expiry_days: 30, implied_volatility: Number.NaN },
    ];
    expect(surfaceQuotesValid(bad)).toBe(false);
  });

  it("rejects a non-positive spot", () => {
    expect(buildSurfaceQuotesRequest(0, QUOTES)).toBeNull();
  });
});
