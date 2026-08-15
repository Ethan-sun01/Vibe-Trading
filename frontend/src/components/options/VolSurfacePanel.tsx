import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Activity, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import {
  buildSurfaceQuotesRequest,
  buildSyntheticSurfaceRequest,
  DEFAULT_SURFACE_PARAMS,
  parseSurfaceQuotesCsv,
  SURFACE_EXPIRY_OPTIONS,
  surfaceQuotesValid,
  type SurfaceLabParams,
  type SurfaceQuote,
  type SyntheticSurfaceRequest,
  type VolSurfaceQuotesRequest,
  type VolSurfaceResponse,
} from "@/lib/options";
import { VolSurfaceHeatmap } from "@/components/charts/VolSurfaceHeatmap";
import { GreeksCurvesChart } from "@/components/charts/GreeksCurvesChart";

const ANALYZE_DEBOUNCE_MS = 500;

type SurfaceMode = "synthetic" | "quotes" | "chain";

const INPUT_CLS =
  "w-full rounded-md border border-border/60 bg-background px-2 py-1.5 text-sm outline-none focus:ring-2 focus:ring-primary/40";

const MODES: SurfaceMode[] = ["synthetic", "quotes", "chain"];

export function VolSurfacePanel() {
  const { t } = useTranslation();

  const [mode, setMode] = useState<SurfaceMode>("synthetic");
  const [params, setParams] = useState<SurfaceLabParams>(DEFAULT_SURFACE_PARAMS);
  const [csvText, setCsvText] = useState("");
  const [csvErrors, setCsvErrors] = useState<string[]>([]);
  const [chainTicker, setChainTicker] = useState("AAPL");
  const [chainCount, setChainCount] = useState(3);
  const [result, setResult] = useState<VolSurfaceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);

  const setParam = (patch: Partial<SurfaceLabParams>) => {
    setParams((prev) => ({ ...prev, ...patch }));
  };

  const toggleExpiry = (days: number) => {
    setParams((prev) => {
      const has = prev.expiries.includes(days);
      const next = has
        ? prev.expiries.filter((d) => d !== days)
        : [...prev.expiries, days].sort((a, b) => a - b);
      if (next.length === 0) return prev;
      return { ...prev, expiries: next };
    });
  };

  const analyze = useCallback(
    async (req: SyntheticSurfaceRequest | VolSurfaceQuotesRequest | null) => {
      if (!req) return;
      const gen = ++generation.current;
      setLoading(true);
      setError(null);
      try {
        const res =
          "quotes" in req
            ? await api.analyzeVolSurface(req)
            : await api.analyzeSyntheticSurface(req);
        if (generation.current === gen) setResult(res);
      } catch (e) {
        if (generation.current === gen) {
          setError(e instanceof Error ? e.message : t("options.surface.errorGeneric"));
        }
      } finally {
        if (generation.current === gen) setLoading(false);
      }
    },
    [t],
  );

  // Synthetic mode re-fits automatically as the smile parameters change.
  useEffect(() => {
    if (mode !== "synthetic") return;
    const req = buildSyntheticSurfaceRequest(params);
    if (!req) return;
    const timer = setTimeout(() => void analyze(req), ANALYZE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [mode, params, analyze]);

  const buildFromCsv = () => {
    const parsed = parseSurfaceQuotesCsv(csvText);
    setCsvErrors(parsed.errors);
    if (!surfaceQuotesValid(parsed.quotes)) {
      setError(t("options.surface.quotesInvalid"));
      return;
    }
    void analyze(
      buildSurfaceQuotesRequest(params.spot, parsed.quotes, {
        greeks_expiry_days: params.greeks_expiry_days,
      }),
    );
  };

  const buildFromChain = async () => {
    const ticker = chainTicker.trim();
    if (!ticker) {
      setError(t("options.surface.chainTickerRequired"));
      return;
    }
    const gen = ++generation.current;
    setLoading(true);
    setError(null);
    try {
      const first = await api.getOptionsChain(ticker);
      if (generation.current !== gen) return;
      const expirations = (first.data?.expirations ?? []).slice(0, chainCount);
      const rest =
        expirations.length > 1
          ? await Promise.all(
              expirations.slice(1).map((e) => api.getOptionsChain(ticker, e)),
            )
          : [];
      const chains = [first, ...rest];

      const quotes: SurfaceQuote[] = [];
      for (const ch of chains) {
        if (!ch.ok || !ch.data) continue;
        for (const row of ch.data.calls) {
          if (
            row.implied_volatility !== null &&
            Number.isFinite(row.implied_volatility) &&
            row.implied_volatility > 0
          ) {
            quotes.push({
              strike: row.strike,
              expiry_days: ch.data.expiration,
              implied_volatility: row.implied_volatility,
            });
          }
        }
      }
      if (quotes.length < 4) {
        if (generation.current === gen) setError(t("options.surface.chainNoQuotes"));
        return;
      }
      const req = buildSurfaceQuotesRequest(params.spot, quotes, {
        greeks_expiry_days: params.greeks_expiry_days,
      });
      if (!req) {
        if (generation.current === gen) setError(t("options.surface.errorGeneric"));
        return;
      }
      const res = await api.analyzeVolSurface(req);
      if (generation.current === gen) setResult(res);
    } catch (e) {
      if (generation.current === gen) {
        setError(e instanceof Error ? e.message : t("options.surface.errorGeneric"));
      }
    } finally {
      if (generation.current === gen) setLoading(false);
    }
  };

  const greeksExpiry = result?.greeks?.expiry_days ?? params.greeks_expiry_days;

  return (
    <div className="flex w-full flex-col gap-4">
      {/* Mode switch */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <Activity className="h-3.5 w-3.5" />
          {t("options.surface.dataSource")}
        </span>
        {MODES.map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={cn(
              "rounded border px-2.5 py-1 text-xs transition-colors",
              mode === m
                ? "border-primary bg-primary text-primary-foreground"
                : "border-muted-foreground/30 text-muted-foreground hover:border-primary hover:text-foreground",
            )}
          >
            {t(`options.surface.mode.${m}`)}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="rounded-xl border border-border/60 bg-card p-4 shadow-sm">
        {mode === "synthetic" && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.spot")}</label>
              <input
                type="number"
                min={0}
                step="any"
                value={Number.isFinite(params.spot) ? params.spot : ""}
                onChange={(e) => setParam({ spot: parseFloat(e.target.value) })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.atmIv")}</label>
              <input
                type="number"
                min={0}
                step="any"
                value={Number.isFinite(params.atm_iv_pct) ? params.atm_iv_pct : ""}
                onChange={(e) => setParam({ atm_iv_pct: parseFloat(e.target.value) })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.skew")}</label>
              <input
                type="number"
                step="any"
                value={Number.isFinite(params.skew_pct) ? params.skew_pct : ""}
                onChange={(e) => setParam({ skew_pct: parseFloat(e.target.value) })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.curvature")}</label>
              <input
                type="number"
                step="any"
                value={Number.isFinite(params.curvature_pct) ? params.curvature_pct : ""}
                onChange={(e) => setParam({ curvature_pct: parseFloat(e.target.value) })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.strikeRange")}</label>
              <input
                type="number"
                min={0}
                step="any"
                value={Number.isFinite(params.strike_range_pct) ? params.strike_range_pct : ""}
                onChange={(e) => setParam({ strike_range_pct: parseFloat(e.target.value) })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.greeksExpiry")}</label>
              <select
                value={params.greeks_expiry_days ?? ""}
                onChange={(e) =>
                  setParam({ greeks_expiry_days: e.target.value === "" ? null : Number(e.target.value) })
                }
                className={cn(INPUT_CLS, "tabular-nums")}
              >
                <option value="">{t("options.surface.greeksOff")}</option>
                {[...params.expiries].sort((a, b) => a - b).map((d) => (
                  <option key={d} value={d}>
                    {d}d
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.expiries")}</label>
              <div className="flex flex-wrap gap-1">
                {SURFACE_EXPIRY_OPTIONS.map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => toggleExpiry(d)}
                    className={cn(
                      "rounded border px-1.5 py-0.5 text-[11px] transition-colors",
                      params.expiries.includes(d)
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-muted-foreground/30 text-muted-foreground hover:border-primary hover:text-foreground",
                    )}
                  >
                    {d}d
                  </button>
                ))}
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] text-muted-foreground">{t("options.surface.strikePoints")}</label>
              <input
                type="number"
                min={5}
                max={101}
                step={1}
                value={params.strike_points}
                onChange={(e) => setParam({ strike_points: parseInt(e.target.value, 10) || 41 })}
                className={cn(INPUT_CLS, "tabular-nums")}
              />
            </div>
          </div>
        )}

        {mode === "quotes" && (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.spot")}</label>
                <input
                  type="number"
                  min={0}
                  step="any"
                  value={Number.isFinite(params.spot) ? params.spot : ""}
                  onChange={(e) => setParam({ spot: parseFloat(e.target.value) })}
                  className={cn(INPUT_CLS, "tabular-nums")}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.greeksExpiry")}</label>
                <select
                  value={params.greeks_expiry_days ?? ""}
                  onChange={(e) =>
                    setParam({ greeks_expiry_days: e.target.value === "" ? null : Number(e.target.value) })
                  }
                  className={cn(INPUT_CLS, "tabular-nums")}
                >
                  <option value="">{t("options.surface.greeksOff")}</option>
                  {[...params.expiries].sort((a, b) => a - b).map((d) => (
                    <option key={d} value={d}>
                      {d}d
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <textarea
              value={csvText}
              onChange={(e) => setCsvText(e.target.value)}
              rows={7}
              spellCheck={false}
              placeholder={t("options.surface.csvPlaceholder")}
              className={cn(INPUT_CLS, "font-mono text-xs leading-relaxed")}
            />
            {csvErrors.length > 0 && (
              <ul className="max-h-24 overflow-y-auto rounded border border-warning/30 bg-warning/5 p-2 text-[11px] text-warning">
                {csvErrors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}
            <div>
              <button
                type="button"
                onClick={buildFromCsv}
                disabled={loading}
                className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {t("options.surface.build")}
              </button>
            </div>
          </div>
        )}

        {mode === "chain" && (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.spot")}</label>
                <input
                  type="number"
                  min={0}
                  step="any"
                  value={Number.isFinite(params.spot) ? params.spot : ""}
                  onChange={(e) => setParam({ spot: parseFloat(e.target.value) })}
                  className={cn(INPUT_CLS, "tabular-nums")}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.chainTicker")}</label>
                <input
                  type="text"
                  value={chainTicker}
                  onChange={(e) => setChainTicker(e.target.value)}
                  className={cn(INPUT_CLS, "uppercase")}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.chainExpirations")}</label>
                <select
                  value={chainCount}
                  onChange={(e) => setChainCount(Number(e.target.value))}
                  className={cn(INPUT_CLS, "tabular-nums")}
                >
                  {[2, 3, 4].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-muted-foreground">{t("options.surface.greeksExpiry")}</label>
                <select
                  value={params.greeks_expiry_days ?? ""}
                  onChange={(e) =>
                    setParam({ greeks_expiry_days: e.target.value === "" ? null : Number(e.target.value) })
                  }
                  className={cn(INPUT_CLS, "tabular-nums")}
                >
                  <option value="">{t("options.surface.greeksOff")}</option>
                  {[...params.expiries].sort((a, b) => a - b).map((d) => (
                    <option key={d} value={d}>
                      {d}d
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div>
              <button
                type="button"
                onClick={() => void buildFromChain()}
                disabled={loading}
                className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                {t("options.surface.loadChain")}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Backend error banner */}
      {error && (
        <div className="rounded border border-danger/30 bg-danger/5 p-3 text-sm text-danger">{error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="flex flex-col gap-4">
          {/* IV heatmap */}
          <section className="rounded-xl border border-border/60 bg-card p-4 shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm font-semibold">{t("options.surface.heatmapTitle")}</div>
              {loading && (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  {t("options.surface.fitting")}
                </span>
              )}
            </div>
            <VolSurfaceHeatmap grid={result.grid} />
            {result.limitations.length > 0 && (
              <ul className="mt-2 list-disc ps-5 text-xs text-muted-foreground">
                {result.limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            )}
          </section>

          {/* Per-expiry SVI fits */}
          <section className="rounded-xl border border-border/60 bg-card p-4 shadow-sm">
            <div className="mb-2 text-sm font-semibold">{t("options.surface.fitTableTitle")}</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-muted-foreground">
                    <th className="pb-1.5 pe-3 font-medium text-start">{t("options.surface.fitExpiry")}</th>
                    <th className="pb-1.5 pe-3 font-medium text-end">{t("options.surface.fitQuotes")}</th>
                    <th className="pb-1.5 pe-3 font-medium text-end">{t("options.surface.fitRmse")}</th>
                    <th className="pb-1.5 pe-3 font-medium text-end">{t("options.surface.fitMaxResidual")}</th>
                    <th className="pb-1.5 pe-3 font-medium text-center">{t("options.surface.fitWings")}</th>
                    <th className="pb-1.5 font-medium text-start">{t("options.surface.fitParams")}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.expiries.map((e) => (
                    <tr key={e.expiry_days} className="border-t border-border/40">
                      <td className="py-1.5 pe-3 tabular-nums">{e.expiry_days}d</td>
                      <td className="py-1.5 pe-3 text-end tabular-nums">{e.n_quotes}</td>
                      <td className="py-1.5 pe-3 text-end tabular-nums">
                        {(e.rmse * 100).toFixed(2)}%
                      </td>
                      <td className="py-1.5 pe-3 text-end tabular-nums">
                        {(e.max_residual * 100).toFixed(2)}%
                      </td>
                      <td className="py-1.5 pe-3 text-center">
                        {e.wings_ok ? (
                          <span className="text-success">✓</span>
                        ) : (
                          <span className="text-warning">⚠</span>
                        )}
                      </td>
                      <td className="py-1.5 font-mono text-[11px] text-muted-foreground tabular-nums">
                        a={e.params.a.toFixed(4)} b={e.params.b.toFixed(4)} ρ={e.params.rho.toFixed(3)}{" "}
                        m₀={e.params.m0.toFixed(3)} σ={e.params.sigma.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Greeks dashboard */}
          {result.greeks && (
            <section className="rounded-xl border border-border/60 bg-card p-4 shadow-sm">
              <div className="mb-2 text-sm font-semibold">
                {t("options.surfaceGreeks.title")}{" "}
                <span className="text-xs font-normal text-muted-foreground">
                  {t("options.surfaceGreeks.atExpiry", { expiry: greeksExpiry })}
                </span>
              </div>
              <GreeksCurvesChart greeks={result.greeks} />
            </section>
          )}
        </div>
      )}
    </div>
  );
}
