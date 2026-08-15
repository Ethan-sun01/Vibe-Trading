import { useEffect, useRef } from "react";
import i18n from "@/i18n";
import type { SurfaceGreeksGrid } from "@/lib/options";
import { getChartTheme } from "@/lib/chart-theme";
import { echarts, CHART_GROUP, connectCharts } from "@/lib/echarts";
import { useThemeDark } from "@/lib/theme-store";

const GREEK_KEYS = ["delta", "gamma", "theta", "vega", "rho"] as const;
type GreekKey = (typeof GREEK_KEYS)[number];

/** Display colour per Greek, fixed so the curves are distinguishable. */
const GREEK_COLORS: Record<GreekKey, string> = {
  delta: "#2f6fed",
  gamma: "#7c3aed",
  theta: "#e05d44",
  vega: "#0f9d58",
  rho: "#b8860b",
};

interface Props {
  greeks: SurfaceGreeksGrid;
  height?: number;
}

function MiniGreekChart({
  keyName,
  strikes,
  values,
  height,
}: {
  keyName: GreekKey;
  strikes: number[];
  values: number[];
  height: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dark = useThemeDark();

  useEffect(() => {
    if (!ref.current) return;
    const t = getChartTheme();
    const chart = echarts.init(ref.current);
    chart.group = CHART_GROUP;
    connectCharts();

    // Downsample long strike grids for display; keep first/last for honesty.
    const maxPoints = 41;
    const stride = Math.max(1, Math.ceil(strikes.length / maxPoints));
    const idx: number[] = [];
    for (let i = 0; i < strikes.length; i += stride) idx.push(i);
    if (idx[idx.length - 1] !== strikes.length - 1) idx.push(strikes.length - 1);

    const labels = idx.map((i) => strikes[i].toLocaleString());

    chart.setOption({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: t.tooltipBg,
        borderColor: t.tooltipBorder,
        textStyle: { color: t.tooltipText, fontSize: 11 },
      },
      grid: { left: 6, right: 6, top: 10, bottom: 4, containLabel: true },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: { color: t.textColor, fontSize: 9, interval: "auto" },
        axisLine: { lineStyle: { color: t.axisColor } },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: t.axisColor, opacity: 0.35 } },
        axisLabel: { color: t.textColor, fontSize: 9 },
      },
      series: [
        {
          type: "line",
          data: idx.map((i) => values[i]),
          showSymbol: false,
          lineStyle: { width: 1.6, color: GREEK_COLORS[keyName] },
          itemStyle: { color: GREEK_COLORS[keyName] },
          emphasis: { focus: "series" },
        },
      ],
    });

    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keyName, strikes, values, dark]);

  return <div ref={ref} style={{ height }} className="w-full" />;
}

/**
 * Surface-consistent Greeks across the strike grid, one mini chart per Greek.
 * Each curve is priced with the fitted surface IV at that strike — the skew
 * bends delta/gamma/vega instead of a single ATM vol pretending it does not
 * exist. Scales differ wildly across Greeks (delta ~[0,1], theta ~[-1,1],
 * vega ~[0,0.5]), so each gets its own axis rather than one unreadable plot.
 */
export function GreeksCurvesChart({ greeks, height = 200 }: Props) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-5">
      {GREEK_KEYS.map((key) => (
        <div key={key} className="min-w-0">
          <div className="mb-1 flex items-center justify-between text-[11px]">
            <span className="font-medium text-foreground">{i18n.t(`options.surfaceGreeks.${key}`)}</span>
            <span className="text-muted-foreground/70">{i18n.t("options.surfaceGreeks.strikeAxis")}</span>
          </div>
          <MiniGreekChart
            keyName={key}
            strikes={greeks.strikes}
            values={greeks[key]}
            height={height}
          />
        </div>
      ))}
    </div>
  );
}
