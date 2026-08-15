import { useEffect, useRef } from "react";
import i18n from "@/i18n";
import { downsampleIndices, type VolSurfaceGrid } from "@/lib/options";
import { getChartTheme } from "@/lib/chart-theme";
import { echarts, CHART_GROUP, connectCharts } from "@/lib/echarts";
import { useThemeDark } from "@/lib/theme-store";

const MAX_DISPLAY_STRIKES = 61;
const MAX_DISPLAY_EXPIRIES = 12;

/** IV colour ramp: calm (low vol) → stressed (high vol), theme-independent. */
const IV_RAMP = ["#2f6fed", "#35b3a0", "#e8c34a", "#e05d44"];

interface Props {
  grid: VolSurfaceGrid;
  height?: number;
}

/**
 * Strike × expiry implied-volatility heatmap. All values come from the fitted
 * surface grid; the chart is display-only, so strikes are downsampled for
 * readability without lying about the underlying values.
 */
export function VolSurfaceHeatmap({ grid, height = 380 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const dark = useThemeDark();

  const hasData = grid.strikes.length > 0 && grid.expiries.length > 0 && grid.iv.length > 0;

  useEffect(() => {
    if (!ref.current || !hasData) return;
    const t = getChartTheme();
    const chart = echarts.init(ref.current);
    chart.group = CHART_GROUP;
    connectCharts();

    const strikeIdx = downsampleIndices(grid.strikes.length, MAX_DISPLAY_STRIKES);
    const expiryIdx = downsampleIndices(grid.expiries.length, MAX_DISPLAY_EXPIRIES);
    const strikeLabels = strikeIdx.map((i) => grid.strikes[i].toLocaleString());
    // Longer expiries on the lower rows: reverse the axis so the near end is
    // on top, matching how term structures are usually drawn.
    const expiryLabels = expiryIdx.map((i) => `${grid.expiries[i]}d`).reverse();

    const data: [number, number, number][] = [];
    let minIv = Infinity;
    let maxIv = -Infinity;
    for (let r = 0; r < expiryIdx.length; r++) {
      const row = grid.iv[expiryIdx[r]] ?? [];
      for (let c = 0; c < strikeIdx.length; c++) {
        const v = row[strikeIdx[c]];
        if (!Number.isFinite(v)) continue;
        const pct = Number((v * 100).toFixed(2));
        data.push([c, r, pct]);
        if (pct < minIv) minIv = pct;
        if (pct > maxIv) maxIv = pct;
      }
    }
    if (!Number.isFinite(minIv)) minIv = 0;
    if (!Number.isFinite(maxIv)) maxIv = 1;

    chart.setOption({
      backgroundColor: "transparent",
      tooltip: {
        position: "top",
        backgroundColor: t.tooltipBg,
        borderColor: t.tooltipBorder,
        textStyle: { color: t.tooltipText, fontSize: 12 },
        formatter: (params: unknown) => {
          const p = params as { data: [number, number, number] };
          const [x, y, v] = p.data;
          return (
            `<b>${i18n.t("options.surface.strikeAxis")}</b>: ${strikeLabels[x] ?? "?"}` +
            `<br/><b>${i18n.t("options.surface.expiryAxis")}</b>: ${expiryLabels[y] ?? "?"}` +
            `<br/><b>${i18n.t("options.surface.iv")}</b>: ${v.toLocaleString()}%`
          );
        },
      },
      grid: { left: "3%", right: "10%", top: "8%", bottom: "14%", containLabel: true },
      xAxis: {
        type: "category",
        data: strikeLabels,
        name: i18n.t("options.surface.strikeAxis"),
        nameTextStyle: { color: t.textColor, fontSize: 10 },
        nameLocation: "middle",
        nameGap: 30,
        axisLabel: { color: t.textColor, fontSize: 10, rotate: 30 },
        axisLine: { lineStyle: { color: t.axisColor } },
        splitArea: { show: false },
      },
      yAxis: {
        type: "category",
        data: expiryLabels,
        inverse: true,
        name: i18n.t("options.surface.expiryAxis"),
        nameTextStyle: { color: t.textColor, fontSize: 10 },
        axisLabel: { color: t.textColor, fontSize: 10, interval: 0 },
        axisLine: { lineStyle: { color: t.axisColor } },
        splitArea: { show: false },
      },
      visualMap: {
        min: minIv,
        max: maxIv,
        precision: 1,
        calculable: true,
        orient: "vertical",
        right: 8,
        top: "center",
        textStyle: { color: t.textColor, fontSize: 11 },
        formatter: (v: number) => `${v.toFixed(0)}%`,
        inRange: { color: IV_RAMP },
      },
      series: [
        {
          name: i18n.t("options.surface.iv"),
          type: "heatmap",
          data,
          label: {
            show: strikeIdx.length <= 16 && expiryIdx.length <= 6,
            fontSize: 10,
            color: t.textColor,
            formatter: (params: unknown) => {
              const p = params as { value: [number, number, number] };
              return `${p.value[2].toFixed(0)}%`;
            },
          },
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: "rgba(0, 0, 0, 0.5)" },
          },
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
  }, [grid, hasData, dark]);

  if (!hasData) {
    return (
      <div className="flex h-[300px] items-center justify-center text-sm text-muted-foreground">
        {i18n.t("options.surface.noData")}
      </div>
    );
  }

  return <div ref={ref} style={{ height }} className="w-full" />;
}
