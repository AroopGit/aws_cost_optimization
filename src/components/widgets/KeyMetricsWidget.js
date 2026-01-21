import React, { useMemo } from 'react';
import { formatCurrency, formatPercent } from '../../utils/dataUtils';

const KeyMetricsWidget = ({ analysisRows }) => {
  const metrics = useMemo(() => {
    if (analysisRows.length === 0) {
      return {
        totalSkus: 0,
        avgPriceChange: 0,
        avgRoi: 0,
        totalRevenue: 0,
      };
    }

    const avgDelta = analysisRows.reduce((sum, r) => sum + r.delta, 0) / analysisRows.length;
    const avgRoi = analysisRows.reduce((sum, r) => sum + r.projectedRoi, 0) / analysisRows.length;
    const totalRevenue = analysisRows.reduce(
      (sum, r) => sum + r.recommendedPrice * r.baselineVolume,
      0
    );

    return {
      totalSkus: analysisRows.length,
      avgPriceChange: avgDelta,
      avgRoi,
      totalRevenue,
    };
  }, [analysisRows]);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-xs uppercase tracking-wide text-slate-500">Total SKUs</div>
        <div className="mt-2 text-2xl font-semibold text-slate-900">{metrics.totalSkus}</div>
      </div>
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-xs uppercase tracking-wide text-slate-500">Avg Price Change</div>
        <div
          className={`mt-2 text-2xl font-semibold ${
            metrics.avgPriceChange >= 0 ? 'text-emerald-600' : 'text-rose-500'
          }`}
        >
          {formatPercent(metrics.avgPriceChange)}
        </div>
      </div>
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-xs uppercase tracking-wide text-slate-500">Avg Projected ROI</div>
        <div className="mt-2 text-2xl font-semibold text-slate-900">
          {metrics.avgRoi.toFixed(1)}x
        </div>
      </div>
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-xs uppercase tracking-wide text-slate-500">Total Revenue</div>
        <div className="mt-2 text-2xl font-semibold text-slate-900">
          {formatCurrency(metrics.totalRevenue)}
        </div>
      </div>
    </div>
  );
};

export default KeyMetricsWidget;

