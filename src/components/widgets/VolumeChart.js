import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const VolumeChart = ({ analysisRows }) => {
  const data = analysisRows.map((row) => {
    const delta = row.delta;
    const demandFactor = Math.max(0.2, 1 - row.elasticity * delta);
    const projectedVolume = Math.round(row.baselineVolume * demandFactor);

    return {
      sku: row.sku,
      baseline: row.baselineVolume,
      projected: projectedVolume,
    };
  });

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-slate-500">
        No data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="sku" />
        <YAxis />
        <Tooltip />
        <Area
          type="monotone"
          dataKey="baseline"
          stackId="1"
          stroke="#94a3b8"
          fill="#cbd5e1"
          name="Baseline Volume"
        />
        <Area
          type="monotone"
          dataKey="projected"
          stackId="1"
          stroke="#10b981"
          fill="#86efac"
          name="Projected Volume"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default VolumeChart;

