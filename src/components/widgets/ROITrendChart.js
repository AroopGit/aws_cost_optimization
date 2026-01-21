import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const ROITrendChart = ({ analysisRows }) => {
  const data = analysisRows.map((row) => ({
    sku: row.sku,
    roi: row.projectedRoi,
    confidence: row.confidence,
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-slate-500">
        No data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="sku" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="roi"
          stroke="#3b82f6"
          strokeWidth={2}
          name="Projected ROI"
        />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="#f59e0b"
          strokeWidth={2}
          name="Confidence %"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ROITrendChart;

