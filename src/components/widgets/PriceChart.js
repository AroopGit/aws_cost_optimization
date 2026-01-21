import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatCurrency } from '../../utils/dataUtils';

const PriceChart = ({ analysisRows }) => {
  // Check if all rows have the same SKU to avoid redundancy in labels
  const allSameSku = analysisRows.length > 0 && analysisRows.slice(0, 7).every(r => r.sku === analysisRows[0].sku);

  const data = analysisRows.slice(0, 7).map((row) => ({
    label: allSameSku ? `${row.channel}-${row.region}` : `${row.sku.split('_').pop()}\n${row.channel}-${row.region}`,
    sku: row.sku,
    channel: row.channel,
    region: row.region,
    basePrice: row.basePrice,
    recommendedPrice: row.recommendedPrice,
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-slate-500">
        No data available. Please select filters and run analysis.
      </div>
    );
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-lg">
          <p className="font-semibold text-slate-900">{data.sku}</p>
          <p className="text-xs text-slate-600">{data.channel} • {data.region}</p>
          <div className="mt-2 space-y-1">
            <p className="text-sm">
              <span className="text-slate-600">Base Price:</span>{' '}
              <span className="font-semibold text-slate-900">{formatCurrency(data.basePrice)}</span>
            </p>
            <p className="text-sm">
              <span className="text-slate-600">Recommended:</span>{' '}
              <span className="font-semibold text-brand-dark">{formatCurrency(data.recommendedPrice)}</span>
            </p>
            <p className="text-sm">
              <span className="text-slate-600">Change:</span>{' '}
              <span className={`font-semibold ${data.recommendedPrice > data.basePrice ? 'text-brand-dark' : 'text-rose-600'}`}>
                {((data.recommendedPrice - data.basePrice) / data.basePrice * 100).toFixed(1)}%
              </span>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomXAxisTick = ({ x, y, payload }) => {
    const parts = payload.value.split('\n');
    const truncate = (str, max) => str.length > max ? str.substring(0, max) + '...' : str;

    return (
      <g transform={`translate(${x},${y}) rotate(-45)`}>
        <text
          x={0}
          y={0}
          dy={8}
          textAnchor="end"
          fill="#64748b"
          fontSize="9"
          style={{ fontSize: '9px', fontWeight: 500 }}
        >
          {parts.map((p, i) => (
            <tspan key={i} x={0} dy={i === 0 ? 0 : 10} fill={i === 0 ? "#64748b" : "#94a3b8"}>
              {truncate(p, 10)}
            </tspan>
          ))}
        </text>
      </g>
    );
  };

  return (
    <div>
      <div className="mb-3 rounded-lg bg-blue-50 p-3 border border-blue-100/50">
        <p className="text-xs text-slate-600 leading-relaxed">
          <span className="font-bold text-blue-700">Top 7 Insights:</span> Comparing <span className="text-blue-400 font-semibold">Base Price</span> vs <span className="text-blue-900 font-bold">AI Recommended Price</span>. Hover for details.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ bottom: 40, left: 10, right: 10 }} barGap={6}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
          <XAxis
            dataKey="label"
            tick={<CustomXAxisTick />}
            height={90}
            interval={0}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(value) => `₹${value.toFixed(0)}`}
            stroke="#64748b"
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="square"
          />
          <Bar
            dataKey="basePrice"
            fill="#93c5fd"
            name="Base Price"
            radius={[4, 4, 0, 0]}
            barSize={16}
          />
          <Bar
            dataKey="recommendedPrice"
            fill="#1e40af"
            name="AI Recommended Price"
            radius={[4, 4, 0, 0]}
            barSize={16}
          />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 text-center text-xs text-slate-500">
        Showing {data.length} of {analysisRows.length} total recommendations
      </div>
    </div>
  );
};

export default PriceChart;
