import React from 'react';
import { formatCurrency, formatPercent } from '../../utils/dataUtils';

const PricingTable = ({ analysisRows, selectedRow, onRowSelect }) => {
  if (analysisRows.length === 0) {
    return (
      <p className="mt-4 text-sm text-slate-600">
        No pricing recommendations available. Run the analysis to see results.
      </p>
    );
  }

  return (
    <div className="mt-4 overflow-x-auto">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3 text-left">SKU</th>
            <th className="px-4 py-3 text-left">Channel</th>
            <th className="px-4 py-3 text-left">Region</th>
            <th className="px-4 py-3 text-right">Base Price</th>
            <th className="px-4 py-3 text-right">Recommended</th>
            <th className="px-4 py-3 text-right">Δ%</th>
            <th className="px-4 py-3 text-center">Confidence</th>
            <th className="px-4 py-3 text-left">Rationale</th>
            <th className="px-4 py-3 text-right">Projected ROI</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200">
          {analysisRows.map((row) => (
            <tr
              key={row.id}
              onClick={() => onRowSelect(row)}
              className={`cursor-pointer transition ${selectedRow?.id === row.id
                  ? 'bg-purple-50 hover:bg-purple-100'
                  : 'hover:bg-slate-50'
                }`}
            >
              <td className="px-4 py-3 font-medium text-slate-900">{row.sku}</td>
              <td className="px-4 py-3 text-slate-600">{row.channel}</td>
              <td className="px-4 py-3 text-slate-600">{row.region}</td>
              <td className="px-4 py-3 text-right text-slate-600">
                {formatCurrency(row.basePrice)}
              </td>
              <td className="px-4 py-3 text-right text-emerald-600">
                {formatCurrency(row.recommendedPrice)}
              </td>
              <td
                className={`px-4 py-3 text-right ${row.delta >= 0 ? 'text-emerald-600' : 'text-rose-500'
                  }`}
              >
                {formatPercent(row.delta)}
              </td>
              <td className="px-4 py-3 text-center text-slate-600">{row.confidence}%</td>
              <td className="px-4 py-3 text-left text-xs text-slate-600">{row.rationale}</td>
              <td className="px-4 py-3 text-right text-slate-600">{row.projectedRoi}x</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default PricingTable;

