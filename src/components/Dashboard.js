import React from 'react';
import PriceChart from './widgets/PriceChart';
import PricingTable from './widgets/PricingTable';
import KeyMetricsWidget from './widgets/KeyMetricsWidget';
import TrendBehaviorWidget from './widgets/TrendBehaviorWidget';
import SeasonalityWidget from './widgets/SeasonalityWidget';
import KeyDriversWidget from './widgets/KeyDriversWidget';
import AnomaliesWidget from './widgets/AnomaliesWidget';
import CsvProcessor from './CsvProcessor';

const panelClass = `rounded-2xl border border-brand-light/20 bg-white/70 backdrop-blur-md p-6 shadow-[0_4px_20px_-4px_rgba(31,117,254,0.05)] hover:shadow-[0_8px_30px_-4px_rgba(31,117,254,0.1)] transition-all duration-300 transform hover:-translate-y-0.5 ring-1 ring-white/40`;

const Dashboard = ({ analysisRows, selectedRow, onRowSelect, rawCsv, onCsvChange, onRun }) => {
  return (
    <div className="space-y-4">
      {/* Key Metrics Summary */}
      <KeyMetricsWidget analysisRows={analysisRows} />

      {/* Pricing Recommendations Table */}
      <div className={panelClass}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">AI-Powered Pricing Recommendations</h2>
          <span className="text-xs text-slate-500">
            Showing {analysisRows.length} of 120 Total SKUs
          </span>
        </div>
        <PricingTable
          analysisRows={analysisRows}
          selectedRow={selectedRow}
          onRowSelect={onRowSelect}
        />
      </div>

      {/* CSV Processing Section */}
      <CsvProcessor />
    </div>
  );
};

const DashboardCharts = ({ analysisRows }) => {
  return (
    <div className="space-y-4 h-full">
      <div className={panelClass}>
        <h3 className="mb-4 text-sm font-semibold text-slate-900">Price Comparison</h3>
        <PriceChart analysisRows={analysisRows} />
      </div>

      {/* Placeholder / Inactive Section Widgets */}
      <div className="space-y-4 opacity-60 grayscale-[0.8] pointer-events-none select-none filter blur-[0.5px]">
        <div className="grid grid-cols-1 gap-4">
          <TrendBehaviorWidget />
          <SeasonalityWidget />
        </div>
        <KeyDriversWidget analysisRows={analysisRows} />
        <AnomaliesWidget />
      </div>
    </div>
  );
};

export { DashboardCharts };
export default Dashboard;
