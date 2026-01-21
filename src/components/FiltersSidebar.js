import React from 'react';

const widgetClass = `rounded-xl border border-slate-200 bg-white p-5 shadow-sm`;

const FiltersSidebar = ({
  selectedSku,
  skus,
  onSkuChange,
  selectedChannel,
  channels,
  onChannelChange,
  selectedRegion,
  regions,
  onRegionChange,
  onRun,
  loading
}) => {
  return (
    <div className="space-y-4">
      <div className={widgetClass}>
        <div className="mb-4 flex items-center gap-2">
          <svg
            className="h-5 w-5 text-emerald-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
            />
          </svg>
          <h3 className="text-sm font-semibold text-slate-900">AI Filters</h3>
        </div>
        <div className="space-y-4">
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">
              Select SKU
            </label>
            <select
              value={selectedSku || ''}
              onChange={(e) => onSkuChange(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-emerald-300 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            >
              <option value="">All SKUs (120 Total)</option>
              {skus.map((sku) => (
                <option key={sku} value={sku}>
                  {sku}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">
              Select Channel
            </label>
            <select
              value={selectedChannel || ''}
              onChange={(e) => onChannelChange(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-emerald-300 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            >
              <option value="">All Channels</option>
              {channels && channels.map((ch) => (
                <option key={ch} value={ch}>
                  {ch}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-slate-500">
              Select Region
            </label>
            <select
              value={selectedRegion || ''}
              onChange={(e) => onRegionChange(e.target.value)}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-emerald-300 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            >
              <option value="">All Regions</option>
              {regions && regions.map((rg) => (
                <option key={rg} value={rg}>
                  {rg}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className={widgetClass}>
        <h3 className="mb-3 text-sm font-semibold text-slate-900">AI Analysis</h3>
        <p className="mb-3 text-xs text-slate-600">
          Run multi-agent AI analysis on selected filters
        </p>
        <button
          onClick={onRun}
          disabled={loading}
          className={`w-full rounded-lg border px-4 py-2 text-sm font-medium text-white shadow-sm transition ${loading
              ? 'border-slate-300 bg-slate-400 cursor-not-allowed'
              : 'border-emerald-300 bg-emerald-500 hover:bg-emerald-600'
            }`}
        >
          {loading ? 'Processing AI Analysis...' : 'Run AI Analysis'}
        </button>
      </div>

      <div className={widgetClass}>
        <h3 className="mb-2 text-sm font-semibold text-slate-900">AI Agents Active</h3>
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-500"></div>
            <span className="text-slate-600">Base Price Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-blue-500"></div>
            <span className="text-slate-600">Promo Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-purple-500"></div>
            <span className="text-slate-600">Competitor Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-orange-500"></div>
            <span className="text-slate-600">Inventory Agent</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FiltersSidebar;
