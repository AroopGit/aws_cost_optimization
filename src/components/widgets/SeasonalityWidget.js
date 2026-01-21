import React from 'react';

const widgetClass = `rounded-xl border border-slate-200 bg-white p-5 shadow-sm`;

const SeasonalityWidget = () => {
  return (
    <div className={widgetClass}>
      <div className="mb-3 flex items-center gap-2">
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
            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <h3 className="text-sm font-semibold text-slate-900">Seasonality Insights</h3>
      </div>
      <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-4 text-sm text-emerald-800">
        Seasonality values in the data remain high (1.0+), meaning baseline demand stays
        resilient through most months with only light dips during non-festival periods.
      </div>
    </div>
  );
};

export default SeasonalityWidget;

