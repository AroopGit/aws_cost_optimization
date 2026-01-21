import React from 'react';

const widgetClass = `rounded-xl border border-slate-200 bg-white p-5 shadow-sm`;

const TrendBehaviorWidget = () => {
  return (
    <div className={widgetClass}>
      <div className="mb-3 flex items-center gap-2">
        <svg
          className="h-5 w-5 text-sky-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
          />
        </svg>
        <h3 className="text-sm font-semibold text-slate-900">Overall Demand Trend</h3>
      </div>
      <div className="rounded-lg border border-sky-100 bg-sky-50 p-4 text-sm text-sky-800">
        Units sold from the dataset show steady growth across North and South regions, with
        promotional lifts keeping demand above 1,500 units monthly.
      </div>
    </div>
  );
};

export default TrendBehaviorWidget;

