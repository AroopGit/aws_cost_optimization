import React from 'react';

const widgetClass = `rounded-xl border border-slate-200 bg-white p-5 shadow-sm`;

const AnomaliesWidget = () => {
  const anomalies = [
    {
      date: '2024-08',
      event: 'Spike',
      description: 'Independence Day and Raksha Bandhan festivals',
      type: 'spike',
    },
    {
      date: '2025-07',
      event: 'Drop',
      description: 'No major festivals or promotions during this period',
      type: 'drop',
    },
  ];

  return (
    <div className={widgetClass}>
      <div className="mb-4 flex items-center gap-2">
        <svg
          className="h-5 w-5 text-amber-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <h3 className="text-sm font-semibold text-slate-900">Detected Anomalies</h3>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {anomalies.map((anomaly, idx) => (
          <div
            key={idx}
            className={`rounded-lg border p-4 ${
              anomaly.type === 'spike'
                ? 'border-emerald-200 bg-emerald-50'
                : 'border-rose-200 bg-rose-50'
            }`}
          >
            <div className="mb-2 flex items-center gap-2">
              <svg
                className="h-4 w-4 text-slate-500"
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
              <span className="text-xs font-medium text-slate-600">{anomaly.date}</span>
            </div>
            <div
              className={`mb-1 text-sm font-medium ${
                anomaly.type === 'spike' ? 'text-emerald-700' : 'text-rose-700'
              }`}
            >
              Event: {anomaly.event}
            </div>
            <div className="text-xs text-slate-600">{anomaly.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnomaliesWidget;

