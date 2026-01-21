import React from 'react';

const widgetClass = `rounded-xl border border-slate-200 bg-white p-5 shadow-sm`;

const KeyDriversWidget = ({ analysisRows }) => {
  const drivers = [
    {
      driver: 'Promotions',
      PromoPct: '0.2267222220020564',
      Months: '2023-05, 2024-10',
      strength: '50.0%',
      reason: 'High promo percentage during summer and festival seasons',
      Festiv: 'None',
    },
    {
      driver: 'Festivals',
      PromoPct: 'None',
      Months: '2023-03, 2023-08, 2024-10',
      strength: '73.0%',
      reason: 'Festivals like Holi, Eid, and Diwali have a significant impact on sales',
      Festiv: '1',
    },
    {
      driver: 'Volume Discounts',
      PromoPct: 'None',
      Months: '2023-02, 2024-07',
      strength: '30.0%',
      reason: 'No volume discounts during certain periods',
      Festiv: 'None',
    },
  ];

  return (
    <div className={widgetClass}>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg
            className="h-5 w-5 text-slate-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <h3 className="text-sm font-semibold text-slate-900">Key Drivers of Sales</h3>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead className="border-b border-slate-200 text-left text-slate-500 uppercase tracking-wide">
            <tr>
              <th className="pb-2 pr-4">Driver</th>
              <th className="pb-2 pr-4">Promo %</th>
              <th className="pb-2 pr-4">Months</th>
              <th className="pb-2 pr-4">Strength</th>
              <th className="pb-2 pr-4">Reason</th>
              <th className="pb-2">Festival</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 text-slate-600">
            {drivers.map((d, idx) => (
              <tr key={idx} className="hover:bg-slate-50">
                <td className="py-2 pr-4 font-medium text-slate-900">{d.driver}</td>
                <td className="py-2 pr-4">{d.PromoPct}</td>
                <td className="py-2 pr-4">{d.Months}</td>
                <td className="py-2 pr-4">{d.strength}</td>
                <td className="py-2 pr-4 text-slate-500">{d.reason}</td>
                <td className="py-2">{d.Festiv}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default KeyDriversWidget;

