import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchForecastPricing } from '../api';

const ForecastingWidget = ({ sku, channel, region }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
    const [days, setDays] = useState(30);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedRow, setExpandedRow] = useState(null);

    const handleRunForecast = async () => {
        setLoading(true);
        setError(null);
        setExpandedRow(null);
        try {
            const data = await fetchForecastPricing({
                sku_id: sku,
                channel: channel,
                region: region,
                start_date: startDate,
                days: parseInt(days)
            });
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const toggleRow = (idx) => {
        setExpandedRow(expandedRow === idx ? null : idx);
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="w-full mt-4 flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white p-4 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition"
            >
                <svg className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Open Forecasting Agent
            </button>
        );
    }

    return (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <svg className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Forecasting Agent
                </h3>
                <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-slate-600">
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-xs font-medium text-slate-500 mb-1">Start Date</label>
                    <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="w-full rounded-lg border-slate-200 text-sm text-slate-900"
                    />
                </div>
                <div>
                    <label className="block text-xs font-medium text-slate-500 mb-1">Duration (Days)</label>
                    <input
                        type="number"
                        value={days}
                        onChange={(e) => setDays(e.target.value)}
                        className="w-full rounded-lg border-slate-200 text-sm text-slate-900"
                    />
                </div>
            </div>

            <button
                onClick={handleRunForecast}
                disabled={loading}
                className="w-full rounded-lg bg-purple-600 px-4 py-2 text-sm font-semibold text-white hover:bg-purple-700 disabled:opacity-50 transition mb-6"
            >
                {loading ? 'Running Forecast...' : 'Run Forecast'}
            </button>

            {error && (
                <div className="mb-4 p-3 bg-red-50 text-red-600 text-xs rounded-lg">
                    {error}
                </div>
            )}

            {results && (
                <div className="space-y-6">
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={results}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(val) => val.slice(5)} />
                                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                    formatter={(val) => [`₹${val}`, '']}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="candidates.price_optimal.price" name="Optimal" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="candidates.price_aggressive.price" name="Aggressive" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                                <Line type="monotone" dataKey="candidates.price_base.price" name="Conservative" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-slate-50 text-slate-500 font-medium">
                                <tr>
                                    <th className="px-3 py-2">Date</th>
                                    <th className="px-3 py-2">Seasonality</th>
                                    <th className="px-3 py-2 text-right">Optimal Price</th>
                                    <th className="px-3 py-2 text-right">Pred. Units</th>
                                    <th className="px-3 py-2"></th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {results.slice(0, 10).map((row, idx) => (
                                    <React.Fragment key={idx}>
                                        <tr className="hover:bg-slate-50 transition cursor-pointer" onClick={() => toggleRow(idx)}>
                                            <td className="px-3 py-2 text-slate-900 font-medium">{row.date}</td>
                                            <td className="px-3 py-2 text-slate-600">{row.seasonality_index.toFixed(2)}</td>
                                            <td className="px-3 py-2 text-right font-medium text-purple-600">₹{row.candidates.price_optimal.price}</td>
                                            <td className="px-3 py-2 text-right text-slate-600">{row.candidates.price_optimal.pred_units}</td>
                                            <td className="px-3 py-2 text-center text-slate-400">
                                                <svg className={`h-4 w-4 transform transition ${expandedRow === idx ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                                </svg>
                                            </td>
                                        </tr>
                                        {expandedRow === idx && (
                                            <tr>
                                                <td colSpan={5} className="px-3 py-3 bg-slate-50/50">
                                                    <div className="space-y-4">
                                                        {/* Agent Recommendations */}
                                                        <div>
                                                            <h4 className="text-xs font-semibold text-slate-700 mb-2">Agent Recommendations (3 Candidates Each)</h4>
                                                            <div className="grid grid-cols-4 gap-2">
                                                                {Object.entries(row.traces).map(([agent, candidates]) => (
                                                                    <div key={agent} className="bg-white p-2 rounded border border-slate-100 shadow-sm">
                                                                        <div className="text-[10px] font-bold text-slate-500 uppercase mb-1">{agent}</div>
                                                                        <div className="space-y-1">
                                                                            {Object.entries(candidates).map(([type, price]) => (
                                                                                <div key={type} className="flex justify-between text-[10px]">
                                                                                    <span className="text-slate-400 capitalize">{type.slice(0, 4)}</span>
                                                                                    <span className="font-medium text-slate-700">₹{price.toFixed(0)}</span>
                                                                                </div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        {/* Final Candidates */}
                                                        <div>
                                                            <h4 className="text-xs font-semibold text-slate-700 mb-2">Final Candidates & Predictions</h4>
                                                            <div className="grid grid-cols-3 gap-2">
                                                                {Object.entries(row.candidates).map(([key, data]) => (
                                                                    <div key={key} className="bg-white p-2 rounded border border-slate-100 shadow-sm">
                                                                        <div className="text-[10px] font-bold text-slate-500 uppercase mb-1">{key.replace('price_', '')}</div>
                                                                        <div className="flex justify-between text-[10px] mb-1">
                                                                            <span className="text-slate-400">Price</span>
                                                                            <span className="font-medium text-purple-600">₹{data.price}</span>
                                                                        </div>
                                                                        <div className="flex justify-between text-[10px] mb-1">
                                                                            <span className="text-slate-400">Units</span>
                                                                            <span className="font-medium text-slate-700">{data.pred_units}</span>
                                                                        </div>
                                                                        <div className="flex justify-between text-[10px]">
                                                                            <span className="text-slate-400">Elasticity</span>
                                                                            <span className="font-medium text-slate-700">{data.elasticity.toFixed(2)}</span>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        {/* Final Selection */}
                                                        <div className="bg-purple-50 p-2 rounded border border-purple-100">
                                                            <div className="flex justify-between items-center">
                                                                <span className="text-xs font-semibold text-purple-800">Final Selection</span>
                                                                <span className="text-sm font-bold text-purple-700">₹{row.selection.price_recommended}</span>
                                                            </div>
                                                            <div className="text-[10px] text-purple-600 mt-1">{row.selection.approval} ({row.selection.change_pct}%)</div>
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                ))}
                            </tbody>
                        </table>
                        {results.length > 10 && (
                            <p className="text-center text-xs text-slate-400 mt-2">Showing first 10 of {results.length} days</p>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ForecastingWidget;
