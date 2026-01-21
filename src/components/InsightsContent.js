import React, { useState } from 'react';
import { ChevronRight, AlertCircle } from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, Line } from 'recharts';

const InsightsContent = ({ data = [] }) => {
    // Group data by priority
    const highPriority = data.filter(d => d.rationale === 'Manager Review' || Math.abs(d.delta) > 0.15);
    const mediumPriority = data.filter(d => d.rationale === 'Auto-approve' && Math.abs(d.delta) > 0.05 && Math.abs(d.delta) <= 0.15);
    const lowPriority = data.filter(d => !highPriority.includes(d) && !mediumPriority.includes(d));



    return (
        <div className="p-8 max-w-[1600px] mx-auto space-y-6">

            {/* Header for content */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <h2 className="text-xl font-bold text-slate-800">Insights <span className="text-slate-400 font-normal">({data.length})</span></h2>
                </div>
                <div>
                    <button className="bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200/50 px-4 py-2 rounded-xl text-sm font-semibold transition-all shadow-sm">
                        Choose Channel
                    </button>
                </div>
            </div>

            {/* Priority Cards */}
            <div className="grid grid-cols-3 gap-6">
                {/* High Priority */}
                <div className="bg-gradient-to-br from-white to-red-50 rounded-2xl p-6 border border-red-100 shadow-[0_4px_20px_-4px_rgba(239,68,68,0.15)] hover:shadow-[0_8px_25px_-4px_rgba(239,68,68,0.25)] transition-all cursor-pointer group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <div className="w-24 h-24 bg-red-500 rounded-full blur-2xl"></div>
                    </div>
                    <div className="flex justify-between items-start mb-6 relative z-10">
                        <h3 className="font-bold text-red-900 flex items-center gap-2 text-lg">
                            High Priority <span className="text-red-600/60 font-medium">({highPriority.length})</span>
                        </h3>
                        <div className="bg-white/60 p-1.5 rounded-full hover:bg-white transition-colors">
                            <ChevronRight size={18} className="text-red-400 group-hover:text-red-600" />
                        </div>
                    </div>
                    <div className="flex justify-between items-end relative z-10">
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">{highPriority.filter(p => p.rationale === 'Manager Review').length}</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">New</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">8</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Review</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">4</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Done</span>
                        </div>
                    </div>
                </div>

                {/* Medium Priority */}
                <div className="bg-gradient-to-br from-white to-amber-50 rounded-2xl p-6 border border-amber-100 shadow-[0_4px_20px_-4px_rgba(245,158,11,0.15)] hover:shadow-[0_8px_25px_-4px_rgba(245,158,11,0.25)] transition-all cursor-pointer group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <div className="w-24 h-24 bg-amber-500 rounded-full blur-2xl"></div>
                    </div>
                    <div className="flex justify-between items-start mb-6 relative z-10">
                        <h3 className="font-bold text-amber-900 flex items-center gap-2 text-lg">
                            Medium Priority <span className="text-amber-600/60 font-medium">({mediumPriority.length})</span>
                        </h3>
                        <div className="bg-white/60 p-1.5 rounded-full hover:bg-white transition-colors">
                            <ChevronRight size={18} className="text-amber-400 group-hover:text-amber-600" />
                        </div>
                    </div>
                    <div className="flex justify-between items-end relative z-10">
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">{mediumPriority.length}</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">New</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">2</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Review</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">2</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Done</span>
                        </div>
                    </div>
                </div>

                {/* Low Priority */}
                <div className="bg-gradient-to-br from-white to-blue-50 rounded-2xl p-6 border border-blue-100 shadow-[0_4px_20px_-4px_rgba(59,130,246,0.15)] hover:shadow-[0_8px_25px_-4px_rgba(59,130,246,0.25)] transition-all cursor-pointer group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <div className="w-24 h-24 bg-blue-500 rounded-full blur-2xl"></div>
                    </div>
                    <div className="flex justify-between items-start mb-6 relative z-10">
                        <h3 className="font-bold text-blue-900 flex items-center gap-2 text-lg">
                            Low Priority <span className="text-blue-600/60 font-medium">({lowPriority.length})</span>
                        </h3>
                        <div className="bg-white/60 p-1.5 rounded-full hover:bg-white transition-colors">
                            <ChevronRight size={18} className="text-blue-400 group-hover:text-blue-600" />
                        </div>
                    </div>
                    <div className="flex justify-between items-end relative z-10">
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">{lowPriority.length}</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">New</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">0</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Review</span>
                        </div>
                        <div className="text-center">
                            <span className="block text-3xl font-extrabold text-slate-800">0</span>
                            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Done</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* KPI Section */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl border border-blue-100/50 shadow-[0_4px_20px_-4px_rgba(59,130,246,0.05)] p-6 hover:shadow-blue-100/50 transition-shadow duration-300">
                <div className="flex items-center gap-2 mb-6">
                    <h3 className="font-bold text-slate-900 text-lg">Baseline Price : with recommended adjustments</h3>
                </div>

                {/* Metrics Row */}
                <div className="grid grid-cols-4 gap-8 mb-8 border-b border-slate-100 pb-8">
                    <div>
                        <p className="text-xs font-bold text-slate-500 mb-1">Stock out %</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold text-slate-800">17%</span>
                            <span className="text-xs font-bold text-red-500">↑ 0.75% this week</span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">Today</p>
                    </div>
                    <div>
                        <p className="text-xs font-bold text-slate-500 mb-1">Days of Supply (DoS)</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold text-slate-800">23</span>
                            <span className="text-xs font-bold text-emerald-500">↓ 3 days this week</span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">Today</p>
                    </div>
                    <div>
                        <p className="text-xs font-bold text-slate-500 mb-1">Fill rate %</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold text-slate-800">82%</span>
                            <span className="text-xs font-bold text-red-500">↓ 0.75% this week</span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">Today</p>
                    </div>
                    <div>
                        <p className="text-xs font-bold text-slate-500 mb-1">OTIF</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold text-slate-800">85%</span>
                            <span className="text-xs font-bold text-emerald-500">↑ 0.75% this week</span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">Today</p>
                    </div>
                </div>

                {/* Recommendation Banner */}
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 flex items-center justify-between mb-8">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                            <AlertCircle size={18} />
                        </div>
                        <div>
                            <h4 className="text-sm font-bold text-blue-900">37 recommendations identified</h4>
                            <p className="text-xs text-blue-700">Accepting all recommendations will improve your demand plan by reducing forecast errors, minimizing bias, and aligning demand with real trends.</p>
                        </div>
                    </div>
                    <button className="bg-white border border-blue-200 text-blue-600 px-4 py-2 rounded-lg text-sm font-bold hover:bg-blue-50 transition-colors shadow-sm">
                        ✨ Learn
                    </button>
                </div>

            </div>
        </div>
    );
};

export const InsightsCharts = () => {
    const chartData = [
        { name: 'Week 1', base: 100, recommended: 102, comp: 98 },
        { name: 'Week 2', base: 100, recommended: 104, comp: 99 },
        { name: 'Week 3', base: 101, recommended: 103, comp: 100 },
        { name: 'Week 4', base: 101, recommended: 105, comp: 102 },
        { name: 'Week 5', base: 102, recommended: 108, comp: 104 },
        { name: 'Week 6', base: 102, recommended: 107, comp: 105 },
        { name: 'Week 7', base: 103, recommended: 110, comp: 106 },
        { name: 'Today', base: 103, recommended: 112, comp: 108 },
    ];

    return (
        <div className="bg-white/90 backdrop-blur-sm rounded-2xl border border-blue-100/50 shadow-sm p-6 h-full">
            <h3 className="font-bold text-slate-900 text-lg mb-4">Price Trend Analysis</h3>
            <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorRec" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748B' }} dy={10} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748B' }} />
                        <Tooltip
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        />
                        <Area type="monotone" dataKey="recommended" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRec)" name="Recommended Price" />
                        <Line type="monotone" dataKey="base" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Base Price" />
                        <Line type="monotone" dataKey="comp" stroke="#f59e0b" strokeWidth={2} dot={false} name="Competitor Price" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default InsightsContent;
