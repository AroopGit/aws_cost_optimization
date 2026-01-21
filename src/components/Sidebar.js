import React, { useState, useEffect } from 'react';
import { Settings, Filter } from 'lucide-react';
import { fetchSkus, fetchChannels, fetchRegions } from '../api';

const Sidebar = ({ selectedSku, selectedChannel, selectedRegion, onFilterChange, onRun, executionSteps = [] }) => {
    const [skus, setSkus] = useState([]);
    const [channels, setChannels] = useState([]);
    const [regions, setRegions] = useState([]);

    useEffect(() => {
        const loadOptions = async () => {
            try {
                const [s, c, r] = await Promise.all([fetchSkus(), fetchChannels(), fetchRegions()]);
                setSkus(s.skus || []);
                setChannels(c.channels || []);
                setRegions(r.regions || []);
            } catch (e) {
                console.error("Failed to load options", e);
            }
        };
        loadOptions();
    }, []);

    const handleChange = (key, value) => {
        onFilterChange({ [key]: value });
    };

    const getStatusBadge = (status) => {
        switch (status) {
            case 'completed': return 'Completed';
            case 'running': return 'Running...';
            default: return 'Pending';
        }
    };

    return (
        <div className="w-64 flex-shrink-0 flex flex-col py-6 bg-gradient-to-b from-[#89CFF0] to-[#DDF3FA] border-r border-white/20 h-screen sticky top-0 shadow-lg z-50 overflow-y-auto">


            <div className="px-6 mb-6">
                <div className="flex items-center gap-2 text-brand-dark mb-4">
                    <Filter size={16} />
                    <span className="text-xs font-bold uppercase tracking-wider">Global Filters</span>
                </div>

                <div className="space-y-4">
                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-slate-500">SKU</label>
                        <select
                            value={selectedSku}
                            onChange={(e) => handleChange('sku', e.target.value)}
                            className="w-full bg-white border border-brand-dark/10 rounded-lg px-3 py-2 text-sm text-slate-700 focus:bg-white focus:border-brand-DEFAULT focus:outline-none focus:ring-1 focus:ring-brand-DEFAULT transition-all custom-select shadow-sm"
                        >
                            <option value="">Select SKU</option>
                            {skus.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                    </div>

                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-slate-500">Channel</label>
                        <select
                            value={selectedChannel}
                            onChange={(e) => handleChange('channel', e.target.value)}
                            className="w-full bg-white border border-brand-dark/10 rounded-lg px-3 py-2 text-sm text-slate-700 focus:bg-white focus:border-brand-DEFAULT focus:outline-none focus:ring-1 focus:ring-brand-DEFAULT transition-all custom-select shadow-sm"
                        >
                            <option value="">Select Channel</option>
                            {channels.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>

                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-slate-500">Region</label>
                        <select
                            value={selectedRegion}
                            onChange={(e) => handleChange('region', e.target.value)}
                            className="w-full bg-white border border-brand-dark/10 rounded-lg px-3 py-2 text-sm text-slate-700 focus:bg-white focus:border-brand-DEFAULT focus:outline-none focus:ring-1 focus:ring-brand-DEFAULT transition-all custom-select shadow-sm"
                        >
                            <option value="">Select Region</option>
                            {regions.map(r => <option key={r} value={r}>{r}</option>)}
                        </select>
                    </div>
                </div>
            </div>

            <div className="px-6 mb-6">
                <button
                    onClick={() => onRun && onRun()}
                    className="w-full py-2.5 px-4 bg-white text-brand-DEFAULT font-bold text-sm rounded-lg shadow-md hover:bg-brand-light/30 transition-all flex items-center justify-center gap-2"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                    Generate Analysis
                </button>
            </div>

            {/* Execution Steps */}
            {executionSteps.length > 0 && (
                <div className="px-3 mb-6">
                    <div className="rounded-xl bg-white/40 backdrop-blur-sm p-4 shadow-sm border border-white/30">
                        <div className="mb-3 flex items-center justify-between">
                            <h3 className="text-xs font-bold text-brand-dark uppercase tracking-wider">Agent Steps</h3>
                            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${executionSteps.filter(s => s.status === 'completed').length === 4
                                ? 'bg-blue-500 text-white'
                                : 'bg-blue-200 text-blue-700'
                                }`}>
                                {executionSteps.filter(s => s.status === 'completed').length === 4 ? '✓' : '...'}
                            </span>
                        </div>
                        <div className="space-y-2.5 relative">
                            <div className="absolute left-1.5 top-2 bottom-2 w-px bg-blue-200/50"></div>
                            {executionSteps.map((step, index) => (
                                <div key={index} className="relative pl-5">
                                    <div className={`absolute left-0 top-1 w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center bg-white z-10 ${step.status === 'completed' ? 'border-blue-500 bg-blue-500' :
                                        step.status === 'running' ? 'border-blue-400' :
                                            'border-blue-200'
                                        }`}>
                                        {step.status === 'completed' && <div className="w-1.5 h-1.5 rounded-full bg-white"></div>}
                                    </div>
                                    <div className={`rounded-lg border p-2 transition-all ${step.status === 'running' ? 'border-blue-400 bg-blue-50/50 shadow-sm' :
                                        step.status === 'completed' ? 'border-blue-100 bg-white/60' :
                                            'border-white/20 bg-white/20'
                                        }`}>
                                        <div className="flex justify-between items-center">
                                            <p className="text-[11px] font-bold text-brand-dark truncate leading-tight">{step.name.split(' - ')[0]}</p>
                                        </div>
                                        {step.result && step.result !== 'Awaiting run' && (
                                            <div className="mt-1 text-[10px] font-bold text-blue-600">
                                                {step.result}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <div className="mt-auto flex flex-col gap-6 items-center px-6">
                <div className="w-full h-px bg-white/20 mb-4"></div>
                <button className="flex items-center gap-3 w-full p-2 text-slate-500 hover:text-brand-dark hover:bg-white/40 rounded-xl transition-all duration-300 group">
                    <Settings size={18} />
                    <span className="text-sm font-medium">Settings</span>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
