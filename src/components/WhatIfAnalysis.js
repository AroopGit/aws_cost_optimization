import React, { useState, useEffect } from 'react';
import { runWhatIfAnalysis, fetchSkus, fetchChannels, fetchRegions } from '../api';

const WhatIfAnalysis = ({ initialSku, initialChannel, initialRegion }) => {
    const [sku, setSku] = useState(initialSku || '');
    const [channel, setChannel] = useState(initialChannel || '');
    const [region, setRegion] = useState(initialRegion || '');



    // Manual params
    const [manualParams, setManualParams] = useState({
        competitor_price: '',
        promo_depth_pct: '',
        inv_days: '',
    });

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Update local state if props change
    useEffect(() => {
        if (initialSku) setSku(initialSku);
        if (initialChannel) setChannel(initialChannel);
        if (initialRegion) setRegion(initialRegion);
    }, [initialSku, initialChannel, initialRegion]);

    const handleManualChange = (e) => {
        setManualParams({ ...manualParams, [e.target.name]: e.target.value });
    };

    const handleRun = async () => {
        if (!sku || !channel || !region) {
            setError("Please select SKU, Channel, and Region.");
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const payload = {
                sku_id: sku,
                channel,
                region,
            };

            if (manualParams.competitor_price) payload.competitor_price = parseFloat(manualParams.competitor_price);
            if (manualParams.promo_depth_pct) payload.promo_depth_pct = parseFloat(manualParams.promo_depth_pct);
            if (manualParams.inv_days) payload.inv_days = parseFloat(manualParams.inv_days);

            const data = await runWhatIfAnalysis(payload);
            setResult(data);
        } catch (err) {
            setError(err.message || "Analysis failed");
        } finally {
            setLoading(false);
        }
    };

    const formatCurrency = (val) => {
        if (val === undefined || val === null) return '-';
        return `₹${val.toFixed(2)}`;
    };

    const formatPct = (val) => {
        if (val === undefined || val === null) return '-';
        return `${val.toFixed(1)}%`;
    };

    return (
        <div className="bg-gradient-to-br from-brand-dark to-black rounded-xl p-6 space-y-6 shadow-2xl ring-1 ring-white/10 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-3 opacity-5 pointer-events-none">
                <svg className="w-64 h-64 text-blue-400" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" /></svg>
            </div>
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-white/10 rounded-lg backdrop-blur-sm">
                        <svg className="w-5 h-5 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                        </svg>
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white tracking-tight">What-If Scenario Analysis</h2>
                        <div className="text-xs text-white/70 font-medium">Simulate market changes & predict outcomes</div>
                    </div>
                </div>
            </div>

            {/* Filters removed - moved to Sidebar */}

            {/* Input Section */}
            <div className="bg-brand-dark/40 rounded-xl p-5 border border-brand-DEFAULT/20 shadow-inner backdrop-blur-sm">
                <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-white/80">Competitor Price</label>
                        <div className="relative">
                            <span className="absolute left-3 top-2 text-white/60 text-xs">₹</span>
                            <input
                                type="number"
                                name="competitor_price"
                                value={manualParams.competitor_price}
                                onChange={handleManualChange}
                                placeholder="290"
                                className="w-full bg-white/10 border border-white/20 rounded-lg pl-6 pr-3 py-1.5 text-sm text-white focus:ring-2 focus:ring-brand-light focus:border-transparent outline-none placeholder-white/30"
                            />
                        </div>
                    </div>
                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-white/80">Promo Depth</label>
                        <div className="relative">
                            <input
                                type="number"
                                name="promo_depth_pct"
                                value={manualParams.promo_depth_pct}
                                onChange={handleManualChange}
                                placeholder="15"
                                className="w-full bg-white/10 border border-white/20 rounded-lg pl-3 pr-8 py-1.5 text-sm text-white focus:ring-2 focus:ring-brand-light focus:border-transparent outline-none placeholder-white/30"
                            />
                            <span className="absolute right-3 top-2 text-white/60 text-xs">%</span>
                        </div>
                    </div>
                    <div className="space-y-1.5">
                        <label className="block text-xs font-medium text-white/80">Inventory Days</label>
                        <input
                            type="number"
                            name="inv_days"
                            value={manualParams.inv_days}
                            onChange={handleManualChange}
                            placeholder="8"
                            className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-1.5 text-sm text-white focus:ring-2 focus:ring-brand-light focus:border-transparent outline-none placeholder-white/30"
                        />
                    </div>
                </div>

                <div className="flex justify-end pt-2 border-t border-white/5">
                    <button
                        onClick={handleRun}
                        disabled={loading}
                        className="px-6 py-2 bg-gradient-to-r from-brand-DEFAULT to-brand-light hover:from-brand-dark hover:to-brand-DEFAULT text-white text-sm font-bold rounded-lg transition-all shadow-lg shadow-brand-dark/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transform active:scale-95"
                    >
                        {loading ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Running Simulation...
                            </>
                        ) : (
                            <>
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                Run Simulation
                            </>
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mt-3 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-xs font-medium text-red-200 flex items-center gap-2">
                        <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        {error}
                    </div>
                )}
            </div>

            {/* Results */}
            {result && (
                <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="grid grid-cols-2 gap-6">
                        {/* Original */}
                        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-4">
                                <div className="w-2 h-2 rounded-full bg-slate-400" />
                                <h3 className="font-semibold text-slate-200">Original Baseline</h3>
                            </div>
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/60">Recommended Price</span>
                                    <span className="font-medium text-white">{formatCurrency(result.original.selection.price_recommended)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/60">Margin</span>
                                    <span className="font-medium text-emerald-400">{formatPct(result.original.selection.margin_pct)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/60">Predicted Units</span>
                                    <span className="font-medium text-white">{result.original.candidates_detail[result.original.selection.source_key]?.pred_units?.toFixed(0) || '-'}</span>
                                </div>
                            </div>
                        </div>

                        {/* What-If */}
                        <div className="bg-brand-DEFAULT/20 rounded-xl p-4 border border-brand-DEFAULT/30 relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-4 opacity-10">
                                <svg className="w-24 h-24 text-emerald-500" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                            </div>
                            <div className="flex items-center gap-2 mb-4 relative z-10">
                                <div className="w-2 h-2 rounded-full bg-emerald-400" />
                                <h3 className="font-semibold text-emerald-100">What-If Scenario</h3>
                            </div>
                            <div className="space-y-3 relative z-10">
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/70">Recommended Price</span>
                                    <span className="font-bold text-white text-lg">{formatCurrency(result.what_if.selection.price_recommended)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/70">Margin</span>
                                    <span className="font-medium text-emerald-400">{formatPct(result.what_if.selection.margin_pct)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-white/70">Predicted Units</span>
                                    <span className="font-medium text-white">{result.what_if.candidates_detail[result.what_if.selection.source_key]?.pred_units?.toFixed(0) || '-'}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Delta & Explanation */}
                    <div className="bg-slate-900/30 rounded-xl p-4 border border-slate-700/30">
                        <h4 className="text-sm font-medium text-white mb-2">Analysis Summary</h4>
                        <p className="text-sm text-white/80 leading-relaxed">
                            {result.explanation || result.nl_summary || "No summary available."}
                        </p>
                        {result.delta && (
                            <div className="mt-3 pt-3 border-t border-slate-700/50 flex gap-4 text-xs">
                                <div className="text-white/60">
                                    Price Delta: <span className={result.delta.price_optimal > 0 ? 'text-emerald-400' : 'text-red-400'}>
                                        {result.delta.price_optimal > 0 ? '+' : ''}{result.delta.price_optimal?.toFixed(2)}
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default WhatIfAnalysis;
