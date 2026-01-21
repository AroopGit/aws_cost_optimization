import React, { useState, useEffect } from 'react';
import { fetchSingleSkuPricing, fetchSkus, fetchChannels, fetchRegions, runWhatIfAnalysis } from '../api';
import DemandCurve from './widgets/DemandCurve';
import ElasticityGauge from './widgets/ElasticityGauge';
import SeasonalityTimeline from './widgets/SeasonalityTimeline';

const WhatIfModal = ({ isOpen, onClose, onRun, sku }) => {
    const [formData, setFormData] = useState({
        promo_depth_pct: '',
        competitor_price: '',
        inv_days: '',
        seasonality_index: '',
        festival_lift: '',
        base_uplift_multiplicative: '',
        min_margin_pct: ''
    });

    if (!isOpen) return null;

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = () => {
        const payload = {};
        Object.keys(formData).forEach(key => {
            if (formData[key] !== '') {
                if (key === 'min_margin_pct') {
                    payload.sop_overrides = { min_margin_pct: parseFloat(formData[key]) };
                } else {
                    payload[key] = parseFloat(formData[key]);
                }
            }
        });
        onRun(payload);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="w-full max-w-md rounded-2xl bg-white shadow-2xl animate-in fade-in zoom-in duration-200 overflow-hidden">
                <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4 bg-slate-50">
                    <h3 className="text-lg font-bold text-slate-900">What-If Scenario: {sku}</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-slate-600">Close</button>
                </div>
                <div className="max-h-[70vh] overflow-y-auto px-6 py-6 space-y-4">
                    <p className="text-sm text-slate-500 mb-4">Enter overrides (blank to skip)</p>

                    {[
                        { label: 'Promo depth % override', name: 'promo_depth_pct', placeholder: 'e.g., 15' },
                        { label: 'Competitor price override', name: 'competitor_price', placeholder: 'e.g., 125.50' },
                        { label: 'Inventory days override', name: 'inv_days', placeholder: 'e.g., 45' },
                        { label: 'Seasonality index override', name: 'seasonality_index', placeholder: 'e.g., 1.2' },
                        { label: 'Festival lift override', name: 'festival_lift', placeholder: 'e.g., 1.1' },
                        { label: 'Base uplift multiplicative', name: 'base_uplift_multiplicative', placeholder: 'e.g., 1.05' },
                        { label: 'Override SOP min_margin_pct', name: 'min_margin_pct', placeholder: 'e.g., 10' }
                    ].map(field => (
                        <div key={field.name}>
                            <label className="block text-xs font-medium text-slate-700 mb-1">{field.label}</label>
                            <input
                                type="number"
                                name={field.name}
                                value={formData[field.name]}
                                onChange={handleChange}
                                placeholder={field.placeholder}
                                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                            />
                        </div>
                    ))}
                </div>
                <div className="border-t border-slate-100 px-6 py-4 flex justify-end gap-3 bg-slate-50">
                    <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-200 rounded-lg">Cancel</button>
                    <button onClick={handleSubmit} className="px-4 py-2 text-sm font-bold text-white bg-emerald-500 hover:bg-emerald-600 rounded-lg shadow-sm">Run What-If</button>
                </div>
            </div>
        </div>
    );
};

const AgentPricingView = ({
    initialSku = '',
    initialChannel = '',
    initialRegion = '',
    onFilterChange,
    pricingData,
    loading,
    executionSteps,
    onGeneratePrice,
    onRunWhatIf
}) => {
    const [selectedAction, setSelectedAction] = useState('');
    const [showWhatIfModal, setShowWhatIfModal] = useState(false);

    // Filter local state removed - using props from App.js/Sidebar directly?
    // Actually the parent App.js passes initialSku etc, but also passes `onFilterChange`.
    // The previous implementation had local state syncing.
    // I should rely on props `initialSku` etc as the checked sources.
    // The `handleGeneratePrice` in previous code used local state `selectedSku`.
    // I should simply use the props `initialSku` (which are actually `selectedSku` from App).

    const handleActionChange = (e) => {
        const action = e.target.value;
        setSelectedAction(action);
        if (action === 'what-if') {
            // Validations are now done by parent or just passed
            if (!initialSku || !initialChannel || !initialRegion) {
                alert('Please select SKU, Channel, and Region first');
                setSelectedAction('');
                return;
            }
            setShowWhatIfModal(true);
        }
    };

    const handleRunWhatIfLocal = (overrides) => {
        onRunWhatIf(overrides);
        setShowWhatIfModal(false);
    }

    // Helpers
    const getBasePrice = () => pricingData?.candidates?.price_base || 0;
    const getOptimalPrice = () => {
        const val = pricingData?.candidates?.price_optimal;
        if (val && val > 0) return val;
        const base = getBasePrice();
        return base > 0 ? base / 0.95 : 0;
    };
    const getAggressivePrice = () => {
        const val = pricingData?.candidates?.price_aggressive;
        if (val && val > 0) return val;
        const opt = getOptimalPrice();
        return opt > 0 ? opt * 1.05 : 0;
    };

    const renderAgentCard = (agentKey, title) => {
        if (!pricingData?.agent_outputs?.[agentKey]) return null;
        const output = pricingData.agent_outputs[agentKey];
        const candidates = output.candidates || {};
        const conservative = candidates.conservative || (output.price * 0.98);
        const neutral = candidates.neutral || output.price;
        const aggressive = candidates.aggressive || (output.price * 1.05);

        return (
            <div className={`rounded-xl p-4 border ${pricingData.isWhatIf ? 'bg-brand-dark/80 border-brand-DEFAULT/50' : 'bg-brand-dark border-white/10'}`}>
                <div className="flex justify-between items-start mb-3">
                    <p className="text-sm font-semibold text-slate-300">{title}</p>
                    <span className="text-xs text-emerald-400 font-mono">
                        ₹{output.price?.toFixed(2)}
                    </span>
                </div>
                <div className="space-y-2 mb-3">
                    <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Conservative</span>
                        <span className="text-slate-300">₹{conservative?.toFixed(2) || '-'}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Optimal</span>
                        <span className="text-slate-300">₹{neutral?.toFixed(2) || '-'}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Aggressive</span>
                        <span className="text-slate-300">₹{aggressive?.toFixed(2) || '-'}</span>
                    </div>
                </div>
                <p className="text-[10px] text-slate-500 border-t border-slate-700 pt-2 truncate">
                    {output.reason}
                </p>
            </div>
        );
    };

    return (
        <div className="space-y-4">


            <div className="grid grid-cols-12 gap-4">
                <div className="col-span-12 space-y-4">
                    {/* Main Price Cards */}
                    {pricingData ? (
                        <div className={`rounded-xl p-4 shadow-lg ${pricingData.isWhatIf ? 'bg-gradient-to-br from-brand-DEFAULT via-brand-dark to-black' : 'bg-gradient-to-br from-brand-light via-brand-DEFAULT to-brand-dark'}`}>
                            <div className="mb-4 flex justify-between items-start">
                                <div>
                                    <h2 className="text-xl font-bold text-white">
                                        {initialSku} • {initialChannel} {initialRegion}
                                    </h2>
                                    <p className="text-xs text-white/80 uppercase tracking-wide">
                                        {pricingData.isWhatIf ? 'What-If Scenario Analysis' : 'AI Pricing Recommendations'}
                                    </p>
                                </div>
                                {pricingData.isWhatIf && (
                                    <span className="bg-white/20 text-white px-3 py-1 rounded-full text-xs font-bold">
                                        SCENARIO ACTIVE
                                    </span>
                                )}
                            </div>

                            <div className="grid grid-cols-3 gap-3 mb-4">
                                <div className="rounded-lg bg-white/20 p-3 backdrop-blur-sm border border-white/10">
                                    <p className="mb-1 text-xs font-medium text-white/70">Base</p>
                                    <p className="text-2xl font-bold text-white">₹{getBasePrice().toFixed(2)}</p>
                                </div>
                                <div className="rounded-lg bg-white/30 p-3 backdrop-blur-sm border border-white/20 shadow-inner">
                                    <p className="mb-1 text-xs font-medium text-white/90">Optimal</p>
                                    <p className="text-3xl font-bold text-white">₹{getOptimalPrice().toFixed(2)}</p>
                                </div>
                                <div className="rounded-lg bg-white/20 p-3 backdrop-blur-sm border border-white/10">
                                    <p className="mb-1 text-xs font-medium text-white/70">Aggressive</p>
                                    <p className="text-2xl font-bold text-white">₹{getAggressivePrice().toFixed(2)}</p>
                                </div>
                            </div>

                            {/* Final Decision Block */}
                            <div className="flex items-center justify-between bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                                <div>
                                    <p className="text-xs font-medium text-white/80">Final Decision</p>
                                    <div className="flex items-baseline gap-2">
                                        <span className="text-xl font-bold text-white">₹{pricingData.selection?.price_recommended?.toFixed(2)}</span>
                                        <span className="text-xs text-white/70">
                                            ({pricingData.selection?.change_pct > 0 ? '+' : ''}{pricingData.selection?.change_pct}%)
                                        </span>
                                    </div>
                                </div>
                                <button className="rounded-lg bg-white text-brand-dark px-4 py-1.5 text-sm font-bold shadow-lg hover:bg-brand-light/20 transition">
                                    Approve
                                </button>
                            </div>

                            {pricingData.delta && (
                                <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-white/80 border-t border-white/10 pt-3">
                                    <div>
                                        <span className="block opacity-70">Rev Impact</span>
                                        <span className="font-bold">{pricingData.delta.revenue_change_pct > 0 ? '+' : ''}{pricingData.delta.revenue_change_pct}%</span>
                                    </div>
                                    <div>
                                        <span className="block opacity-70">Vol Impact</span>
                                        <span className="font-bold">{pricingData.delta.units_change_pct > 0 ? '+' : ''}{pricingData.delta.units_change_pct}%</span>
                                    </div>
                                    <div>
                                        <span className="block opacity-70">Margin Impact</span>
                                        <span className="font-bold">{pricingData.delta.margin_change_pct > 0 ? '+' : ''}{pricingData.delta.margin_change_pct}%</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-12 text-center h-64 flex flex-col items-center justify-center">
                            <p className="text-lg font-medium text-slate-600">Ready to Generate</p>
                            <p className="mt-2 text-sm text-slate-500">Select filters and click Generate Price</p>
                        </div>
                    )}

                    {/* Agent Insights Grid */}
                    {pricingData && (
                        <div className="rounded-2xl bg-brand-dark p-6 shadow-xl">
                            <div className="mb-4 flex items-center justify-between">
                                <h3 className="text-lg font-bold text-white">Multi-Agent Analysis</h3>
                                <span className="text-xs text-slate-400">Detailed breakdown by agent strategy</span>
                            </div>
                            <div className="grid grid-cols-4 gap-4">
                                {renderAgentCard('base', 'Base Agent')}
                                {renderAgentCard('promo', 'Promo Agent')}
                                {renderAgentCard('comp', 'Competitor Agent')}
                                {renderAgentCard('inventory', 'Inventory Agent')}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <WhatIfModal
                isOpen={showWhatIfModal}
                onClose={() => setShowWhatIfModal(false)}
                onRun={handleRunWhatIfLocal}
                sku={initialSku}
            />
        </div>
    );
};

const AgentPricingCharts = ({ pricingData }) => {

    return (
        <div className="space-y-4 h-full">
            {/* Advanced Analytics */}
            {pricingData && (
                <div className="space-y-4">
                    <div className="rounded-2xl bg-white p-4 shadow-sm border border-slate-100">
                        <h4 className="text-xs font-bold text-slate-700 uppercase mb-3">Price Elasticity</h4>
                        <ElasticityGauge elasticity={pricingData.elasticity || -1.5} />
                    </div>
                    <div className="rounded-2xl bg-white p-4 shadow-sm border border-slate-100">
                        <h4 className="text-xs font-bold text-slate-700 uppercase mb-3">Demand Curve</h4>
                        <DemandCurve
                            currentPrice={pricingData.selection?.price_recommended || 100}
                            elasticity={pricingData.elasticity || -1.5}
                        />
                    </div>
                    <div className="rounded-2xl bg-white p-4 shadow-sm border border-slate-100">
                        <h4 className="text-xs font-bold text-slate-700 uppercase mb-3">Seasonality Trend</h4>
                        <SeasonalityTimeline seasonalityIndex={pricingData.agent_outputs?.base?.meta?.seasonality || 1.0} />
                    </div>
                </div>
            )}
        </div>
    );
};

export { AgentPricingCharts };
export default AgentPricingView;

