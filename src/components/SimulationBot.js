import React, { useState, useMemo } from 'react';
import { formatCurrency, formatPercent } from '../utils/dataUtils';

const SimulationBot = ({ analysisRows, onClose }) => {
  const [selectedSku, setSelectedSku] = useState('');
  const [simulationPrice, setSimulationPrice] = useState(0);
  const [simulationResults, setSimulationResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const selectedRow = useMemo(() => {
    return analysisRows.find((row) => row.sku === selectedSku) || null;
  }, [analysisRows, selectedSku]);

  const handleSimulate = async () => {
    if (!selectedRow) return;

    setIsRunning(true);
    setSimulationPrice(selectedRow.basePrice);

    // Simulate bot analysis
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const price = simulationPrice || selectedRow.basePrice;
    const delta = (price - selectedRow.basePrice) / selectedRow.basePrice;
    const demandFactor = Math.max(0.2, 1 - selectedRow.elasticity * delta);
    const projectedVolume = Math.round(selectedRow.baselineVolume * demandFactor);
    const projectedRevenue = +(projectedVolume * price).toFixed(0);
    const marginPerUnit = price * selectedRow.baselineMarginPct;
    const projectedMargin = +(projectedVolume * marginPerUnit).toFixed(0);
    const baselineMargin = selectedRow.baselineVolume * selectedRow.basePrice * selectedRow.baselineMarginPct;
    const projectedRoi = +((projectedMargin - baselineMargin) / baselineMargin).toFixed(2);

    setSimulationResults({
      price,
      delta,
      projectedVolume,
      projectedRevenue,
      projectedMargin,
      projectedRoi,
      baselineRevenue: selectedRow.baselineVolume * selectedRow.basePrice,
      baselineMargin,
    });

    setIsRunning(false);
  };

  const uniqueSkus = useMemo(() => {
    return [...new Set(analysisRows.map((row) => row.sku))];
  }, [analysisRows]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 px-4">
      <div className="w-full max-w-4xl rounded-2xl bg-white shadow-2xl ring-1 ring-slate-200 max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
              <svg
                className="h-6 w-6 text-blue-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Simulation Bot</h2>
              <p className="text-xs text-slate-500">AI-powered price simulation and analysis</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div className="space-y-6 px-6 py-6">
          <div>
            <label className="mb-2 block text-sm font-medium text-slate-900">
              Select SKU to Simulate
            </label>
            <select
              value={selectedSku}
              onChange={(e) => {
                setSelectedSku(e.target.value);
                setSimulationResults(null);
                const row = analysisRows.find((r) => r.sku === e.target.value);
                if (row) setSimulationPrice(row.basePrice);
              }}
              className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="">Choose a SKU...</option>
              {uniqueSkus.map((sku) => (
                <option key={sku} value={sku}>
                  {sku}
                </option>
              ))}
            </select>
          </div>

          {selectedRow && (
            <>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <h3 className="mb-3 text-sm font-semibold text-slate-900">Current Recommendation</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-500">Base Price:</span>
                    <span className="ml-2 font-medium text-slate-900">
                      {formatCurrency(selectedRow.basePrice)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Recommended:</span>
                    <span className="ml-2 font-medium text-emerald-600">
                      {formatCurrency(selectedRow.recommendedPrice)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Elasticity:</span>
                    <span className="ml-2 font-medium text-slate-900">
                      {selectedRow.elasticity.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500">Confidence:</span>
                    <span className="ml-2 font-medium text-slate-900">
                      {selectedRow.confidence}%
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <label className="mb-2 flex items-center justify-between text-sm text-slate-600">
                  <span>Simulated Price</span>
                  <span className="font-medium text-slate-900">
                    {formatCurrency(simulationPrice || selectedRow.basePrice)}
                  </span>
                </label>
                <input
                  type="range"
                  min={+(selectedRow.basePrice * 0.8).toFixed(2)}
                  max={+(selectedRow.basePrice * 1.2).toFixed(2)}
                  step="0.05"
                  value={simulationPrice || selectedRow.basePrice}
                  onChange={(e) => {
                    setSimulationPrice(+e.target.value);
                    setSimulationResults(null);
                  }}
                  className="w-full accent-blue-500"
                />
                <div className="mt-2 flex justify-between text-xs text-slate-500">
                  <span>{formatCurrency(+(selectedRow.basePrice * 0.8).toFixed(2))}</span>
                  <span>Baseline: {formatCurrency(selectedRow.basePrice)}</span>
                  <span>{formatCurrency(+(selectedRow.basePrice * 1.2).toFixed(2))}</span>
                </div>
              </div>

              <button
                onClick={handleSimulate}
                disabled={isRunning}
                className="w-full rounded-lg border border-blue-300 bg-blue-500 px-4 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-blue-600 disabled:opacity-50"
              >
                {isRunning ? 'Running Simulation...' : 'Run Simulation'}
              </button>

              {simulationResults && (
                <div className="space-y-4 rounded-lg border border-blue-200 bg-blue-50 p-4">
                  <h3 className="text-sm font-semibold text-blue-900">Simulation Results</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="rounded-lg bg-white p-3">
                      <div className="text-xs text-slate-500">Price Change</div>
                      <div className="mt-1 text-base font-semibold text-slate-900">
                        {formatPercent(simulationResults.delta)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-white p-3">
                      <div className="text-xs text-slate-500">Projected Volume</div>
                      <div className="mt-1 text-base font-semibold text-slate-900">
                        {simulationResults.projectedVolume.toLocaleString()} units
                      </div>
                    </div>
                    <div className="rounded-lg bg-white p-3">
                      <div className="text-xs text-slate-500">Projected Revenue</div>
                      <div className="mt-1 text-base font-semibold text-slate-900">
                        {formatCurrency(simulationResults.projectedRevenue)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-white p-3">
                      <div className="text-xs text-slate-500">Projected Margin</div>
                      <div className="mt-1 text-base font-semibold text-slate-900">
                        {formatCurrency(simulationResults.projectedMargin)}
                      </div>
                    </div>
                    <div className="col-span-2 rounded-lg bg-white p-3">
                      <div className="text-xs text-slate-500">Projected ROI</div>
                      <div className="mt-1 text-lg font-semibold text-blue-600">
                        {(simulationResults.projectedRoi * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 rounded-lg bg-white p-3 text-xs text-slate-600">
                    <div className="mb-2 font-medium text-slate-900">Comparison:</div>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>Baseline Revenue:</span>
                        <span className="font-medium">{formatCurrency(simulationResults.baselineRevenue)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Projected Revenue:</span>
                        <span className="font-medium text-blue-600">
                          {formatCurrency(simulationResults.projectedRevenue)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Difference:</span>
                        <span className={`font-medium ${simulationResults.projectedRevenue > simulationResults.baselineRevenue
                            ? 'text-emerald-600'
                            : 'text-rose-600'
                          }`}>
                          {formatCurrency(
                            simulationResults.projectedRevenue - simulationResults.baselineRevenue
                          )}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimulationBot;

