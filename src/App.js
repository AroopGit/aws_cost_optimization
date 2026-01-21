import React, { useState, useMemo, useEffect } from 'react';
import Dashboard, { DashboardCharts } from './components/Dashboard';
import LLMExplanationPanel from './components/LLMExplanationPanel';
import ForecastingWidget from './components/ForecastingWidget';
import SimulationBot from './components/SimulationBot';
import AgentPricingView, { AgentPricingCharts } from './components/AgentPricingView';
import WhatIfAnalysis from './components/WhatIfAnalysis';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardLayout from './components/DashboardLayout';
import InsightsContent, { InsightsCharts } from './components/InsightsContent';
import { fetchBatchPricing, fetchSingleSkuPricing, runWhatIfAnalysis } from './api';
import './App.css';

function App() {
  const [analysisRows, setAnalysisRows] = useState([]);
  const [selectedSku, setSelectedSku] = useState('');
  const [selectedChannel, setSelectedChannel] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedRow, setSelectedRow] = useState(null);
  const [showBot, setShowBot] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('Insights');

  // Agent/Pricing State
  const [pricingData, setPricingData] = useState(null);
  const [whatIfData, setWhatIfData] = useState(null);
  const [pricingLoading, setPricingLoading] = useState(false);
  const [executionSteps, setExecutionSteps] = useState([
    {
      name: 'Base Agent - Demand Forecasting',
      status: 'pending',
      description: 'Analyzing historical trend, seasonality, and base price uplift.',
      result: 'Awaiting run',
      agent: 'base'
    },
    {
      name: 'Promo Agent - Promotion Planning',
      status: 'pending',
      description: 'Evaluating promo depth vs SOP guardrails.',
      result: 'Pending promo plan',
      agent: 'promo'
    },
    {
      name: 'Competitor Agent - Market Monitoring',
      status: 'pending',
      description: 'Scanning market shelves to close pricing gaps.',
      result: 'Pending competitor scan',
      agent: 'competitor'
    },
    {
      name: 'Inventory Agent - Supply Guardrail',
      status: 'pending',
      description: 'Applying inventory signal adjustments before final price.',
      result: 'Pending execution',
      agent: 'inventory'
    }
  ]);

  // Fetch metadata and initial data on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const batchData = await fetchBatchPricing(120);

        const mappedRows = (batchData.results || []).map((row, idx) => ({
          id: `${row.sku_id}-${row.channel}-${row.region}-${idx}`,
          sku: row.sku_id,
          channel: row.channel,
          region: row.region,
          basePrice: row.price_base,
          recommendedPrice: row.final_price,
          delta: row.change_pct / 100,
          confidence: row.approval === 'Auto-approve' ? 95 : row.approval === 'Manager Review' ? 75 : 50,
          projectedRoi: 12.5,
          rationale: row.approval,
          elasticity: -1.2,
          baselineVolume: 1000,
          baselineMarginPct: row.margin_pct / 100,
        }));
        setAnalysisRows(mappedRows);
      } catch (err) {
        console.error("Failed to load data:", err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const filteredRows = useMemo(() => {
    let filtered = analysisRows;
    if (selectedSku) {
      filtered = filtered.filter((row) => row.sku === selectedSku);
    }
    if (selectedChannel) {
      filtered = filtered.filter((row) => row.channel === selectedChannel);
    }
    if (selectedRegion) {
      filtered = filtered.filter((row) => row.region === selectedRegion);
    }
    return filtered;
  }, [analysisRows, selectedSku, selectedChannel, selectedRegion]);

  const handleFilterChange = (filters) => {
    if (filters.sku !== undefined) setSelectedSku(filters.sku);
    if (filters.channel !== undefined) setSelectedChannel(filters.channel);
    if (filters.region !== undefined) setSelectedRegion(filters.region);
  };

  const handleRun = async () => {
    setLoading(true);
    try {
      const batchData = await fetchBatchPricing(120, selectedSku, selectedChannel, selectedRegion);
      const mappedRows = (batchData.results || []).map((row, idx) => ({
        id: `${row.sku_id}-${row.channel}-${row.region}-${idx}`,
        sku: row.sku_id,
        channel: row.channel,
        region: row.region,
        basePrice: row.price_base,
        recommendedPrice: row.final_price,
        delta: row.change_pct / 100,
        confidence: row.approval === 'Auto-approve' ? 95 : row.approval === 'Manager Review' ? 75 : 50,
        projectedRoi: 12.5,
        rationale: row.approval,
        elasticity: -1.2,
        baselineVolume: 1000,
        baselineMarginPct: row.margin_pct / 100,
      }));
      setAnalysisRows(mappedRows);
    } catch (err) {
      console.error("Failed to re-run analysis:", err);
    } finally {
      setLoading(false);
    }
  };

  // Agent Pricing Logic moved to App
  const handleGeneratePrice = async () => {
    if (!selectedSku || !selectedChannel || !selectedRegion) {
      alert('Please select SKU, Channel, and Region');
      return;
    }

    setPricingLoading(true);
    setPricingData(null);
    setWhatIfData(null);

    // Reset execution steps
    const steps = executionSteps.map(s => ({ ...s, status: 'pending', result: 'Awaiting run' }));
    setExecutionSteps(steps);

    try {
      // Simulate agent execution steps with delays
      for (let i = 0; i < steps.length; i++) {
        steps[i].status = 'running';
        steps[i].result = 'Processing...';
        setExecutionSteps([...steps]);
        await new Promise(resolve => setTimeout(resolve, 600));
      }

      const data = await fetchSingleSkuPricing(selectedSku, selectedChannel, selectedRegion);

      if (data.agent_outputs) {
        steps[0].status = 'completed';
        steps[0].result = `Base: ₹${data.agent_outputs.base?.price?.toFixed(2)}`;
        steps[1].status = 'completed';
        steps[1].result = `Promo: ₹${data.agent_outputs.promo?.price?.toFixed(2)}`;
        steps[2].status = 'completed';
        steps[2].result = `Comp: ₹${data.agent_outputs.comp?.price?.toFixed(2)}`;
        steps[3].status = 'completed';
        steps[3].result = `Final: ₹${data.agent_outputs.inventory?.price?.toFixed(2)}`;
        setExecutionSteps([...steps]);
      }

      setPricingData(data);
    } catch (error) {
      console.error('Error fetching pricing:', error);
      alert('Failed to generate pricing. Please check if backend is running and try again.');
      const failedSteps = steps.map(s => ({ ...s, status: 'pending', result: 'Failed - please retry' }));
      setExecutionSteps(failedSteps);
    } finally {
      setPricingLoading(false);
    }
  };

  const handleRunWhatIf = async (overrides) => {
    setPricingLoading(true);
    try {
      const payload = {
        sku_id: selectedSku,
        channel: selectedChannel,
        region: selectedRegion,
        ...overrides
      };
      const data = await runWhatIfAnalysis(payload);
      setWhatIfData(data);

      const mappedData = {
        sku_id: selectedSku,
        channel: selectedChannel,
        region: selectedRegion,
        agent_outputs: {
          base: { price: data.what_if.candidates.price_base, reason: data.what_if.reasons.base, candidates: { neutral: data.what_if.candidates.price_base } },
          promo: { price: data.what_if.candidates.price_promo, reason: JSON.stringify(data.what_if.reasons.promo), candidates: { neutral: data.what_if.candidates.price_promo } },
          comp: { price: data.what_if.candidates.price_comp, reason: JSON.stringify(data.what_if.reasons.comp), candidates: { neutral: data.what_if.candidates.price_comp } },
          inventory: { price: data.what_if.candidates.price_inventory, reason: JSON.stringify(data.what_if.reasons.inv), candidates: { neutral: data.what_if.candidates.price_inventory } }
        },
        candidates: data.what_if.candidates,
        selection: data.what_if.selection,
        elasticity: data.what_if.elasticity,
        isWhatIf: true,
        delta: data.delta
      };
      setPricingData(mappedData);

      const steps = executionSteps.map(s => ({ ...s, status: 'completed', result: 'Updated (What-If)' }));
      setExecutionSteps(steps);

    } catch (error) {
      console.error('What-If failed:', error);
      alert('Failed to run What-If analysis');
    } finally {
      setPricingLoading(false);
    }
  };

  const getContent = () => {
    let mainContent = null;
    let rightSidebarContent = null;

    switch (activeTab) {
      case 'Insights':
        mainContent = <InsightsContent data={analysisRows} />;
        rightSidebarContent = <InsightsCharts />;
        break;
      case 'Price':
        mainContent = (
          <div className="p-4 max-w-[1200px] mx-auto space-y-4">
            <AgentPricingView
              initialSku={selectedSku}
              initialChannel={selectedChannel}
              initialRegion={selectedRegion}
              onFilterChange={handleFilterChange}
              pricingData={pricingData}
              loading={pricingLoading}
              executionSteps={executionSteps}
              onGeneratePrice={handleGeneratePrice}
              onRunWhatIf={handleRunWhatIf}
            />
            <WhatIfAnalysis
              initialSku={selectedSku}
              initialChannel={selectedChannel}
              initialRegion={selectedRegion}
            />
          </div>
        );
        rightSidebarContent = (
          <div className="space-y-6">
            <AgentPricingCharts pricingData={pricingData} />
            <LLMExplanationPanel
              selectedRow={selectedRow}
              analysisRows={filteredRows}
              filters={{ sku: selectedSku, channel: selectedChannel, region: selectedRegion }}
            />
          </div>
        );
        break;
      case 'Review':
      case 'Overview':
        mainContent = (
          <div className="p-6 max-w-[1600px] mx-auto space-y-6">
            <div className="flex items-center justify-between px-2">
              <h2 className="text-2xl font-bold text-slate-800">Market Overview</h2>
              <div className="text-sm text-slate-500">
                Showing {filteredRows.length} items
              </div>
            </div>
            <div className="rounded-2xl bg-transparent p-1 overflow-hidden">
              {loading ? (
                <div className="flex h-64 items-center justify-center">
                  <div className="text-slate-400">Loading market data...</div>
                </div>
              ) : (
                <Dashboard
                  analysisRows={filteredRows}
                  selectedRow={selectedRow}
                  onRowSelect={setSelectedRow}
                  onRun={handleRun}
                  theme="light"
                />
              )}
            </div>
          </div>
        );
        rightSidebarContent = (
          <div className="space-y-6">
            <DashboardCharts analysisRows={filteredRows} />
            <LLMExplanationPanel
              selectedRow={selectedRow}
              analysisRows={filteredRows}
              filters={{ sku: selectedSku, channel: selectedChannel, region: selectedRegion }}
            />
            <ForecastingWidget
              sku={selectedRow ? selectedRow.sku : selectedSku}
              channel={selectedRow ? selectedRow.channel : selectedChannel}
              region={selectedRow ? selectedRow.region : selectedRegion}
            />
          </div>
        );
        break;
      default:
        mainContent = <InsightsContent data={analysisRows} />;
        rightSidebarContent = <InsightsCharts />;
    }

    return { mainContent, rightSidebarContent };
  };

  const { mainContent, rightSidebarContent } = getContent();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-brand-light/5 to-brand-DEFAULT/5 flex font-sans text-slate-900 selection:bg-brand-light/30 selection:text-brand-dark">
      <Sidebar
        selectedSku={selectedSku}
        selectedChannel={selectedChannel}
        selectedRegion={selectedRegion}
        onFilterChange={handleFilterChange}
        executionSteps={executionSteps}
        onRun={() => {
          if (activeTab === 'Price') {
            handleGeneratePrice();
          } else if (selectedSku && selectedChannel && selectedRegion) {
            setActiveTab('Price');
            // We need a slight delay or effect to trigger run after tab switch, 
            // but since handleGeneratePrice uses state, we can just call it.
            // However, the Price view might not be mounted yet. 
            // For simplicity, let's just switch tab and let user run, 
            // OR call handleGeneratePrice immediately. 
            // handleGeneratePrice depends on 'selectedSku' state which is already set.
            setTimeout(() => handleGeneratePrice(), 100);
          } else {
            handleRun();
          }
        }}
        onRunBatch={handleRun}
      />
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <Header />
        <DashboardLayout
          activeTab={activeTab}
          onTabChange={setActiveTab}
          rightSidebar={rightSidebarContent}
        >
          {mainContent}
        </DashboardLayout>
      </div>

      {showBot && (
        <SimulationBot
          analysisRows={analysisRows}
          onClose={() => setShowBot(false)}
        />
      )}
    </div>
  );
}

export default App;
