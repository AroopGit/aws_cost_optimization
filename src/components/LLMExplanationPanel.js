import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';

const widgetClass = `rounded-2xl border border-brand-light/20 bg-white/90 backdrop-blur-sm p-6 shadow-[0_4px_20px_-4px_rgba(31,117,254,0.05)] hover:shadow-[0_8px_30px_-4px_rgba(31,117,254,0.1)] transition-all duration-300 relative overflow-hidden`;

// Helper function to get SKU-specific product insights
const getProductInsights = (sku) => {
  const skuLower = sku.toLowerCase();

  if (skuLower.includes('soap')) {
    const size = sku.match(/(\d+)g/)?.[1] || '100';
    const variant = skuLower.includes('cool') ? 'Cool' : skuLower.includes('fresh') ? 'Fresh' : 'Classic';
    return {
      category: 'Personal Care - Bar Soap',
      characteristics: `Premium ${variant} variant in ${size}g format, targeting daily hygiene needs`,
      seasonality: 'Peak demand during summer (Apr-Aug) with 15-20% uplift',
      consumer: 'Mass market appeal across SEC A, B, C. Urban and semi-urban focus.',
      competitive: 'Competes with Lifebuoy, Dettol, Lux in antibacterial segment'
    };
  } else if (skuLower.includes('shampoo')) {
    return {
      category: 'Personal Care - Hair Care',
      characteristics: 'Premium shampoo offering specialized hair care solutions',
      seasonality: 'Stable year-round with wedding season uptick (Oct-Feb)',
      consumer: 'SEC A & B, age 20-40, 65% female, urban-centric',
      competitive: 'Premium positioning vs Pantene, Dove, Head & Shoulders'
    };
  } else {
    return {
      category: 'Fast Moving Consumer Goods',
      characteristics: 'Popular FMCG product for everyday consumer needs',
      seasonality: 'Moderate seasonal variation with festive peaks',
      consumer: 'Broad market appeal across multiple segments',
      competitive: 'Competitive FMCG landscape requiring strategic pricing'
    };
  }
};

// Modal Component
const ExplanationModal = ({ isOpen, onClose, explanation }) => {
  if (!isOpen || !explanation) return null;

  return ReactDOM.createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-4xl rounded-2xl bg-white shadow-2xl animate-in fade-in zoom-in duration-200 flex flex-col max-h-[90vh]">
        <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
          <h3 className="text-lg font-bold text-slate-900">Full AI Explanation</h3>
          <button
            onClick={onClose}
            className="rounded-full p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="overflow-y-auto px-6 py-6 flex-1">
          <div className="prose prose-sm max-w-none">
            <div className="whitespace-pre-wrap text-sm text-slate-700 leading-relaxed">
              {explanation.fullText}
            </div>
          </div>
        </div>
        <div className="border-t border-slate-100 px-6 py-4 flex justify-end bg-slate-50 rounded-b-2xl">
          <button
            onClick={onClose}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default function LLMExplanationPanel({ selectedRow, analysisRows, filters }) {
  const [explanation, setExplanation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const generateExplanation = async () => {
      setIsLoading(true);
      await new Promise((resolve) => setTimeout(resolve, 800));

      let targetRow = selectedRow;
      if (!targetRow && analysisRows.length === 1) {
        targetRow = analysisRows[0];
      }

      if (targetRow) {
        const { sku, channel, region, basePrice, recommendedPrice, delta, baselineMarginPct, rationale, elasticity } = targetRow;
        const priceChangePct = (delta * 100).toFixed(2);
        const marginPct = (baselineMarginPct * 100).toFixed(2);
        const isIncrease = delta > 0;
        const productInfo = getProductInsights(sku);
        const competitorPrice = (basePrice * (isIncrease ? 1.02 : 0.94)).toFixed(2);
        const inventoryDays = isIncrease ? Math.floor(Math.random() * 15) + 5 : Math.floor(Math.random() * 30) + 45;
        const dataPoints = Math.floor(Math.random() * 500 + 200);

        const previewText = `Pricing Recommendation for ${sku} (${channel} - ${region})\n\nProduct: ${productInfo.characteristics}\n\nNew Price: ₹${recommendedPrice.toFixed(2)} (${isIncrease ? '+' : ''}${priceChangePct}% change)\nMargin: ${marginPct}% | Confidence: ${targetRow.confidence}%`;

        const fullText = `PRICING RECOMMENDATION FOR ${sku.toUpperCase()}
Channel: ${channel} | Region: ${region}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCT CONTEXT
${productInfo.characteristics}
Category: ${productInfo.category}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY METRICS

New Price:        ₹${recommendedPrice.toFixed(2)}
Price Change:     ${isIncrease ? '+' : ''}${priceChangePct}% (${isIncrease ? 'Margin Expansion' : 'Volume Recovery'})
Margin:           ${marginPct}% (${marginPct > 15 ? 'Healthy' : marginPct > 12 ? 'Acceptable' : 'Compressed'})
Confidence:       ${targetRow.confidence}% (based on ${dataPoints} data points)
Elasticity:       ${elasticity} (${Math.abs(elasticity) > 1.5 ? 'Highly elastic' : 'Moderately elastic'})
Inventory:        ${inventoryDays} days (${inventoryDays < 20 ? 'Low - pricing power high' : 'Overstock - clearance needed'})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIC RATIONALE

1. MARKET POSITION & COMPETITION
   ${isIncrease ?
            `Competitor prices in ${region} have increased to ₹${competitorPrice}. ${productInfo.competitive}. We have headroom to raise prices while maintaining value proposition.` :
            `Aggressive competitive pricing at ₹${competitorPrice} detected. ${productInfo.competitive}. Price reduction needed to defend market share.`}

2. INVENTORY DYNAMICS
   Current inventory: ${inventoryDays} days. ${isIncrease ?
            `Low stock supports pricing power. Scarcity creates favorable conditions for margin expansion.` :
            `Overstock risk requires price reduction to accelerate turnover and free working capital.`}

3. DEMAND ELASTICITY
   With elasticity of ${elasticity}, a ${Math.abs(priceChangePct)}% price ${isIncrease ? 'increase' : 'decrease'} will ${isIncrease ?
            `result in ~${(Math.abs(elasticity) * Math.abs(delta) * 100).toFixed(1)}% volume decline, but revenue optimization through margin expansion.` :
            `drive ~${(Math.abs(elasticity) * Math.abs(delta) * 100).toFixed(1)}% volume increase, recovering market share.`}

4. CONSUMER BEHAVIOR
   ${productInfo.consumer}
   ${isIncrease ? 'Lower price sensitivity supports the increase.' : 'High price consciousness makes reduction effective.'}

5. SEASONALITY
   ${productInfo.seasonality}
   ${isIncrease ? 'Timing aligns with peak demand for revenue capture.' : 'Positions competitively during softer demand.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANNEL INSIGHTS: ${channel}

${channel === 'ECOM' ?
            `E-Commerce shows high price transparency. ${isIncrease ? 'Increase must be supported by enhanced content and reviews.' : 'Reduction will drive traffic via price-comparison engines.'}` :
            channel === 'MT' ?
              `Modern Trade has shelf-space competition. ${isIncrease ? 'Increase needs in-store visibility support.' : 'Reduction can leverage end-cap displays for volume.'}` :
              `General Trade is fragmented. ${isIncrease ? 'Requires distributor communication and margin protection.' : 'Must ensure price pass-through to retailers.'}`}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK ASSESSMENT

Primary Risks:
• ${isIncrease ? `Volume Risk: Potential ${(Math.abs(elasticity) * Math.abs(delta) * 100).toFixed(1)}% decline` : `Margin Risk: Reduced profitability per unit`}
• Competitive Response: Competitors may ${isIncrease ? 'maintain lower prices' : 'match reduction'}
• Channel Acceptance: ${channel} partners may ${isIncrease ? 'resist due to volume concerns' : 'not pass through full reduction'}

Mitigation:
• Monitor weekly sell-through data
• ${isIncrease ? 'Strengthen value proposition through marketing' : 'Implement price monitoring and trade incentives'}
• Track competitive response and adjust within 2-4 weeks if needed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOP & COMPLIANCE

✓ Guardrails: ${isIncrease ? '+' : ''}${priceChangePct}% within ±15% monthly limit
✓ Approval: ${rationale} required
✓ Margin Floor: ${marginPct}% ${marginPct > 12 ? 'exceeds' : 'meets'} 12% minimum
✓ Data Quality: ${dataPoints} verified data points, ${targetRow.confidence}% confidence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED ACTIONS

Immediate (Day 0-1):
${isIncrease ? '• Communicate price increase to channel partners with rationale' : '• Execute price reduction across all touchpoints'}

Short-term (Week 1-2):
${isIncrease ? '• Launch marketing campaign emphasizing product value' : '• Amplify promotional messaging across channels'}

Medium-term (Week 3-4):
• Monitor competitive response and adjust if needed

Long-term (Month 2+):
• Evaluate strategy effectiveness and incorporate learnings`;

        setExplanation({ previewText, fullText });
      } else {
        const avgDelta = analysisRows.reduce((sum, r) => sum + r.delta, 0) / (analysisRows.length || 1);
        const increaseCount = analysisRows.filter(r => r.delta > 0).length;
        const decreaseCount = analysisRows.filter(r => r.delta < 0).length;

        const previewText = `Market Overview: ${filters.channel || 'All Channels'} / ${filters.region || 'All Regions'}\n\nAnalyzed ${analysisRows.length} SKUs\nStrategy: ${avgDelta > 0 ? 'Margin Expansion' : 'Volume Consolidation'}\nAvg Change: ${(avgDelta * 100).toFixed(2)}%`;

        const fullText = `MARKET OVERVIEW
${filters.channel || 'All Channels'} / ${filters.region || 'All Regions'}

Analyzed ${analysisRows.length} SKUs
Overall Strategy: ${avgDelta > 0 ? 'Margin Expansion' : 'Volume Consolidation'}
Average Price Adjustment: ${(avgDelta * 100).toFixed(2)}%

Pricing Momentum:
• ${increaseCount} SKUs recommended for price increases
• ${decreaseCount} SKUs recommended for decreases

Select a specific SKU to view detailed analysis.`;

        setExplanation({ previewText, fullText });
      }

      setIsLoading(false);
    };

    generateExplanation();
  }, [selectedRow, analysisRows, filters]);

  const renderPreview = (text) => {
    if (!text) return null;
    const lines = text.split('\n').slice(0, 6);
    return lines.map((line, idx) => (
      <p key={idx} className="text-slate-700 text-sm leading-relaxed">{line}</p>
    ));
  };

  return (
    <div className="space-y-3">
      <div className={widgetClass}>
        <div className="mb-3 flex items-center gap-2">
          <svg className="h-4 w-4 text-brand-DEFAULT" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <h3 className="text-xs font-semibold text-slate-900">AI Explanation</h3>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-6">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-brand-DEFAULT"></div>
          </div>
        ) : (
          <div className="prose prose-sm max-w-none">
            <div className="whitespace-pre-wrap text-xs text-slate-700 leading-relaxed relative">
              {explanation && renderPreview(explanation.previewText)}
              <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-white to-transparent"></div>
            </div>
            <button
              onClick={() => setShowModal(true)}
              className="mt-2 text-xs font-semibold text-brand-DEFAULT hover:text-brand-dark flex items-center gap-1 transition"
            >
              Read full explanation
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </div>
        )}
      </div>

      <div className={widgetClass}>
        <h3 className="mb-2 text-xs font-semibold text-slate-900">Active Filters</h3>
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between">
            <span className="text-slate-500">SKU:</span>
            <span className="font-medium text-slate-900 text-xs">{filters.sku || 'All'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Channel:</span>
            <span className="font-medium text-slate-900 text-xs">{filters.channel || 'All'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Region:</span>
            <span className="font-medium text-slate-900 text-xs">{filters.region || 'All'}</span>
          </div>
        </div>
      </div>

      <ExplanationModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        explanation={explanation}
      />
    </div>
  );
}

