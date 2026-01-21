export const formatCurrency = (value) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2,
  }).format(value);

export const formatPercent = (value) =>
  `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;

const generateRationale = () => {
  const factors = [
    { label: 'Promo ROI lift', weight: 0.4 },
    { label: 'Competitive position', weight: 0.35 },
    { label: 'Elasticity trend', weight: 0.25 },
  ];
  return factors
    .map((f) => `${f.label} • ${(f.weight * 100).toFixed(0)}% weight`)
    .join(' · ');
};

export const parseCsv = (text) => {
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length <= 1) return [];
  const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
  const skuIndex = headers.indexOf('sku');
  const basePriceIndex =
    headers.indexOf('base_price') !== -1
      ? headers.indexOf('base_price')
      : headers.indexOf('baseprice');
  const volumeIndex = headers.indexOf('baseline_volume');

  return lines.slice(1).map((line, idx) => {
    const cells = line.split(',').map((c) => c.trim());
    const sku = cells[skuIndex] ?? `SKU-${idx + 1}`;
    const basePrice = parseFloat(cells[basePriceIndex]) || 10 + idx * 2;
    const baselineVolume = parseFloat(cells[volumeIndex]) || 1000 + idx * 125;
    return { sku, basePrice, baselineVolume };
  });
};

export const runMockAnalysis = (rows) =>
  rows.map((row, idx) => {
    const elasticity = 0.8 + Math.random() * 0.6;
    const lift = (Math.random() - 0.2) * 0.2;
    const recommendedPrice = +(row.basePrice * (1 + lift)).toFixed(2);
    const delta = (recommendedPrice - row.basePrice) / row.basePrice;
    const confidence = Math.round(70 + Math.random() * 25);
    const projectedRoi = +(5 + Math.random() * 10).toFixed(1);
    const baselineMarginPct = 0.22 + Math.random() * 0.1;

    return {
      id: `${row.sku}-${idx}`,
      sku: row.sku,
      basePrice: row.basePrice,
      recommendedPrice,
      delta,
      confidence,
      projectedRoi,
      rationale: generateRationale(),
      elasticity,
      baselineVolume: row.baselineVolume,
      baselineMarginPct,
    };
  });

