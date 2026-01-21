const API_BASE_URL = 'http://localhost:8000';

export const fetchSkus = async () => {
    const response = await fetch(`${API_BASE_URL}/dataset/skus`);
    if (!response.ok) throw new Error('Failed to fetch SKUs');
    return response.json();
};

export const fetchChannels = async () => {
    const response = await fetch(`${API_BASE_URL}/dataset/channels`);
    if (!response.ok) throw new Error('Failed to fetch Channels');
    return response.json();
};

export const fetchRegions = async () => {
    const response = await fetch(`${API_BASE_URL}/dataset/regions`);
    if (!response.ok) throw new Error('Failed to fetch Regions');
    return response.json();
};

export const fetchBatchPricing = async (limit = 100, skuId = '', channel = '', region = '') => {
    const params = new URLSearchParams({ limit });
    if (skuId) params.append('sku_id', skuId);
    if (channel) params.append('channel', channel);
    if (region) params.append('region', region);

    const response = await fetch(`${API_BASE_URL}/pricing/batch?${params.toString()}`);
    if (!response.ok) throw new Error('Failed to fetch batch pricing');
    return response.json();
};

export const fetchSingleSkuPricing = async (skuId, channel, region) => {
    const response = await fetch(`${API_BASE_URL}/pricing/single-sku`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sku_id: skuId, channel, region }),
    });
    if (!response.ok) throw new Error('Failed to fetch single SKU pricing');
    return response.json();
};

export const runWhatIfAnalysis = async (payload) => {
    const response = await fetch(`${API_BASE_URL}/pricing/what-if-enhanced`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Failed to run What-If analysis');
    return response.json();
};

export const fetchForecastPricing = async (payload) => {
    const response = await fetch(`${API_BASE_URL}/pricing/forecast-pricing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Failed to run Forecast');
    return response.json();
};
