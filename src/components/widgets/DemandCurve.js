import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceDot
} from 'recharts';

const DemandCurve = ({ currentPrice, elasticity, baseDemand = 1000 }) => {
    // Generate curve data points
    // Q = Q0 * (P/P0)^elasticity
    const data = [];
    const range = 0.3; // +/- 30% price range
    const minPrice = currentPrice * (1 - range);
    const maxPrice = currentPrice * (1 + range);
    const steps = 20;

    for (let i = 0; i <= steps; i++) {
        const price = minPrice + (i * (maxPrice - minPrice)) / steps;
        // Simple constant elasticity model
        const demand = baseDemand * Math.pow(price / currentPrice, elasticity);
        data.push({
            price: price,
            demand: Math.round(demand),
        });
    }

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-lg">
                    <p className="text-sm font-semibold text-slate-900">Price: ₹{payload[0].payload.price.toFixed(2)}</p>
                    <p className="text-sm text-blue-600">Demand: {payload[0].value} units</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                        dataKey="price"
                        type="number"
                        domain={['auto', 'auto']}
                        tickFormatter={(val) => `₹${val.toFixed(0)}`}
                        label={{ value: 'Price (₹)', position: 'insideBottom', offset: -5, fontSize: 12 }}
                        height={30}
                    />
                    <YAxis
                        dataKey="demand"
                        label={{ value: 'Demand (Units)', angle: -90, position: 'insideLeft', fontSize: 12 }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line
                        type="monotone"
                        dataKey="demand"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        dot={false}
                    />
                    {/* Mark current price */}
                    <ReferenceDot x={currentPrice} y={baseDemand} r={6} fill="#10b981" stroke="white" strokeWidth={2} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default DemandCurve;
