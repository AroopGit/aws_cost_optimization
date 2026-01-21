import React from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';

const SeasonalityTimeline = ({ seasonalityIndex = 1.0 }) => {
    // Simulate a seasonality curve peaking at the current index if > 1, or dipping if < 1
    // This is a placeholder visualization since we don't have the full 12-month curve from backend yet
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const currentMonth = new Date().getMonth();

    const data = months.map((month, index) => {
        // Create a synthetic curve that aligns with the provided seasonality index for the current month
        // Simple sine wave shifted to peak/trough at current month
        const dist = Math.abs(index - currentMonth);
        const factor = seasonalityIndex > 1.0 ? 1 : -1;
        const mag = Math.abs(seasonalityIndex - 1.0);

        // Decay effect away from current month
        const val = 1.0 + (mag * Math.cos(dist * 0.5)) * factor;

        return {
            month,
            index: Math.max(0.5, val) // Ensure reasonable bounds
        };
    });

    return (
        <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorIndex" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0.5, 1.5]} hide />
                    <Tooltip
                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        formatter={(value) => [value.toFixed(2), 'Index']}
                    />
                    <Area
                        type="monotone"
                        dataKey="index"
                        stroke="#3b82f6"
                        fillOpacity={1}
                        fill="url(#colorIndex)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SeasonalityTimeline;
