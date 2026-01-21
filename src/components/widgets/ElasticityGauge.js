import React from 'react';

const ElasticityGauge = ({ elasticity }) => {
    // Elasticity usually ranges from -0.5 (inelastic) to -3.0 (elastic)
    // We'll normalize this to a 0-100 scale for the gauge
    // 0 = -0.5 (Inelastic), 50 = -1.5 (Unit), 100 = -3.0 (Elastic)

    const value = Math.abs(elasticity);
    let label = 'Unit Elastic';
    let color = 'bg-yellow-500';
    let width = '50%';

    if (value < 1.0) {
        label = 'Inelastic';
        color = 'bg-blue-300';
        width = `${Math.min((value / 1.0) * 33, 33)}%`;
    } else if (value > 2.0) {
        label = 'Highly Elastic';
        color = 'bg-blue-600';
        width = `${Math.min(66 + ((value - 2.0) / 2.0) * 33, 100)}%`;
    } else {
        label = 'Elastic';
        color = 'bg-blue-400';
        width = `${Math.min(33 + ((value - 1.0) / 1.0) * 33, 66)}%`;
    }

    return (
        <div className="flex flex-col items-center justify-center p-4">
            <div className="relative h-32 w-64 overflow-hidden">
                <div className="absolute left-0 top-0 h-32 w-64 rounded-t-full bg-slate-100"></div>
                <div
                    className={`absolute left-0 top-0 h-32 w-64 origin-bottom rounded-t-full transition-all duration-1000 ease-out ${color} opacity-20`}
                    style={{
                        transform: `rotate(${(Math.min(Math.max(value, 0.5), 3.5) - 0.5) / 3 * 180 - 180}deg)`,
                        width: '100%' // The rotation handles the "fill" effect for this specific gauge design
                    }}
                ></div>
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
                    <p className="text-3xl font-bold text-slate-800">{elasticity.toFixed(2)}</p>
                    <p className="text-sm font-medium text-slate-500">{label}</p>
                </div>
            </div>
            <div className="mt-2 flex w-full justify-between px-4 text-xs text-slate-400">
                <span>Inelastic (0.5)</span>
                <span>Elastic (3.0+)</span>
            </div>
        </div>
    );
};

export default ElasticityGauge;
