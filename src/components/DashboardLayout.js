import React from 'react';
import { MoreHorizontal } from 'lucide-react';

const DashboardLayout = ({ activeTab, onTabChange, children, rightSidebar }) => {
    return (
        <div className="flex-1 bg-transparent flex flex-col h-full overflow-hidden">
            {/* Secondary Navigation */}
            <div className="bg-white/60 backdrop-blur-md border-b border-brand-light/20 px-8 flex-shrink-0">
                <div className="flex gap-8">
                    {['Overview', 'Price', 'Review', 'Insights'].map(tab => (
                        <button
                            key={tab}
                            onClick={() => onTabChange(tab)}
                            className={`py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === tab
                                ? 'border-brand-DEFAULT text-brand-DEFAULT'
                                : 'border-transparent text-slate-500 hover:text-slate-700'
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </div>

            {/* Sub-header / Status Bar */}
            <div className="bg-white/60 backdrop-blur-md border-b border-brand-light/20 px-8 py-3 flex items-center justify-between shadow-sm z-10 relative flex-shrink-0">
                <div className="flex items-center gap-2 text-sm">
                    <span className="font-bold text-slate-800">Enterprise</span>
                    <span className="text-slate-400">/</span>
                    <span className="font-bold text-slate-800">Plan</span>
                    <span className="bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wide">Open</span>
                </div>
                <div className="flex items-center gap-4">
                    <span className="text-xs text-slate-500">Last Plan Generated: 11/3/2024 5:30 AM PST</span>
                    <button className="p-1.5 hover:bg-slate-100 rounded text-slate-400">
                        <MoreHorizontal size={16} />
                    </button>
                    <button className="bg-amber-500 hover:bg-amber-600 text-white px-4 py-1.5 rounded-lg text-sm font-bold shadow-sm transition-colors">
                        Publish
                    </button>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex flex-1 overflow-hidden">
                <div className="flex-1 overflow-y-auto">
                    {children}
                </div>
                {rightSidebar && (
                    <aside className="w-[400px] flex-shrink-0 border-l border-brand-light/20 bg-white/60 backdrop-blur-xl overflow-y-auto p-4 z-30 shadow-[-4px_0_20px_rgba(0,0,0,0.02)]">
                        {rightSidebar}
                    </aside>
                )}
            </div>
        </div>
    );
};

export default DashboardLayout;
