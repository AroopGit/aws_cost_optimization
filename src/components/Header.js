import React from 'react';
import { Download, Bell, HelpCircle, User, MessageSquare } from 'lucide-react';

const Header = () => {
    return (
        <header className="bg-white/80 backdrop-blur-xl border-b border-brand-light/20 px-6 py-3 flex items-center justify-between sticky top-0 z-40 shadow-sm supports-[backdrop-filter]:bg-white/60">
            <div className="flex items-center gap-4">
                {/* Logo removed and moved to sidebar */}
            </div>

            <div className="flex items-center gap-3">
                <button className="flex items-center gap-2 px-4 py-2 bg-brand-DEFAULT/10 hover:bg-brand-DEFAULT/20 text-brand-dark rounded-lg text-sm font-medium transition-all shadow-sm border border-brand-DEFAULT/20">
                    <Download size={16} />
                    Download CSV
                </button>

                <div className="h-6 w-px bg-slate-200 mx-2"></div>

                <button className="p-2 text-slate-500 hover:text-brand-DEFAULT transition-colors relative">
                    <HelpCircle size={20} />
                </button>
                <button className="p-2 text-slate-500 hover:text-brand-DEFAULT transition-colors relative">
                    <MessageSquare size={20} />
                </button>
                <button className="p-2 text-slate-500 hover:text-brand-DEFAULT transition-colors relative">
                    <Bell size={20} />
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
                </button>

                <div className="ml-2 w-8 h-8 bg-gradient-to-br from-brand-DEFAULT to-brand-dark rounded-full flex items-center justify-center text-white font-bold text-xs shadow-md cursor-pointer ring-2 ring-brand-light/20 hover:ring-brand-light/40 transition-all">
                    RS
                </div>
            </div>
        </header>
    );
};

export default Header;
