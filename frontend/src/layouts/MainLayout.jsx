import { useState, useEffect } from 'react';
import {
    LayoutDashboard, Settings, CheckCircle, BarChart2, LineChart, Menu, X
} from 'lucide-react';
import { cn } from "../lib/utils";
import { motion, AnimatePresence } from "framer-motion";

export const MainLayout = ({ children, activeTab, setActiveTab }) => {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth < 1024);
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    const navItems = [
        { id: 'setup', label: 'Genome Setup', icon: Settings },
        { id: 'check', label: 'Consistency Check', icon: CheckCircle },
        { id: 'bench', label: 'Market Benchmarking', icon: BarChart2 },
        { id: 'analytics', label: 'Data Analytics', icon: LineChart },
    ];

    const handleNavClick = (id) => {
        setActiveTab(id);
        if (isMobile) setMobileMenuOpen(false);
    };

    const SidebarContent = () => (
        <>
            <div className="p-6 lg:p-8 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-3 group cursor-pointer">
                    <div className="p-2 bg-indigo-500/10 border border-indigo-500/20 rounded-xl group-hover:bg-indigo-500/20 transition-all duration-500 group-hover:rotate-12 group-hover:shadow-[0_0_20px_rgba(99,102,241,0.2)]">
                        <LayoutDashboard size={24} className="text-indigo-400 stroke-[2]" />
                    </div>
                    <h1 className="font-black text-2xl tracking-tighter text-white">
                        Genome<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">AI</span>
                    </h1>
                </div>
                {isMobile && (
                    <button onClick={() => setMobileMenuOpen(false)} className="p-2 text-gray-400 hover:text-white">
                        <X size={24} />
                    </button>
                )}
            </div>

            <nav className="p-4 flex-1 space-y-2 mt-4 overflow-y-auto">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                        <button
                            key={item.id}
                            onClick={() => handleNavClick(item.id)}
                            className={cn(
                                "w-full flex items-center gap-4 px-4 py-3.5 rounded-xl text-sm font-semibold transition-all duration-300 group relative overflow-hidden",
                                isActive
                                    ? "bg-white/10 text-white shadow-lg border border-white/5"
                                    : "text-gray-400 hover:text-white hover:bg-white/5 border border-transparent"
                            )}
                        >
                            {isActive && (
                                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-indigo-500 rounded-r-full shadow-[0_0_10px_rgba(99,102,241,0.8)]" />
                            )}
                            <Icon
                                size={20}
                                className={cn(
                                    "transition-colors duration-300",
                                    isActive ? "text-indigo-400 stroke-[2.5]" : "text-gray-500 group-hover:text-gray-300 stroke-[2]"
                                )}
                            />
                            <span className="tracking-wide relative z-10">{item.label}</span>
                        </button>
                    );
                })}
            </nav>

            <div className="p-6 border-t border-white/5">
                <div className="flex items-center gap-4 p-4 bg-[#111116] rounded-2xl border border-white/5 hover:border-white/10 transition-colors cursor-pointer group">
                    <div className="w-10 h-10 shrink-0 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white font-black text-lg shadow-inner group-hover:shadow-[0_0_15px_rgba(99,102,241,0.4)] transition-shadow">
                        S
                    </div>
                    <div className="text-sm flex-1 truncate">
                        <p className="font-bold text-gray-200">Shark Admin</p>
                        <p className="text-indigo-400 font-bold text-[10px] tracking-widest uppercase mt-1">Enterprise Plan</p>
                    </div>
                </div>
            </div>
        </>
    );

    return (
        <div className="min-h-screen bg-[#050505] flex font-sans text-gray-200 selection:bg-indigo-500/30 selection:text-indigo-200 overflow-hidden relative">

            {/* Background Orbs */}
            <div className="fixed top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-gradient-to-br from-indigo-500/10 to-transparent blur-[120px] pointer-events-none -z-10" />
            <div className="fixed bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-gradient-to-br from-violet-500/10 to-transparent blur-[120px] pointer-events-none -z-10" />

            {/* Mobile Topbar */}
            {isMobile && (
                <header className="absolute top-0 inset-x-0 h-20 bg-[#0A0A0C]/90 backdrop-blur-xl border-b border-white/5 flex items-center justify-between px-6 z-30">
                    <div className="flex items-center gap-3">
                        <div className="p-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                            <LayoutDashboard size={20} className="text-indigo-400 stroke-[2.5]" />
                        </div>
                        <h1 className="font-black text-xl tracking-tighter text-white">
                            Genome<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">AI</span>
                        </h1>
                    </div>
                    <button onClick={() => setMobileMenuOpen(true)} className="p-2 text-gray-400 hover:text-white bg-white/5 rounded-lg border border-white/10">
                        <Menu size={24} />
                    </button>
                </header>
            )}

            {/* Desktop Sidebar */}
            {!isMobile && (
                <aside className="w-[280px] shrink-0 bg-[#0A0A0C]/80 backdrop-blur-2xl border-r border-white/5 flex flex-col z-20 shadow-2xl">
                    <SidebarContent />
                </aside>
            )}

            {/* Mobile Sidebar Overlay */}
            <AnimatePresence>
                {isMobile && mobileMenuOpen && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            onClick={() => setMobileMenuOpen(false)}
                            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
                        />
                        <motion.aside
                            initial={{ x: "-100%" }} animate={{ x: 0 }} exit={{ x: "-100%" }}
                            transition={{ type: "spring", damping: 25, stiffness: 200 }}
                            className="fixed inset-y-0 left-0 w-[280px] bg-[#0A0A0C]/95 backdrop-blur-2xl border-r border-white/10 flex flex-col z-50 shadow-2xl"
                        >
                            <SidebarContent />
                        </motion.aside>
                    </>
                )}
            </AnimatePresence>

            {/* Main Container */}
            <main className={cn(
                "flex-1 overflow-y-auto relative z-10 scroll-smooth custom-scrollbar",
                isMobile ? "pt-20" : ""
            )}>
                <div className="max-w-7xl mx-auto p-6 lg:p-12 pb-24">
                    {children}
                </div>
            </main>

        </div>
    );
};
