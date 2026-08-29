import { useState, useEffect } from 'react';
import { BarChart2, Loader2, ArrowRight, Target, ShieldCheck } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { NeuralNetworkSVG } from "../components/ui/NeuralNetworkSVG";
import { API_BASE } from "../lib/constants";
import {
    BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell,
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { motion } from "framer-motion";

export const Benchmarking = ({ profile }) => {
    const [brands, setBrands] = useState([]);
    const [competitor, setCompetitor] = useState("");
    const [metric, setMetric] = useState("Sentiment Distribution");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE}/brands`)
            .then(res => res.json())
            .then(data => {
                const list = data.brands || [];
                setBrands(list);
                if (list.length > 0) {
                    // Default to first competitor (not user_brand if possible)
                    const firstComp = list.find(b => b.brand_id !== 'user_brand') || list[0];
                    setCompetitor(firstComp.brand_id);
                }
            })
            .catch(err => console.error(err));
    }, []);

    const handleBenchmark = async () => {
        if (!competitor) return;
        setLoading(true);
        try {
            console.log(`Starting simulation for ${profile?.brand_id || "user_brand"} vs ${competitor}`);
            const payload = { 
                my_brand: profile?.brand_id || "user_brand", 
                competitor, 
                metric 
            };
            console.log("Benchmark payload:", payload);
            const res = await fetch(`${API_BASE}/benchmark`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                const data = await res.json();
                console.log("Benchmark response received:", data);
                setResult(data);
                console.log("State 'result' updated with benchmark data.");
            } else {
                console.error("Benchmark API failed with status:", res.status);
            }
        } catch (err) {
            console.error("Consistency analysis failed:", err);
            setResult({ error: "Network anomaly: Connection to engine failed." });
        }
        setLoading(false);
    };

    const chartData = result ? [
        { name: result.my_brand.name, value: result.my_brand.value, label: result.my_brand.label, isMine: true },
        { name: result.competitor.name, value: result.competitor.value, label: result.competitor.label, isMine: false }
    ] : [];

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both relative max-w-7xl mx-auto px-4 sm:px-6 mb-20">
            <div className="mb-12 flex items-center gap-5 relative z-10">
                <div className="p-3.5 bg-gradient-to-br from-orange-400 to-rose-500 rounded-2xl shadow-[0_0_30px_rgba(249,115,22,0.2)] text-white border border-orange-400/30">
                    <BarChart2 size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Market Benchmarking</h2>
                    <p className="text-gray-400 mt-2 text-lg">Compare your brand parameters directly against top competitors.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
                <Card className="lg:col-span-4 flex flex-col" delay={100}>
                    <h3 className="text-xl font-bold text-white mb-8 tracking-wide flex items-center gap-2">
                        <ShieldCheck size={20} className="text-orange-500" />
                        Simulation Control
                    </h3>
                    <div className="space-y-8 flex-1">
                        <div className="group">
                            <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-gray-500 mb-3 transition-colors group-focus-within:text-orange-400">Reference Competitor</label>
                            <div className="relative">
                                <select
                                    value={competitor}
                                    onChange={(e) => setCompetitor(e.target.value)}
                                    className="w-full px-4 py-4 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer font-bold text-sm"
                                >
                                    {brands.map(b => (
                                        <option key={b.brand_id} value={b.brand_id} className="bg-[#111116] text-white">
                                            {b.brand_name}
                                        </option>
                                    ))}
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                </div>
                            </div>
                        </div>

                        <div className="group">
                            <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-gray-500 mb-3 transition-colors group-focus-within:text-orange-400">Head-to-Head Metric</label>
                            <div className="relative">
                                <select
                                    value={metric}
                                    onChange={(e) => setMetric(e.target.value)}
                                    className="w-full px-4 py-4 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer font-bold text-sm"
                                >
                                    <option className="bg-[#111116] text-white">Sentiment Distribution</option>
                                    <option className="bg-[#111116] text-white">Keyword Overlap</option>
                                    <option className="bg-[#111116] text-white">Readability Level</option>
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                </div>
                            </div>
                        </div>
                    </div>

                    <Button 
                        primary 
                        className="w-full bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-400 hover:to-rose-500 shadow-[0_0_20px_rgba(249,115,22,0.3)] text-lg mt-10 py-5 group" 
                        onClick={handleBenchmark} 
                        disabled={loading || !competitor}
                    >
                        {loading ? <Loader2 className="animate-spin" /> : <>Run Simulation <ArrowRight size={18} className="ml-2 transition-transform group-hover:translate-x-1" /></>}
                    </Button>
                </Card>

                <Card className="lg:col-span-8 flex flex-col p-8 min-h-[600px] border-orange-500/20 shadow-[0_0_40px_rgba(249,115,22,0.05)] overflow-hidden" delay={200}>
                    <NeuralNetworkSVG />

                    {!result ? (
                        <div className="flex-1 flex flex-col items-center justify-center text-center relative z-10">
                            <div className="relative mb-8">
                                <div className="absolute inset-0 bg-gradient-to-tr from-orange-500/20 to-rose-500/20 blur-[40px] rounded-full animate-pulse" />
                                <Target size={72} className="text-orange-500/80 relative z-10" />
                            </div>
                            <h3 className="text-2xl font-bold text-white mb-4 tracking-tight">Awaiting Parameters</h3>
                            <p className="text-gray-500 max-w-sm text-lg leading-relaxed mx-auto">
                                Select a competitor target and configure the head-to-head metric to compute a multi-dimensional analysis.
                            </p>
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col w-full relative z-10 animate-in fade-in duration-700">
                            <div className="flex items-center justify-between mb-12 border-b border-white/5 pb-8">
                                <div>
                                    <h3 className="text-xl md:text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-rose-400 uppercase tracking-tight">{metric}</h3>
                                    <p className="text-gray-400 font-bold tracking-widest text-[10px] mt-1 uppercase">Direct Head-to-Head Comparison</p>
                                </div>
                                <div className="flex gap-6">
                                    <div className="flex items-center gap-3">
                                        <div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_15px_rgba(249,115,22,0.8)]" />
                                        <span className="text-[10px] font-black text-gray-200 uppercase tracking-widest">{result.my_brand.name}</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="w-3 h-3 rounded-full bg-gray-600" />
                                        <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest">{result.competitor.name}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 flex-1 items-center">
                                {/* Bar Chart Section */}
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ duration: 0.5 }}
                                    className="w-full flex flex-col items-center justify-center bg-[#09090B]/50 rounded-3xl border border-white/5 p-6 shadow-2xl"
                                >
                                    <ResponsiveContainer width="100%" height={320}>
                                        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }} barSize={70}>
                                            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6B7280', fontSize: 12, fontStyle: 'italic' }} dy={10} />
                                            <Tooltip
                                                cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                                                content={({ active, payload }) => {
                                                    if (active && payload && payload.length) {
                                                        return (
                                                            <div className="bg-[#111116]/95 backdrop-blur-xl border border-white/10 p-5 rounded-2xl shadow-2xl">
                                                                <p className="text-white font-black text-xl mb-1">{payload[0].payload.name}</p>
                                                                <p className="text-orange-400 text-xs font-black uppercase tracking-widest">{payload[0].payload.label}</p>
                                                            </div>
                                                        );
                                                    }
                                                    return null;
                                                }}
                                            />
                                            <Bar dataKey="value" radius={[15, 15, 0, 0]}>
                                                {chartData.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.isMine ? 'url(#orangeGradient)' : 'url(#grayGradient)'} />
                                                ))}
                                            </Bar>
                                            <defs>
                                                <linearGradient id="orangeGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor="#FB923C" />
                                                    <stop offset="100%" stopColor="#E11D48" />
                                                </linearGradient>
                                                <linearGradient id="grayGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor="#3F3F46" />
                                                    <stop offset="100%" stopColor="#18181B" />
                                                </linearGradient>
                                            </defs>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </motion.div>

                                {/* Radar Chart Section */}
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ duration: 0.5, delay: 0.2 }}
                                    className="w-full flex flex-col items-center justify-center bg-[#09090B]/50 rounded-3xl border border-white/5 p-6 shadow-2xl"
                                >
                                    <ResponsiveContainer width="100%" height={320}>
                                        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={result.radar_data}>
                                            <PolarGrid stroke="rgba(255,255,255,0.05)" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#9CA3AF', fontSize: 10, fontWeight: 900 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                            <Radar name={result.competitor.name} dataKey="B" stroke="#4B5563" fill="#1F2937" fillOpacity={0.4} />
                                            <Radar name={result.my_brand.name} dataKey="A" stroke="#F43F5E" strokeWidth={3} fill="url(#radarGradient)" fillOpacity={0.65} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: 'rgba(9, 9, 11, 0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px', backdropFilter: 'blur(10px)' }}
                                                itemStyle={{ color: '#fff', fontSize: '12px', fontWeight: 'bold' }}
                                            />
                                            <defs>
                                                <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor="#FB923C" />
                                                    <stop offset="100%" stopColor="#F43F5E" />
                                                </linearGradient>
                                            </defs>
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </motion.div>
                            </div>
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
};
