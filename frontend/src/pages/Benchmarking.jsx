import { useState } from 'react';
import { BarChart2, Loader2, ArrowRight, Target } from 'lucide-react';
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
    const [competitor, setCompetitor] = useState("omega");
    const [metric, setMetric] = useState("Sentiment Distribution");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleBenchmark = async () => {
        if (!profile) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/benchmark`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ my_brand: profile.name, competitor, metric })
            });
            if (res.ok) {
                const data = await res.json();
                setResult(data);
            }
        } catch (err) {
            console.error(err);
        }
        setLoading(false);
    };

    const chartData = result ? [
        { name: result.my_brand.name, value: result.my_brand.value, label: result.my_brand.label, isMine: true },
        { name: result.competitor.name, value: result.competitor.value, label: result.competitor.label, isMine: false }
    ] : [];

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both relative">
            <div className="mb-12 flex items-center gap-5 relative z-10">
                <div className="p-3.5 bg-gradient-to-br from-orange-400 to-rose-500 rounded-2xl shadow-[0_0_30px_rgba(249,115,22,0.2)] text-white border border-orange-400/30">
                    <BarChart2 size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Market Benchmarking</h2>
                    <p className="text-gray-400 mt-2 text-lg">Compare your brand parameters directly against top competitors.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8 relative z-10">
                <Card className="lg:col-span-1 flex flex-col" delay={100}>
                    <h3 className="text-xl font-bold text-white mb-8 tracking-wide">Simulation Parameters</h3>
                    <div className="space-y-8 flex-1">
                        <div className="group">
                            <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2 transition-colors group-focus-within:text-orange-400">Target Competitor</label>
                            <div className="relative">
                                <select
                                    value={competitor}
                                    onChange={(e) => setCompetitor(e.target.value)}
                                    className="w-full px-4 py-3.5 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer"
                                >
                                    <option value="rolex" className="bg-[#111116] text-white">Rolex</option>
                                    <option value="omega" className="bg-[#111116] text-white">Omega</option>
                                    <option value="tagheuer" className="bg-[#111116] text-white">Tag Heuer</option>
                                    <option value="tissot" className="bg-[#111116] text-white">Tissot</option>
                                    <option value="titan" className="bg-[#111116] text-white">Titan</option>
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                </div>
                            </div>
                        </div>

                        <div className="group">
                            <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2 transition-colors group-focus-within:text-orange-400">Head-to-Head Metric</label>
                            <div className="relative">
                                <select
                                    value={metric}
                                    onChange={(e) => setMetric(e.target.value)}
                                    className="w-full px-4 py-3.5 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer"
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

                    <Button primary className="w-full bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-400 hover:to-rose-500 shadow-orange-500/20 text-lg mt-8 py-4" onClick={handleBenchmark} disabled={loading || !profile}>
                        {loading ? <Loader2 className="animate-spin" /> : <>Run Simulation <ArrowRight size={18} /></>}
                    </Button>
                </Card>

                <Card className="lg:col-span-2 flex flex-col p-8 min-h-[600px] border-orange-500/20 shadow-[0_0_40px_rgba(249,115,22,0.05)] overflow-hidden" delay={200}>
                    <NeuralNetworkSVG />

                    {!result ? (
                        <div className="flex-1 flex flex-col items-center justify-center text-center relative z-10">
                            <div className="relative mb-8">
                                <div className="absolute inset-0 bg-gradient-to-tr from-orange-500/20 to-rose-500/20 blur-[40px] rounded-full animate-pulse" />
                                <Target size={72} className="text-orange-500/80 relative z-10" />
                            </div>
                            <h3 className="text-2xl font-bold text-white mb-4">Awaiting Parameters</h3>
                            <p className="text-gray-500 max-w-md text-lg leading-relaxed">Select a competitor and configure parameters to compute a multi-dimensional metric analysis.</p>
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col w-full relative z-10">
                            <div className="flex items-center justify-between mb-8 border-b border-white/5 pb-6">
                                <div>
                                    <h3 className="text-xl md:text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-rose-400">{metric}</h3>
                                    <p className="text-gray-400 font-medium tracking-wide">Direct Comparison</p>
                                </div>
                                <div className="flex gap-4">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.5)]" />
                                        <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">{result.my_brand.name}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-gray-500" />
                                        <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">{result.competitor.name}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 flex-1">
                                {/* Bar Chart Section */}
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.1 }}
                                    className="w-full flex flex-col items-center justify-center bg-[#0C0C0E]/50 rounded-2xl border border-white/5 p-4"
                                >
                                    <ResponsiveContainer width="100%" height={300}>
                                        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }} barSize={60}>
                                            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 14, fontWeight: 600 }} dy={10} />
                                            <Tooltip
                                                cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                                                content={({ active, payload }) => {
                                                    if (active && payload && payload.length) {
                                                        return (
                                                            <div className="bg-[#111116]/90 backdrop-blur-md border border-white/10 p-4 rounded-xl shadow-2xl">
                                                                <p className="text-white font-bold text-lg mb-1">{payload[0].payload.name}</p>
                                                                <p className="text-gray-400 text-sm font-medium">{payload[0].payload.label}</p>
                                                            </div>
                                                        );
                                                    }
                                                    return null;
                                                }}
                                            />
                                            <Bar dataKey="value" radius={[12, 12, 0, 0]}>
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
                                                    <stop offset="0%" stopColor="#4B5563" />
                                                    <stop offset="100%" stopColor="#1F2937" />
                                                </linearGradient>
                                            </defs>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </motion.div>

                                {/* Radar Chart Section */}
                                <motion.div
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.3 }}
                                    className="w-full flex flex-col items-center justify-center bg-[#0C0C0E]/50 rounded-2xl border border-white/5 p-4"
                                >
                                    <ResponsiveContainer width="100%" height={320}>
                                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={result.radar_data}>
                                            <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#9CA3AF', fontSize: 11, fontWeight: 600 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                            <Radar name={result.competitor.name} dataKey="B" stroke="#6B7280" fill="#4B5563" fillOpacity={0.3} />
                                            <Radar name={result.my_brand.name} dataKey="A" stroke="#F43F5E" strokeWidth={2} fill="url(#radarGradient)" fillOpacity={0.6} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: 'rgba(17, 17, 22, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                                                itemStyle={{ color: '#fff' }}
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
