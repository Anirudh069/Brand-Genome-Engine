import { useState, useEffect } from 'react';
import { LineChart as LineChartIcon, Activity, Flame, Share2, BarChart3 } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Metric } from "../components/ui/Metric";
import { API_BASE } from "../lib/constants";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, ScatterChart, Scatter, ZAxis, BarChart, Bar, Cell } from 'recharts';

export const Analytics = () => {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE}/analytics`)
            .then(res => res.json())
            .then(d => setData(d))
            .catch(err => console.error(err));
    }, []);

    // 1. Line Chart Data (Trajectory)
    const lineData = data?.trend ? [
        { name: 'Jan', score: data.trend[0] },
        { name: 'Feb', score: data.trend[1] },
        { name: 'Mar', score: data.trend[2] },
        { name: 'Apr', score: data.trend[3] },
        { name: 'May', score: data.trend[4] },
    ] : [];

    // 2. Bar Chart Data (Tone Histogram)
    const toneData = [
        { name: 'Very Casual', userBrand: 5, competitors: 12 },
        { name: 'Casual', userBrand: 15, competitors: 25 },
        { name: 'Neutral', userBrand: 22, competitors: 40 },
        { name: 'Formal', userBrand: 45, competitors: 18 },
        { name: 'Very Formal', userBrand: 13, competitors: 5 },
    ];

    // 3. Scatter Plot Data (t-SNE Clustering)
    const tsneData = [
        { x: -40, y: 30, z: 200, name: 'Competitor A (Luxury)', cluster: 'luxury' },
        { x: -35, y: 40, z: 200, name: 'Competitor B (Premium)', cluster: 'luxury' },
        { x: -50, y: 20, z: 200, name: 'Competitor C (Heritage)', cluster: 'luxury' },
        { x: 30, y: -20, z: 200, name: 'Competitor D (Sport)', cluster: 'sport' },
        { x: 40, y: -10, z: 200, name: 'Competitor E (Active)', cluster: 'sport' },
        { x: 5, y: 50, z: 200, name: 'Competitor F (Fashion)', cluster: 'fashion' },
        { x: 15, y: 45, z: 200, name: 'Competitor G (Trendy)', cluster: 'fashion' },
        // User Brand
        { x: -20, y: 35, z: 400, name: 'Your Brand (Rolex)', cluster: 'you' },
    ];

    // 4. Heatmap Matrix Data (Messaging Pillars)
    const heatmapThemes = ['Sustainability', 'Precision', 'Heritage', 'Value', 'Innovation'];
    const heatmapBrands = ['Your Brand', 'Omega', 'Tag Heuer', 'Tissot'];
    // Mock weights (0 to 1)
    const heatmapData = [
        [0.1, 0.9, 0.95, 0.2, 0.8], // Your Brand
        [0.3, 0.8, 0.7, 0.4, 0.7],  // Omega
        [0.2, 0.9, 0.6, 0.5, 0.4],  // Tag Heuer
        [0.4, 0.6, 0.5, 0.85, 0.3], // Tissot
    ];

    const getHeatmapColor = (weight) => {
        // Map 0-1 to blue scale opacity
        return `rgba(59, 130, 246, ${weight * 0.9 + 0.1})`;
    };

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both">
            <div className="mb-12 flex items-center gap-5">
                <div className="p-3.5 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-2xl shadow-[0_0_30px_rgba(59,130,246,0.2)] text-white border border-blue-400/30">
                    <LineChartIcon size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Data Analytics</h2>
                    <p className="text-gray-400 mt-2 text-lg">Multi-dimensional visualizations of your brand's genome.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mb-12">
                <Metric label="Copies Analyzed" value={data ? data.total_analyzed : "..."} trend="Live Tracking" delay={100} />
                <Metric label="Avg Consistency" value={data ? `${data.avg_consistency}%` : "..."} trend="Current" delay={200} />
                <Metric label="Deviations Fixed" value={data ? data.deviations_fixed : "..."} delay={300} />
            </div>

            {/* Row 1: Area Chart (Trajectory) */}
            <Card delay={400} className="w-full mb-8">
                <div className="flex justify-between items-center mb-10">
                    <div>
                        <h3 className="text-2xl font-bold text-white mb-2">Consistency Trajectory</h3>
                        <p className="text-gray-400">Score variations over time.</p>
                    </div>
                </div>

                <div className="h-[300px] w-full mt-4">
                    {data ? (
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <AreaChart data={lineData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6B7280', fontSize: 13, fontWeight: 600 }} dy={10} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6B7280', fontSize: 13, fontWeight: 600 }} domain={['dataMin - 5', 'dataMax + 5']} />
                                <Tooltip contentStyle={{ backgroundColor: 'rgba(17, 17, 22, 0.9)', backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} />
                                <Area type="monotone" dataKey="score" stroke="#3B82F6" strokeWidth={4} fillOpacity={1} fill="url(#colorScore)" activeDot={{ r: 8, fill: '#60A5FA', stroke: '#fff', strokeWidth: 2 }} />
                            </AreaChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-white/5 rounded-2xl">
                            <Activity className="text-blue-500/20 mb-4 animate-pulse" />
                        </div>
                    )}
                </div>
            </Card>

            {/* Row 2: Histogram & Scatter */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Tone Histogram */}
                <Card delay={500} className="flex flex-col">
                    <div className="mb-8">
                        <div className="flex items-center gap-3 mb-2">
                            <BarChart3 className="text-purple-400" size={24} />
                            <h3 className="text-xl font-bold text-white">Tone Distribution</h3>
                        </div>
                        <p className="text-gray-400 text-sm">Formality density mapping (Your Brand vs Market Avg)</p>
                    </div>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <BarChart data={toneData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4B5563', fontSize: 11 }} />
                                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#18181B', borderColor: '#3F3F46', color: '#fff' }} />
                                <Bar dataKey="competitors" name="Market Avg" fill="#4B5563" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="userBrand" name="Your Brand" fill="#A855F7" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* t-SNE Plot */}
                <Card delay={600} className="flex flex-col">
                    <div className="mb-8">
                        <div className="flex items-center gap-3 mb-2">
                            <Share2 className="text-teal-400" size={24} />
                            <h3 className="text-xl font-bold text-white">t-SNE Embeddings</h3>
                        </div>
                        <p className="text-gray-400 text-sm">Semantic proximity of brand profiles in 2D vector space</p>
                    </div>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <XAxis type="number" dataKey="x" name="Dimension 1" hide />
                                <YAxis type="number" dataKey="y" name="Dimension 2" hide />
                                <ZAxis type="number" dataKey="z" range={[60, 400]} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        return <div className="bg-[#18181B] px-3 py-2 border border-[#3F3F46] rounded-md text-white font-bold">{payload[0].payload.name}</div>
                                    }
                                    return null;
                                }} />
                                <Scatter name="Competitors" data={tsneData.filter(d => d.cluster !== 'you')} fill="#4B5563" />
                                <Scatter name="Your Brand" data={tsneData.filter(d => d.cluster === 'you')} fill="#2DD4BF" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </Card>
            </div>

            {/* Row 3: Heatmap */}
            <Card delay={700} className="w-full">
                <div className="mb-8">
                    <div className="flex items-center gap-3 mb-2">
                        <Flame className="text-orange-400" size={24} />
                        <h3 className="text-xl font-bold text-white">Messaging Pillars Heatmap</h3>
                    </div>
                    <p className="text-gray-400 text-sm">Thematic intensity matrix highlighting core keyword groupings</p>
                </div>

                <div className="overflow-x-auto">
                    <div className="min-w-[600px]">
                        {/* Headers */}
                        <div className="flex mb-4">
                            <div className="w-32 shrink-0"></div>
                            {heatmapThemes.map((theme, i) => (
                                <div key={i} className="flex-1 text-center text-xs font-bold text-gray-400 tracking-wider uppercase -rotate-12 transform origin-bottom-left lg:rotate-0">
                                    {theme}
                                </div>
                            ))}
                        </div>
                        {/* Matrix Rows */}
                        {heatmapBrands.map((brand, i) => (
                            <div key={i} className="flex items-center mb-2 group">
                                <div className={`w-32 shrink-0 text-sm font-bold truncate pr-4 ${i === 0 ? 'text-white' : 'text-gray-500'}`}>
                                    {brand}
                                </div>
                                {heatmapData[i].map((weight, j) => (
                                    <div key={j} className="flex-1 px-1">
                                        <div
                                            className="h-10 w-full rounded-md border border-white/5 transition-transform group-hover:scale-[1.02] flex items-center justify-center opacity-90 hover:opacity-100"
                                            style={{ backgroundColor: getHeatmapColor(weight) }}
                                            title={`${brand} -> ${heatmapThemes[j]}: ${(weight * 100).toFixed(0)}%`}
                                        >
                                            {/* Optional: text label for density if large enough */}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            </Card>

        </div>
    );
};
