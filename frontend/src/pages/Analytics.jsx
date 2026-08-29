import { useState, useEffect } from 'react';
import { LineChart as LineChartIcon, Activity, Flame, Share2, BarChart3, Fingerprint } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Metric } from "../components/ui/Metric";
import { API_BASE } from "../lib/constants";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, ScatterChart, Scatter, ZAxis, BarChart, Bar, Cell } from 'recharts';

export const Analytics = ({ profile }) => {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE}/analytics`)
            .then(res => res.json())
            .then(d => setData(d))
            .catch(err => console.error(err));
    }, []);

    // 1. Line Chart Data (Trajectory)
    const lineData = data?.trend ? [
        { name: 'Jan', score: data.trend[0] || 70 },
        { name: 'Feb', score: data.trend[1] || 75 },
        { name: 'Mar', score: data.trend[2] || 80 },
        { name: 'Apr', score: data.trend[3] || 85 },
        { name: 'May', score: data.trend[4] || 84 },
    ] : [];

    // 2. Bar Chart Data (Tone Histogram)
    // Map bins to labels: 0.0-0.2 (Very Casual), 0.2-0.4 (Casual), etc.
    const toneLabels = ['V. Casual', 'Casual', 'Mixed', 'Formal', 'V. Formal'];
    const toneData = data?.tone_histogram?.counts ? data.tone_histogram.counts.reduce((acc, count, i) => {
        // Group 10 bins into 5 labels for cleaner UI
        const index = Math.floor(i / 2);
        if (!acc[index]) acc[index] = { name: toneLabels[index], competitors: 0, userBrand: 0 };
        acc[index].competitors += count;
        return acc;
    }, []) : [];

    // Add user brand peak to histogram
    if (toneData.length > 0 && profile?.avg_formality !== undefined) {
        const userIdx = Math.min(4, Math.floor(profile.avg_formality * 5));
        toneData[userIdx].userBrand = 25; // Visual peak for user brand
    }

    // 3. Scatter Plot Data (t-SNE Clustering)
    const tsnePoints = data?.tsne_points ? data.tsne_points.map(p => ({
        ...p,
        z: p.brand_id === 'user_brand' ? 400 : 200,
        name: p.brand_name
    })) : [];

    const userPoint = tsnePoints.find(p => p.brand_id === 'user_brand') || 
                      (profile?.tsne_x !== undefined ? { x: profile.tsne_x, y: profile.tsne_y, z: 400, name: profile.name || "Your Brand", brand_id: 'user_brand' } : null);
    const competitorPoints = tsnePoints.filter(p => p.brand_id !== 'user_brand');

    // 4. Heatmap Matrix Data (Messaging Pillars)
    const heatmapThemes = data?.heatmap?.pillars || ['Pillar A', 'Pillar B', 'Pillar C', 'Pillar D', 'Pillar E'];
    const heatmapBrands = data?.heatmap?.brands || [];

    const getHeatmapColor = (weight) => {
        // Map 0-1 to blue scale opacity
        return `rgba(99, 102, 241, ${weight * 0.9 + 0.1})`;
    };

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-7xl mx-auto px-4 sm:px-6 mb-20">
            <div className="mb-12 flex items-center gap-5">
                <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-cyan-400 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.2)] text-white border border-indigo-400/30">
                    <LineChartIcon size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">System Analytics</h2>
                    <p className="text-gray-400 mt-2 text-lg">Multi-dimensional visualizations of the cross-market brand genome.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mb-12">
                <Metric label="Copies Analyzed" value={data ? data.total_analyzed : "..."} trend="Live Tracking" delay={100} />
                <Metric label="Avg Consistency" value={data ? `${data.avg_consistency}%` : "..."} trend="System-wide" delay={200} />
                <Metric label="Deviations Fixed" value={data ? data.deviations_fixed : "..."} trend="Post-Rewrite" delay={300} />
            </div>

            {/* Row 1: Area Chart (Trajectory) */}
            <Card delay={400} className="w-full mb-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
                    <Activity size={160} />
                </div>
                <div className="flex justify-between items-center mb-10">
                    <div>
                        <h3 className="text-2xl font-bold text-white mb-2">Consistency Trajectory</h3>
                        <p className="text-gray-400 text-sm">Aggregated score variations over the engine's operation period.</p>
                    </div>
                </div>

                <div className="h-[300px] w-full mt-4">
                    {data ? (
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <AreaChart data={lineData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366F1" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#6366F1" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#6B7280', fontSize: 13, fontWeight: 600 }} dy={10} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#6B7280', fontSize: 13, fontWeight: 600 }} domain={[0, 100]} />
                                <Tooltip contentStyle={{ backgroundColor: 'rgba(17, 17, 22, 0.9)', backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} />
                                <Area type="monotone" dataKey="score" stroke="#6366F1" strokeWidth={4} fillOpacity={1} fill="url(#colorScore)" activeDot={{ r: 8, fill: '#818CF8', stroke: '#fff', strokeWidth: 2 }} />
                            </AreaChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-white/5 rounded-2xl p-12">
                            <Activity className="text-indigo-500/20 mb-4 animate-pulse" size={48} />
                            <span className="text-gray-500 font-bold uppercase tracking-widest text-[10px]">Synchronizing...</span>
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
                            <h3 className="text-xl font-bold text-white">Genome Tone Distribution</h3>
                        </div>
                        <p className="text-gray-400 text-sm">Formality density mapping (Market Average vs Your Brand Projection)</p>
                    </div>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <BarChart data={toneData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4B5563', fontSize: 11 }} />
                                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#18181B', borderColor: '#3F3F46', color: '#fff' }} />
                                <Bar dataKey="competitors" name="Market Base" fill="#1F2937" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="userBrand" name="Your Brand Projection" fill="#818CF8" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* t-SNE Plot */}
                <Card delay={600} className="flex flex-col">
                    <div className="mb-8 overflow-hidden relative">
                        <div className="flex items-center gap-3 mb-2">
                            <Share2 className="text-teal-400" size={24} />
                            <h3 className="text-xl font-bold text-white">t-SNE Semantic Proximity</h3>
                        </div>
                        <p className="text-gray-400 text-sm">2D projection of brand centroids from 384-dimension vector space</p>
                    </div>
                    <div className="h-[300px] w-full border border-white/5 rounded-2xl bg-[#09090B]/50 p-4">
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <ScatterChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                <XAxis type="number" dataKey="x" name="D1" hide domain={['auto', 'auto']} />
                                <YAxis type="number" dataKey="y" name="D2" hide domain={['auto', 'auto']} />
                                <ZAxis type="number" dataKey="z" range={[80, 400]} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        const p = payload[0].payload;
                                        return (
                                            <div className="bg-[#111116] px-4 py-2.5 border border-white/10 rounded-xl shadow-2xl text-white">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <div className={`w-2 h-2 rounded-full ${p.brand_id === 'user_brand' ? 'bg-indigo-400' : 'bg-gray-500'}`} />
                                                    <span className="font-black text-sm uppercase tracking-tight">{p.name}</span>
                                                </div>
                                                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">
                                                    Vector Rank: {p.brand_id === 'user_brand' ? 'Active' : 'Reference'}
                                                </div>
                                            </div>
                                        )
                                    }
                                    return null;
                                }} />
                                <Scatter name="Competitors" data={competitorPoints} fill="#1F2937" stroke="rgba(255,255,255,0.1)" />
                                {userPoint && (
                                    <Scatter 
                                        name="Your Brand" 
                                        data={[userPoint]} 
                                        fill="#6366F1" 
                                        shape={(props) => (
                                            <g>
                                                <circle cx={props.cx} cy={props.cy} r={12} fill="#6366F1" className="animate-pulse" filter="blur(4px)" opacity="0.5" />
                                                <circle cx={props.cx} cy={props.cy} r={6} fill="#6366F1" stroke="#fff" strokeWidth={2} />
                                            </g>
                                        )}
                                    />
                                )}
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </Card>
            </div>

            {/* Row 3: Heatmap */}
            <Card delay={700} className="w-full relative overflow-hidden">
               <div className="absolute -bottom-10 -right-10 p-12 opacity-5 pointer-events-none">
                    <Flame size={200} />
                </div>
                <div className="mb-10">
                    <div className="flex items-center gap-3 mb-2">
                        <Flame className="text-orange-400" size={24} />
                        <h3 className="text-xl font-bold text-white">Messaging Pillar Intensity</h3>
                    </div>
                    <p className="text-gray-400 text-sm">Heatmap of thematic keyword density aggregated across the competitive landscape.</p>
                </div>

                <div className="overflow-x-auto">
                    <div className="min-w-[700px]">
                        {/* Headers */}
                        <div className="flex mb-6 border-b border-white/5 pb-4">
                            <div className="w-40 shrink-0 text-[10px] font-black text-gray-600 uppercase tracking-[0.2em]">Target Identity</div>
                            {heatmapThemes.map((theme, i) => (
                                <div key={i} className="flex-1 text-center text-[10px] font-black text-gray-400 tracking-[0.15em] uppercase">
                                    {theme}
                                </div>
                            ))}
                        </div>
                        {/* Matrix Rows */}
                        <div className="space-y-4">
                            {heatmapBrands.map((brand, i) => (
                                <div key={i} className="flex items-center group">
                                    <div className={`w-40 shrink-0 text-xs font-black truncate pr-4 uppercase tracking-tight ${brand.brand_id === 'user_brand' ? 'text-indigo-400 flex items-center gap-2' : 'text-gray-500'}`}>
                                        {brand.brand_id === 'user_brand' && <Fingerprint size={12} />}
                                        {brand.brand_name}
                                    </div>
                                    {brand.weights.map((weight, j) => (
                                        <div key={j} className="flex-1 px-1.5">
                                            <div
                                                className="h-12 w-full rounded-xl border border-white/5 transition-all group-hover:scale-[1.02] flex items-center justify-center shadow-lg relative overflow-hidden"
                                                style={{ backgroundColor: getHeatmapColor(weight) }}
                                                title={`${brand.brand_name} -> ${heatmapThemes[j]}: ${(weight * 100).toFixed(0)}%`}
                                            >
                                                {weight > 0.8 && <div className="absolute inset-0 bg-white/5 animate-pulse" />}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Card>

        </div>
    );
};
