import { useState, useEffect } from 'react';
import { LineChart as LineChartIcon, Activity, Flame, Share2, BarChart3, Fingerprint } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Metric } from "../components/ui/Metric";
import { fetchAnalytics } from "../lib/analyticsApi";
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, ScatterChart, Scatter, ZAxis, BarChart, Bar } from 'recharts';

// Fixed deterministic palette for the (at most ~10) competitor brands in the t-SNE plot.
const BRAND_COLORS = [
    "#6366F1", "#22D3EE", "#F97316", "#A855F7", "#10B981",
    "#EAB308", "#EC4899", "#3B82F6", "#F43F5E", "#84CC16",
];

const colorForBrand = (brandId, order) => {
    const idx = order.indexOf(brandId);
    return BRAND_COLORS[idx >= 0 ? idx % BRAND_COLORS.length : 0];
};

export const Analytics = ({ profile }) => {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchAnalytics()
            .then(d => setData(d))
            .catch(err => setError(err.message || String(err)));
    }, []);

    const pillarNames = data?.pillars?.names || [];
    const pillarKeywords = data?.pillars?.keywords || {};

    const heatmapBrands = data?.heatmap?.brands || [];
    const heatmapValues = data?.heatmap?.values || [];

    const tsnePoints = data?.tsne?.points || [];
    const brandOrder = [...new Set(tsnePoints.map(p => p.brand_id))];

    const toneTotals = data?.tone?.totals || {};
    const toneLabels = data?.tone?.labels || [];
    const userToneLabel = profile?.initialized ? profile?.tone_label : null;
    const toneData = toneLabels.map(label => ({
        name: label,
        competitors: toneTotals[label] || 0,
        userBrand: userToneLabel === label ? 1 : 0,
    }));

    const scoreTrend = data?.history?.score_trend || [];
    const trendData = scoreTrend.map((entry, i) => ({
        name: `#${i + 1}`,
        score: entry.score,
        event_type: entry.event_type,
    }));

    const counts = data?.history?.counts || { consistency: 0, benchmark: 0, rewrite: 0, total: 0 };

    const getHeatmapColor = (value) => {
        // value is already scaled 0-100 by the backend
        const opacity = Math.min(1, Math.max(0.1, value / 100));
        return `rgba(99, 102, 241, ${opacity})`;
    };

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-7xl mx-auto px-4 sm:px-6 mb-20">
            <div className="mb-12 flex items-center gap-5">
                <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-cyan-400 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.2)] text-white border border-indigo-400/30">
                    <LineChartIcon size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">System Analytics</h2>
                    <p className="text-gray-400 mt-2 text-lg">Database-derived analytics across the competitive brand corpus.</p>
                </div>
            </div>

            {error && (
                <Card className="w-full mb-8 border-red-500/30">
                    <p className="text-red-400 text-sm">Failed to reach analytics API: {error}</p>
                </Card>
            )}

            {/* Counters */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 sm:gap-6 md:gap-8 mb-12">
                <Metric label="Consistency Checks" value={data ? counts.consistency : "..."} trend="From analysis_history" delay={100} />
                <Metric label="Benchmarks" value={data ? counts.benchmark : "..."} trend="From analysis_history" delay={150} />
                <Metric label="Rewrites" value={data ? counts.rewrite : "..."} trend="From analysis_history" delay={200} />
                <Metric label="Total Analyses" value={data ? counts.total : "..."} trend="Live count" delay={250} />
            </div>

            {/* Row 1: Score Trend */}
            <Card delay={400} className="w-full mb-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
                    <Activity size={160} />
                </div>
                <div className="flex justify-between items-center mb-10">
                    <div>
                        <h3 className="text-2xl font-bold text-white mb-2">Score History Trend</h3>
                        <p className="text-gray-400 text-sm">Chronological pre-analysis scores from real analysis_history events.</p>
                    </div>
                </div>

                <div className="h-[300px] w-full mt-4">
                    {!data ? (
                        <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-white/5 rounded-2xl p-12">
                            <Activity className="text-indigo-500/20 mb-4 animate-pulse" size={48} />
                            <span className="text-gray-500 font-bold uppercase tracking-widest text-[10px]">Synchronizing...</span>
                        </div>
                    ) : trendData.length === 0 ? (
                        <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-white/5 rounded-2xl p-12">
                            <Activity className="text-gray-600 mb-4" size={48} />
                            <span className="text-gray-500 font-semibold text-sm">No score history yet.</span>
                        </div>
                    ) : (
                        <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                            <AreaChart data={trendData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
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
                    )}
                </div>
            </Card>

            {/* Row 2: Tone & t-SNE */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Tone Distribution */}
                <Card delay={500} className="flex flex-col">
                    <div className="mb-8">
                        <div className="flex items-center gap-3 mb-2">
                            <BarChart3 className="text-purple-400" size={24} />
                            <h3 className="text-xl font-bold text-white">Genome Tone Distribution</h3>
                        </div>
                        <p className="text-gray-400 text-sm">Deterministic tone label counts across the competitor corpus (formality + sentiment).</p>
                    </div>
                    <div className="h-[300px] w-full">
                        {data && toneData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                                <BarChart data={toneData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                                    <YAxis axisLine={false} tickLine={false} tick={{ fill: '#4B5563', fontSize: 11 }} allowDecimals={false} />
                                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#18181B', borderColor: '#3F3F46', color: '#fff' }} />
                                    <Bar dataKey="competitors" name="Competitor Corpus" fill="#1F2937" radius={[4, 4, 0, 0]} />
                                    {userToneLabel && (
                                        <Bar dataKey="userBrand" name="Your Brand" fill="#818CF8" radius={[4, 4, 0, 0]} />
                                    )}
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="w-full h-full flex items-center justify-center border border-dashed border-white/5 rounded-2xl">
                                <span className="text-gray-500 font-semibold text-sm">{data ? "No tone data available." : "Synchronizing..."}</span>
                            </div>
                        )}
                    </div>
                </Card>

                {/* t-SNE Plot */}
                <Card delay={600} className="flex flex-col">
                    <div className="mb-8 overflow-hidden relative">
                        <div className="flex items-center gap-3 mb-2">
                            <Share2 className="text-teal-400" size={24} />
                            <h3 className="text-xl font-bold text-white">Chunk-Level t-SNE Projection</h3>
                        </div>
                        <p className="text-gray-400 text-sm">
                            {data ? `${tsnePoints.length} sampled brand_chunks (384-d embeddings), random_state=${data?.tsne?.random_state}` : "2D projection of sampled brand_chunks embeddings"}
                        </p>
                    </div>
                    <div className="h-[300px] w-full border border-white/5 rounded-2xl bg-[#09090B]/50 p-4">
                        {data && tsnePoints.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                                <ScatterChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                    <XAxis type="number" dataKey="x" name="D1" hide domain={['auto', 'auto']} />
                                    <YAxis type="number" dataKey="y" name="D2" hide domain={['auto', 'auto']} />
                                    <ZAxis type="number" range={[60, 60]} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                                        if (active && payload && payload.length) {
                                            const p = payload[0].payload;
                                            return (
                                                <div className="bg-[#111116] px-4 py-2.5 border border-white/10 rounded-xl shadow-2xl text-white">
                                                    <div className="font-black text-sm uppercase tracking-tight">{p.brand_name}</div>
                                                    <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{p.chunk_id}</div>
                                                </div>
                                            )
                                        }
                                        return null;
                                    }} />
                                    {brandOrder.map(brandId => (
                                        <Scatter
                                            key={brandId}
                                            name={tsnePoints.find(p => p.brand_id === brandId)?.brand_name || brandId}
                                            data={tsnePoints.filter(p => p.brand_id === brandId)}
                                            fill={colorForBrand(brandId, brandOrder)}
                                        />
                                    ))}
                                </ScatterChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="w-full h-full flex items-center justify-center">
                                <span className="text-gray-500 font-semibold text-sm">{data ? "No t-SNE points available." : "Synchronizing..."}</span>
                            </div>
                        )}
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
                    <p className="text-gray-400 text-sm">
                        TF-IDF heatmap over 5 fixed pillar concepts; keyword sets per pillar are auto-derived from the competitor corpus (hover a column header).
                    </p>
                </div>

                {heatmapBrands.length === 0 || pillarNames.length === 0 ? (
                    <div className="w-full h-40 flex items-center justify-center border border-dashed border-white/5 rounded-2xl">
                        <span className="text-gray-500 font-semibold text-sm">{data ? "No heatmap data available." : "Synchronizing..."}</span>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <div className="min-w-[700px]">
                            {/* Headers */}
                            <div className="flex mb-6 border-b border-white/5 pb-4">
                                <div className="w-40 shrink-0 text-[10px] font-black text-gray-600 uppercase tracking-[0.2em]">Competitor</div>
                                {pillarNames.map((pillar, i) => (
                                    <div
                                        key={i}
                                        className="flex-1 text-center text-[10px] font-black text-gray-400 tracking-[0.15em] uppercase cursor-help"
                                        title={`Derived terms: ${(pillarKeywords[pillar] || []).map(k => k.term).join(', ')}`}
                                    >
                                        {pillar}
                                    </div>
                                ))}
                            </div>
                            {/* Matrix Rows */}
                            <div className="space-y-4">
                                {heatmapBrands.map((brandName, i) => (
                                    <div key={i} className="flex items-center group">
                                        <div className="w-40 shrink-0 text-xs font-black truncate pr-4 uppercase tracking-tight text-gray-500 flex items-center gap-2">
                                            <Fingerprint size={12} className="opacity-40" />
                                            {brandName}
                                        </div>
                                        {(heatmapValues[i] || []).map((value, j) => (
                                            <div key={j} className="flex-1 px-1.5">
                                                <div
                                                    className="h-12 w-full rounded-xl border border-white/5 transition-all group-hover:scale-[1.02] flex items-center justify-center shadow-lg relative overflow-hidden"
                                                    style={{ backgroundColor: getHeatmapColor(value) }}
                                                    title={`${brandName} -> ${pillarNames[j]}: ${value.toFixed(1)}`}
                                                >
                                                    {value > 80 && <div className="absolute inset-0 bg-white/5 animate-pulse" />}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </Card>

        </div>
    );
};

