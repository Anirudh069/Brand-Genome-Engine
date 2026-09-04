import { useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, BarChart2, Loader2, ShieldCheck, Sparkles, Target } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { fetchBenchmarkBrands, runBenchmark } from '../lib/benchmarkApi';

const METRIC_OPTIONS = [
  { value: 'tone', label: 'Tone' },
  { value: 'sentiment', label: 'Sentiment' },
  { value: 'readability', label: 'Readability' },
];

const CHART_COLORS = {
  user: '#FB923C',
  competitor: '#6B7280',
};

const metricLabelFor = (metric) => METRIC_OPTIONS.find((option) => option.value === metric)?.label || metric;

export const Benchmarking = ({ profile, onGoToSetup }) => {
  const [competitors, setCompetitors] = useState([]);
  const [competitorBrandId, setCompetitorBrandId] = useState('');
  const [metric, setMetric] = useState('tone');
  const [brandsLoading, setBrandsLoading] = useState(true);
  const [brandsError, setBrandsError] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState(null);
  const [result, setResult] = useState(null);

  const hasGenome = Boolean(profile?.initialized);
  const userDesignation = profile?.designation || profile?.brand_name || profile?.name || 'Active Brand';

  useEffect(() => {
    let cancelled = false;

    const loadCompetitors = async () => {
      setBrandsLoading(true);
      setBrandsError(null);
      try {
        const data = await fetchBenchmarkBrands();
        if (cancelled) return;
        const list = Array.isArray(data) ? data : [];
        setCompetitors(list);
        setCompetitorBrandId((current) => current || list[0]?.brand_id || '');
      } catch (error) {
        if (cancelled) return;
        setBrandsError(error.message || 'Failed to load benchmark competitors.');
      } finally {
        if (!cancelled) {
          setBrandsLoading(false);
        }
      }
    };

    loadCompetitors();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleBenchmark = async () => {
    if (!hasGenome || !competitorBrandId || !metric) return;

    setBenchmarkLoading(true);
    setBenchmarkError(null);
    try {
      const data = await runBenchmark({ competitorBrandId, metric });
      setResult(data);
    } catch (error) {
      setResult(null);
      setBenchmarkError(error.message || 'Benchmark request failed.');
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const chartData = result
    ? result.labels.map((label, index) => ({
        label,
        user: result.user_series[index],
        competitor: result.competitor_series[index],
      }))
    : [];

  const selectedCompetitor = competitors.find((item) => item.brand_id === competitorBrandId);

  return (
    <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both relative max-w-7xl mx-auto px-4 sm:px-6 mb-20">
      <div className="mb-12 flex items-center gap-5 relative z-10">
        <div className="p-3.5 bg-gradient-to-br from-orange-400 to-rose-500 rounded-2xl shadow-[0_0_30px_rgba(249,115,22,0.2)] text-white border border-orange-400/30">
          <BarChart2 size={28} />
        </div>
        <div>
          <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Market Benchmarking</h2>
          <p className="text-gray-400 mt-2 text-lg">Compare the active genome against one real competitor from SQLite.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        <Card className="lg:col-span-4 flex flex-col" delay={100}>
          <h3 className="text-xl font-bold text-white mb-8 tracking-wide flex items-center gap-2">
            <ShieldCheck size={20} className="text-orange-500" />
            Benchmark Controls
          </h3>

          <div className="space-y-6 flex-1">
            <div className="rounded-2xl border border-white/5 bg-[#09090B]/60 p-4">
              <span className="block text-[10px] font-bold tracking-[0.2em] uppercase text-gray-500 mb-2">User Brand</span>
              <div className="text-white font-bold text-lg">{userDesignation}</div>
              <p className="text-xs text-gray-500 mt-2">The active genome is implied by the persisted user profile.</p>
            </div>

            {!hasGenome && (
              <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4 text-amber-200">
                <div className="flex items-start gap-3">
                  <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                  <div>
                    <div className="font-bold">Genome required</div>
                    <p className="text-sm text-amber-100/80 mt-1">Initialize the genome before running a benchmark. No fallback user profile is used.</p>
                    {onGoToSetup && (
                      <button
                        type="button"
                        onClick={onGoToSetup}
                        className="mt-3 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-amber-100 hover:text-white"
                      >
                        Go to Genome Setup
                        <ArrowRight size={14} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div className="group">
              <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-gray-500 mb-3 transition-colors group-focus-within:text-orange-400">
                Reference Competitor
              </label>
              <div className="relative">
                <select
                  value={competitorBrandId}
                  onChange={(event) => {
                    setCompetitorBrandId(event.target.value);
                    setResult(null);
                    setBenchmarkError(null);
                  }}
                  disabled={brandsLoading || competitors.length === 0}
                  className="w-full px-4 py-4 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer font-bold text-sm disabled:opacity-50"
                >
                  {brandsLoading && <option value="">Loading competitors...</option>}
                  {!brandsLoading && competitors.length === 0 && <option value="">No competitors available</option>}
                  {competitors.map((competitor) => (
                    <option key={competitor.brand_id} value={competitor.brand_id} className="bg-[#111116] text-white">
                      {competitor.designation}
                    </option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
            </div>

            <div className="group">
              <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-gray-500 mb-3 transition-colors group-focus-within:text-orange-400">
                Metric
              </label>
              <div className="relative">
                <select
                  value={metric}
                  onChange={(event) => {
                    setMetric(event.target.value);
                    setResult(null);
                    setBenchmarkError(null);
                  }}
                  className="w-full px-4 py-4 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500 outline-none transition-all duration-300 text-gray-100 appearance-none shadow-inner cursor-pointer font-bold text-sm"
                >
                  {METRIC_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value} className="bg-[#111116] text-white">
                      {option.label}
                    </option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              </div>
            </div>

            <Button
              primary
              className="w-full bg-gradient-to-r from-orange-500 to-rose-600 hover:from-orange-400 hover:to-rose-500 shadow-[0_0_20px_rgba(249,115,22,0.3)] text-lg mt-8 py-5 group"
              onClick={handleBenchmark}
              disabled={brandsLoading || benchmarkLoading || !hasGenome || !competitorBrandId || !metric}
            >
              {benchmarkLoading ? (
                <Loader2 className="animate-spin" />
              ) : (
                <>
                  Run Benchmark
                  <ArrowRight size={18} className="ml-2 transition-transform group-hover:translate-x-1" />
                </>
              )}
            </Button>

            {brandsError && (
              <div className="text-xs font-bold px-4 py-3 rounded-xl border bg-red-500/10 border-red-500/20 text-red-400">
                {brandsError}
              </div>
            )}

            {benchmarkError && (
              <div className="text-xs font-bold px-4 py-3 rounded-xl border bg-red-500/10 border-red-500/20 text-red-400">
                {benchmarkError}
              </div>
            )}

            {selectedCompetitor && (
              <div className="rounded-2xl border border-white/5 bg-[#09090B]/60 p-4 text-xs text-gray-400">
                Comparing against <span className="font-bold text-white">{selectedCompetitor.designation}</span> with the <span className="font-bold text-white">{metricLabelFor(metric)}</span> metric.
              </div>
            )}
          </div>
        </Card>

        <Card className="lg:col-span-8 flex flex-col p-8 min-h-[600px] border-orange-500/20 shadow-[0_0_40px_rgba(249,115,22,0.05)] overflow-hidden" delay={200}>
          <div className="absolute inset-0 pointer-events-none opacity-10">
            <Sparkles className="absolute right-8 top-8 text-orange-400" size={72} />
          </div>

          {!hasGenome ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center relative z-10">
              <div className="relative mb-8">
                <div className="absolute inset-0 bg-gradient-to-tr from-orange-500/20 to-rose-500/20 blur-[40px] rounded-full animate-pulse" />
                <Target size={72} className="text-orange-500/80 relative z-10" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 tracking-tight">Genome Not Initialized</h3>
              <p className="text-gray-500 max-w-sm text-lg leading-relaxed mx-auto">
                The benchmark compares the active user genome against one selected competitor. Initialize the genome first, then rerun the comparison.
              </p>
            </div>
          ) : !result ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center relative z-10">
              <div className="relative mb-8">
                <div className="absolute inset-0 bg-gradient-to-tr from-orange-500/20 to-rose-500/20 blur-[40px] rounded-full animate-pulse" />
                <Target size={72} className="text-orange-500/80 relative z-10" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 tracking-tight">Awaiting Benchmark</h3>
              <p className="text-gray-500 max-w-sm text-lg leading-relaxed mx-auto">
                Select a competitor and one metric, then run the comparison to render a single chart and summary.
              </p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col w-full relative z-10 animate-in fade-in duration-700">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between mb-10 border-b border-white/5 pb-6">
                <div>
                  <h3 className="text-xl md:text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-rose-400 uppercase tracking-tight">
                    {metricLabelFor(result.metric)}
                  </h3>
                  <p className="text-gray-400 font-bold tracking-widest text-[10px] mt-1 uppercase">User vs Competitor</p>
                </div>

                <div className="flex flex-wrap gap-3 text-[10px] font-black uppercase tracking-widest">
                  <span className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-orange-500/10 text-orange-300 border border-orange-500/20">
                    <span className="w-2.5 h-2.5 rounded-full bg-orange-400" />
                    {result.user_brand?.designation || userDesignation}
                  </span>
                  <span className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-white/5 text-gray-300 border border-white/10">
                    <span className="w-2.5 h-2.5 rounded-full bg-gray-500" />
                    {result.competitor_brand?.designation || selectedCompetitor?.designation || 'Competitor'}
                  </span>
                </div>
              </div>

              <div className="rounded-3xl border border-white/5 bg-[#09090B]/50 p-5 md:p-6 shadow-2xl">
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }} barCategoryGap="28%">
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={{ fill: '#9CA3AF', fontSize: 12, fontWeight: 800 }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fill: '#6B7280', fontSize: 12 }} tickLine={false} axisLine={false} />
                    <Tooltip
                      cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                      content={({ active, payload }) => {
                        if (!active || !payload || !payload.length) return null;
                        const row = payload[0].payload;
                        return (
                          <div className="bg-[#111116]/95 backdrop-blur-xl border border-white/10 p-4 rounded-2xl shadow-2xl">
                            <p className="text-white font-black text-base mb-2">{row.label}</p>
                            <div className="space-y-1 text-xs font-bold uppercase tracking-widest">
                              <p className="text-orange-300">User: {row.user}</p>
                              <p className="text-gray-300">Competitor: {row.competitor}</p>
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="user" name={result.user_brand?.designation || userDesignation} radius={[10, 10, 0, 0]}>
                      {chartData.map((entry) => (
                        <Cell key={`user-${entry.label}`} fill={CHART_COLORS.user} />
                      ))}
                    </Bar>
                    <Bar dataKey="competitor" name={result.competitor_brand?.designation || selectedCompetitor?.designation || 'Competitor'} radius={[10, 10, 0, 0]}>
                      {chartData.map((entry) => (
                        <Cell key={`competitor-${entry.label}`} fill={CHART_COLORS.competitor} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-6 rounded-2xl border border-white/5 bg-[#09090B]/60 p-5 text-gray-300">
                <div className="text-[10px] font-black uppercase tracking-[0.2em] text-gray-500 mb-2">Summary</div>
                <p className="text-sm md:text-base leading-relaxed">{result.summary_text}</p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};