import { useState } from 'react';
import { CheckCircle, Loader2, Sparkles, AlertCircle, Info, BookOpen, Fingerprint } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { TextArea } from "../components/ui/TextArea";
import { Button } from "../components/ui/Button";
import { BrandSelector } from "../components/ui/BrandSelector";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { SuggestionsPanel } from "../components/ui/SuggestionsPanel";
import { RewritePanel } from "../components/ui/RewritePanel";
import { GroundingExamples } from "../components/ui/GroundingExamples";
import { API_BASE } from "../lib/constants";

const ENABLE_REWRITE_UI = true;

// Helper component for rendering Before/After scores with animated progress bars
const DualMetricBar = ({ label, beforeVal, afterVal, baseColor }) => {
    return (
        <div className="group mb-6">
            <div className="flex justify-between text-xs font-bold tracking-widest uppercase mb-3">
                <span className="text-gray-400 group-hover:text-gray-200 transition-colors flex items-center gap-2">
                    {label}
                </span>
                <div className="flex gap-4">
                    <span className="text-gray-400">B: {beforeVal}%</span>
                    <span className="text-indigo-300">A: {afterVal}%</span>
                </div>
            </div>
            <div className="space-y-2">
                {/* Before Bar */}
                <div className="w-full bg-[#1A1A24] rounded-full h-1.5 overflow-hidden border border-white/5 opacity-60">
                    <div
                        className="bg-gray-500 h-full rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${beforeVal}%` }}
                    />
                </div>
                {/* After Bar */}
                <div className="w-full bg-[#1A1A24] rounded-full h-2.5 overflow-hidden border border-indigo-500/20 shadow-inner">
                    <div
                        className={`bg-gradient-to-r ${baseColor} h-full rounded-full transition-all duration-1000 ease-out relative shadow-[0_0_10px_currentColor]`}
                        style={{ width: `${afterVal}%` }}
                    >
                        <div className="absolute inset-0 bg-white/20 w-full h-full animate-[shimmer_2s_infinite]" />
                    </div>
                </div>
            </div>
        </div>
    );
};


export const ConsistencyCheck = () => {
    const [selectedBrand, setSelectedBrand] = useState(null);
    const [copyText, setCopyText] = useState("This watch is awesome and super easy to wear every day. Cool design.");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleAction = async () => {
        if (!selectedBrand || !copyText) return;
        setLoading(true);
        setResult(null);

        try {
            // Using /api/rewrite if UI is enabled, else /api/check-consistency
            const endpoint = ENABLE_REWRITE_UI ? "rewrite" : "check-consistency";
            const res = await fetch(`${API_BASE}/${endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    brand_id: selectedBrand.brand_id,
                    text: copyText,
                    n_grounding_chunks: 3
                }),
            });

            const data = await res.json();

            if (!res.ok || data?.error) {
                const rawErr = data?.error || data?.detail?.error || "unknown";
                const friendlyErrors = {
                    profile_missing: "Genome Calibration Required: This brand hasn't been initialized in the engine yet.",
                    text_too_short: "Text processing requires at least 10 words for accurate semantic extraction.",
                };
                setResult({ error: friendlyErrors[rawErr] || rawErr });
                setLoading(false);
                return;
            }

            if (ENABLE_REWRITE_UI) {
                setResult({
                    score_before: data.score_before,
                    score_after: data.score_after,
                    rewritten_text: data.rewritten_text,
                    suggestions: data.suggestions,
                    grounding_chunks_used: data.grounding_chunks_used,
                    diagnostics: data.score_after?.diagnostics || ["Optimal alignment detected across core pillars."],
                    error: null,
                });
            } else {
                setResult({
                    score_before: data,
                    score_after: data,
                    rewritten_text: "",
                    suggestions: [],
                    grounding_chunks_used: [],
                    diagnostics: data.diagnostics || [],
                    error: null,
                });
            }
        } catch (err) {
            console.error(err);
            setResult({ error: "Network anomaly: Connection to engine failed." });
        }

        setLoading(false);
    };

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-7xl mx-auto px-4 sm:px-6">

            <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div className="flex items-center gap-5">
                    <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.3)] text-white border border-indigo-400/30">
                        <Sparkles size={28} />
                    </div>
                    <div>
                        <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">
                            {ENABLE_REWRITE_UI ? "Rewrite Engine" : "Analysis Engine"}
                        </h2>
                        <p className="text-gray-400 mt-2 text-lg">Align and elevate copy perfectly to the brand genome.</p>
                    </div>
                </div>
                <div className="flex flex-col items-start md:items-end gap-2">
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em]">Active Brand Target</span>
                    <BrandSelector selectedId={selectedBrand?.brand_id} onSelect={setSelectedBrand} />
                </div>
            </div>

            <ErrorBanner error={result?.error} />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column: Input and Engine Controls */}
                <div className="lg:col-span-7 flex flex-col gap-6">
                    <Card delay={100} className="flex flex-col relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                            <Sparkles size={120} />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-6 tracking-wide flex items-center gap-3">
                            <BookOpen size={20} className="text-indigo-400" />
                            Source Material
                        </h3>
                        <TextArea
                            label="Raw Non-Compliant Copy"
                            rows={8}
                            value={copyText}
                            onChange={(e) => setCopyText(e.target.value)}
                            className="font-mono text-sm leading-relaxed mb-6 bg-[#09090B]/50"
                        />
                        <Button
                            primary
                            className="w-full text-lg gap-3 py-5 group"
                            onClick={handleAction}
                            disabled={loading || !selectedBrand || copyText.length < 5}
                        >
                            {loading ? <Loader2 className="animate-spin" /> : (
                                <>
                                    <Sparkles size={20} className="group-hover:scale-125 transition-transform" />
                                    {ENABLE_REWRITE_UI ? "Ground & Rewrite" : "Analyze Consistency"}
                                </>
                            )}
                        </Button>

                        {/* RAG Examples Section */}
                        {ENABLE_REWRITE_UI && result && !result.error && result.grounding_chunks_used?.length > 0 && (
                            <div className="mt-8 pt-8 border-t border-white/5">
                                <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <Info size={14} /> Grounding Context (RAG)
                                </h4>
                                <GroundingExamples chunks={result.grounding_chunks_used} />
                            </div>
                        )}
                    </Card>

                    {ENABLE_REWRITE_UI && result && !result.error && (
                        <div className="space-y-6 animate-in slide-in-from-top-4 duration-700">
                            <SuggestionsPanel suggestions={result.suggestions} />
                            <RewritePanel text={result.rewritten_text} />
                        </div>
                    )}
                </div>

                {/* Right Column: Analytics & Scoring */}
                <div className="lg:col-span-5 flex flex-col gap-6">
                    {!result || result.error ? (
                        <Card delay={200} className="flex flex-col items-center justify-center p-16 text-center border-dashed border-white/10 bg-transparent h-full min-h-[500px]">
                            <div className="p-8 rounded-full bg-white/5 mb-8 animate-pulse text-indigo-500/20">
                                <Fingerprint size={64} />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-200">Awaiting Submissions</h3>
                            <p className="text-gray-500 mt-4 max-w-sm text-lg leading-relaxed">
                                {selectedBrand
                                    ? `Engine ready for ${selectedBrand.brand_name}. Paste copy to run deep semantic mapping.`
                                    : "Select a brand target above to activate the engine."
                                }
                            </p>
                            {!selectedBrand && (
                                <div className="mt-8 flex items-center gap-2 text-amber-500/50 bg-amber-500/5 px-4 py-2 rounded-lg border border-amber-500/10 text-xs font-bold uppercase tracking-widest">
                                    <AlertCircle size={14} /> Brand Profile Missing
                                </div>
                            )}
                        </Card>
                    ) : (
                        <Card delay={300} className="h-full flex flex-col">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="text-xl font-bold text-white tracking-wide">Alignment Shift</h3>
                                <div className="px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                                    Live Matrix
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 mb-10 pb-8 border-b border-white/10">
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Original State</p>
                                    <div className="text-4xl font-black text-gray-300">
                                        {result.score_before?.overall_score ?? 0}<span className="text-lg text-gray-600 font-bold ml-1">/100</span>
                                    </div>
                                </div>
                                <div className="p-4 bg-indigo-500/5 rounded-2xl border border-indigo-500/10 text-right">
                                    <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-2">
                                        {ENABLE_REWRITE_UI ? "Post-Rewrite" : "Analysed State"}
                                    </p>
                                    <div className="text-5xl font-black text-white drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]">
                                        {result.score_after?.overall_score ?? 0}<span className="text-lg text-indigo-400 font-bold ml-1">/100</span>
                                    </div>
                                </div>
                            </div>

                            {/* Diagnostics Breakdown */}
                            <div className="mb-10">
                                <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <AlertCircle size={12} /> Diagnostics Breakdown
                                </h4>
                                <div className="space-y-3">
                                    {(result.diagnostics || []).map((diag, i) => (
                                        <div key={i} className="flex gap-3 text-sm text-gray-400 bg-white/5 p-3 rounded-xl border border-white/5 italic">
                                            <div className="mt-1 w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0" />
                                            {diag}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-8 flex-1">
                                <DualMetricBar
                                    label="Tone Resonance"
                                    beforeVal={result.score_before?.tone_pct ?? 0}
                                    afterVal={result.score_after?.tone_pct ?? 0}
                                    baseColor="from-amber-400 to-orange-500"
                                />
                                <DualMetricBar
                                    label="Vocabulary Overlap"
                                    beforeVal={result.score_before?.vocab_overlap_pct ?? 0}
                                    afterVal={result.score_after?.vocab_overlap_pct ?? 0}
                                    baseColor="from-indigo-500 to-blue-500"
                                />
                                <DualMetricBar
                                    label="Sentiment Alignment"
                                    beforeVal={result.score_before?.sentiment_alignment_pct ?? 0}
                                    afterVal={result.score_after?.sentiment_alignment_pct ?? 0}
                                    baseColor="from-emerald-400 to-teal-500"
                                />
                                <DualMetricBar
                                    label="Readability Match"
                                    beforeVal={result.score_before?.readability_match_pct ?? 0}
                                    afterVal={result.score_after?.readability_match_pct ?? 0}
                                    baseColor="from-purple-500 to-pink-500"
                                />
                            </div>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
};
