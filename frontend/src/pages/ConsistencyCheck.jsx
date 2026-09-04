import { useMemo, useState } from 'react';
import { AlertCircle, CheckCircle, Loader2, Sparkles, ShieldAlert, Fingerprint } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { TextArea } from "../components/ui/TextArea";
import { Button } from "../components/ui/Button";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { scoreConsistency } from "../lib/consistencyApi";

const clampScore = (value) => Math.max(0, Math.min(100, Number(value) || 0));

const renderScalar = (value) => {
    if (value === null || value === undefined) return "—";
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "object") {
        const entries = Object.entries(value).filter(([, item]) => item !== null && item !== undefined && item !== "");
        return entries.map(([key, item]) => `${key}: ${Array.isArray(item) ? item.join(", ") : String(item)}`).join(" · ");
    }
    return String(value);
};

const severityClass = (severity) => {
    if (severity === "high") return "text-red-300 border-red-500/20 bg-red-500/10";
    if (severity === "moderate") return "text-amber-300 border-amber-500/20 bg-amber-500/10";
    return "text-sky-300 border-sky-500/20 bg-sky-500/10";
};


export const ConsistencyCheck = ({ profile, onGoToSetup }) => {
    const [copyText, setCopyText] = useState("This timepiece balances precision craftsmanship with calm confidence and enduring elegance.");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [forceGenomeBlock, setForceGenomeBlock] = useState(false);
    const genomeMissing = forceGenomeBlock || !(profile && profile.initialized);

    const featureEntries = useMemo(() => Object.entries(result?.feature_breakdown || {}), [result]);

    const handleAction = async () => {
        if (genomeMissing || !copyText.trim()) return;
        setLoading(true);
        setResult(null);
        setError(null);

        try {
            const data = await scoreConsistency(copyText);
            setResult(data);
        } catch (err) {
            if (err.code === "genome_not_initialized") {
                setForceGenomeBlock(true);
                setError("Genome Setup Required: Initialize the active genome before scoring copy.");
            } else {
                setError(err.message || "Unable to score copy.");
            }
            setResult(null);
        }

        setLoading(false);
    };

    if (genomeMissing) {
        return (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-6xl mx-auto px-4 sm:px-6">
                <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="flex items-center gap-5">
                        <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-cyan-400 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.2)] text-white border border-indigo-400/30">
                            <Sparkles size={28} />
                        </div>
                        <div>
                            <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Consistency Check</h2>
                            <p className="text-gray-400 mt-2 text-lg">Score copy against the active persisted genome.</p>
                        </div>
                    </div>
                </div>

                <Card className="border-dashed border-amber-500/20 bg-amber-500/5">
                    <div className="flex items-start gap-4">
                        <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-300">
                            <ShieldAlert size={24} />
                        </div>
                        <div className="flex-1">
                            <h3 className="text-xl font-bold text-white mb-2">Genome Setup Required</h3>
                            <p className="text-gray-400 max-w-2xl">
                                Consistency scoring is blocked until the active user genome has been initialized. Set up the genome first, then return here to evaluate copy against the persisted user profile.
                            </p>
                            <div className="mt-6 flex flex-wrap gap-3">
                                <Button primary onClick={onGoToSetup}>Go to Genome Setup</Button>
                            </div>
                        </div>
                    </div>
                </Card>
            </div>
        );
    }

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-7xl mx-auto px-4 sm:px-6">

            <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div className="flex items-center gap-5">
                    <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.3)] text-white border border-indigo-400/30">
                        <Sparkles size={28} />
                    </div>
                    <div>
                        <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Consistency Check</h2>
                        <p className="text-gray-400 mt-2 text-lg">Score copy against the active persisted genome.</p>
                    </div>
                </div>
                <div className="flex flex-col items-start md:items-end gap-2 text-right">
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em]">Active Genome</span>
                    <span className="text-sm font-semibold text-gray-200">{profile?.designation || profile?.brand_name || "Initialized genome"}</span>
                </div>
            </div>

            <ErrorBanner error={error} />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-7 flex flex-col gap-6">
                    <Card delay={100} className="flex flex-col relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                            <Sparkles size={120} />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-6 tracking-wide flex items-center gap-3">
                            <Fingerprint size={20} className="text-indigo-400" />
                            Copy To Evaluate
                        </h3>
                        <TextArea
                            label="Input Copy"
                            rows={8}
                            value={copyText}
                            onChange={(e) => setCopyText(e.target.value)}
                            className="font-mono text-sm leading-relaxed mb-6 bg-[#09090B]/50"
                        />
                        <Button
                            primary
                            className="w-full text-lg gap-3 py-5 group"
                            onClick={handleAction}
                            disabled={loading || genomeMissing || !copyText.trim()}
                        >
                            {loading ? <Loader2 className="animate-spin" /> : (
                                <>
                                    <Sparkles size={20} className="group-hover:scale-125 transition-transform" />
                                    Check Consistency
                                </>
                            )}
                        </Button>
                    </Card>

                </div>

                <div className="lg:col-span-5 flex flex-col gap-6">
                    {!result ? (
                        <Card delay={200} className="flex flex-col items-center justify-center p-16 text-center border-dashed border-white/10 bg-transparent h-full min-h-[500px]">
                            <div className="p-8 rounded-full bg-white/5 mb-8 animate-pulse text-indigo-500/20">
                                <Fingerprint size={64} />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-200">Awaiting Submission</h3>
                            <p className="text-gray-500 mt-4 max-w-sm text-lg leading-relaxed">
                                Paste copy to score against the persisted genome. The result will show feature-level drift and diagnostics.
                            </p>
                        </Card>
                    ) : (
                        <Card delay={300} className="h-full flex flex-col">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="text-xl font-bold text-white tracking-wide">Overall Consistency</h3>
                                <div className="px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                                    Live Score
                                </div>
                            </div>

                            <div className="grid grid-cols-1 gap-4 mb-10 pb-8 border-b border-white/10">
                                <div className="p-5 bg-indigo-500/5 rounded-2xl border border-indigo-500/10">
                                    <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-2">Overall Score</p>
                                    <div className="flex items-end gap-3">
                                        <div className="text-6xl font-black text-white drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]">
                                            {clampScore(result.score_overall)}
                                        </div>
                                        <div className="text-lg text-indigo-300 font-bold pb-2">/100</div>
                                    </div>
                                </div>
                            </div>

                            <div className="mb-10">
                                <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <CheckCircle size={12} /> Brand Mentions
                                </h4>
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-sm text-gray-300">
                                    <span className="font-bold text-white">{result.brand_name_mentions?.count ?? 0}</span> neutral mentions of <span className="font-semibold text-indigo-300">{result.brand_name_mentions?.designation || profile?.designation || profile?.brand_name || "the active genome"}</span>
                                </div>
                            </div>

                            <div className="space-y-8 flex-1">
                                <div>
                                    <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                        <AlertCircle size={12} /> Feature Breakdown
                                    </h4>
                                    <div className="space-y-3">
                                        {featureEntries.map(([name, item]) => (
                                            <div key={name} className="bg-white/5 p-4 rounded-2xl border border-white/5">
                                                <div className="flex items-start justify-between gap-4 mb-3">
                                                    <div>
                                                        <div className="text-sm font-bold text-white capitalize tracking-wide">{name.replace(/_/g, " ")}</div>
                                                        <div className="text-xs text-gray-500 mt-1">{renderScalar(item.details?.matched_keywords || item.input_value)}</div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className="text-xl font-black text-white">{clampScore(item.score)}</div>
                                                        <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">/100</div>
                                                    </div>
                                                </div>
                                                <div className="h-2 rounded-full bg-[#1A1A24] overflow-hidden border border-white/5">
                                                    <div className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400" style={{ width: `${clampScore(item.score)}%` }} />
                                                </div>
                                                <div className="mt-3 text-xs text-gray-400 space-y-1">
                                                    <div>Input: {renderScalar(item.input_value)}</div>
                                                    <div>Target: {renderScalar(item.target_value)}</div>
                                                    <div>Delta: {renderScalar(item.delta)}</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                        <AlertCircle size={12} /> Diagnostics
                                    </h4>
                                    <div className="space-y-3">
                                        {(result.diagnostic_breakdown || []).length > 0 ? (
                                            result.diagnostic_breakdown.map((diag, index) => (
                                                <div key={`${diag.dimension}-${index}`} className={`flex flex-col gap-2 text-sm p-4 rounded-2xl border ${severityClass(diag.severity)}`}>
                                                    <div className="flex items-center justify-between gap-3">
                                                        <span className="font-bold uppercase tracking-widest text-[10px]">{diag.dimension.replace(/_/g, " ")}</span>
                                                        <span className="text-[10px] font-black uppercase tracking-widest">{diag.severity}</span>
                                                    </div>
                                                    <div className="text-gray-100">{diag.message}</div>
                                                    <div className="text-xs text-gray-300">Suggestion: {diag.suggestion}</div>
                                                </div>
                                            ))
                                        ) : (
                                            <div className="flex gap-3 text-sm text-emerald-300 bg-emerald-500/10 p-4 rounded-2xl border border-emerald-500/20">
                                                <div className="mt-1 w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                                                The copy is closely aligned with the active genome across the scored dimensions.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
};
