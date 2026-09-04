import { useMemo, useState } from 'react';
import {
    AlertCircle, ArrowRight, BookOpen, ChevronDown, ChevronUp, Loader2, ShieldAlert, Sparkles, Wand2,
} from 'lucide-react';
import { Card } from '../components/ui/Card';
import { TextArea } from '../components/ui/TextArea';
import { Button } from '../components/ui/Button';
import { ErrorBanner } from '../components/ui/ErrorBanner';
import { runRewrite } from '../lib/rewriteApi';

const clampScore = (value) => Math.max(0, Math.min(100, Number(value) || 0));

const severityClass = (severity) => {
    if (severity === 'high') return 'text-red-300 border-red-500/20 bg-red-500/10';
    if (severity === 'moderate') return 'text-amber-300 border-amber-500/20 bg-amber-500/10';
    return 'text-sky-300 border-sky-500/20 bg-sky-500/10';
};

const PRECONDITION_MESSAGES = {
    genome_not_initialized: {
        title: 'Genome Setup Required',
        message: 'Rewrite is blocked until the active user genome has been initialized.',
        action: 'setup_genome',
    },
    user_genome_chunks_missing: {
        title: 'Genome Corpus Missing',
        message: 'The active user genome has no chunked source text yet. Re-run genome initialization.',
    },
    index_missing: {
        title: 'RAG Index Missing',
        message: 'The semantic RAG index has not been built yet. Rebuild required (Stage 7).',
    },
    index_stale: {
        title: 'RAG Index Stale',
        message: 'The RAG index no longer matches the current genome corpus. Rebuild required (Stage 7).',
    },
    user_grounding_not_indexed: {
        title: 'User Grounding Not Indexed',
        message: 'The active user genome is not present in the RAG index. Rebuild required (Stage 7).',
    },
    rewrite_provider_unavailable: {
        title: 'Rewrite Provider Unavailable',
        message: 'The OpenAI rewrite provider is not configured or unreachable.',
    },
    rewrite_provider_rate_limited: {
        title: 'Rewrite Provider Rate-Limited',
        message: 'The OpenAI rewrite provider is rate-limited. Try again shortly.',
    },
    rewrite_provider_error: {
        title: 'Rewrite Provider Error',
        message: 'The OpenAI rewrite provider returned an error.',
    },
    rewrite_provider_invalid_response: {
        title: 'Rewrite Provider Invalid Response',
        message: 'The rewrite provider returned an empty or invalid response.',
    },
};

const GroundingSnippets = ({ chunks }) => {
    const [isOpen, setIsOpen] = useState(false);
    if (!chunks || chunks.length === 0) return null;

    return (
        <div className="bg-black/20 border border-white/5 rounded-xl overflow-hidden mt-2">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <BookOpen className="text-gray-400" size={18} />
                    <span className="text-sm font-semibold text-gray-200 tracking-wide uppercase">
                        Grounding Snippets Used ({chunks.length})
                    </span>
                </div>
                {isOpen ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
            </button>

            {isOpen && (
                <div className="p-4 bg-black/40 border-t border-white/5 space-y-4">
                    {chunks.map((chunk) => (
                        <div key={chunk.chunk_id} className="border-l-2 border-indigo-500/30 pl-3 py-1">
                            <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-gray-500 mb-1">
                                <span>Rank {chunk.rank}</span>
                                <span>·</span>
                                <span>{chunk.source_type}</span>
                                <span>·</span>
                                <span>similarity {Number(chunk.score).toFixed(3)}</span>
                            </div>
                            <p className="text-sm text-gray-300 leading-relaxed">&ldquo;{chunk.chunk_text}&rdquo;</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export const Rewrite = ({ profile, onGoToSetup }) => {
    const [copyText, setCopyText] = useState('This watch is super cool and awesome, and pretty nice to wear every day.');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [errorCode, setErrorCode] = useState(null);
    const [forceGenomeBlock, setForceGenomeBlock] = useState(false);
    const genomeMissing = forceGenomeBlock || !(profile && profile.initialized);

    const featureEntries = useMemo(() => {
        const before = result?.feature_breakdown_before || {};
        const after = result?.feature_breakdown_after || {};
        return Object.keys(before).map((name) => ({ name, before: before[name], after: after[name] }));
    }, [result]);

    const handleRewrite = async () => {
        if (genomeMissing || !copyText.trim()) return;
        setLoading(true);
        setResult(null);
        setError(null);
        setErrorCode(null);

        try {
            const data = await runRewrite({ text: copyText });
            setResult(data);
        } catch (err) {
            if (err.code === 'genome_not_initialized') {
                setForceGenomeBlock(true);
            }
            setErrorCode(err.code || null);
            setError(PRECONDITION_MESSAGES[err.code]?.message || err.message || 'Rewrite request failed.');
        }

        setLoading(false);
    };

    if (genomeMissing) {
        return (
            <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-6xl mx-auto px-4 sm:px-6">
                <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div className="flex items-center gap-5">
                        <div className="p-3.5 bg-gradient-to-br from-fuchsia-500 to-indigo-500 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.2)] text-white border border-indigo-400/30">
                            <Wand2 size={28} />
                        </div>
                        <div>
                            <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Rewrite</h2>
                            <p className="text-gray-400 mt-2 text-lg">Grounded, brand-aligned OpenAI rewrite of your copy.</p>
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
                                Rewrite is blocked until the active user genome has been initialized. Set up the genome first,
                                then return here to rewrite copy grounded in your own brand voice.
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
                    <div className="p-3.5 bg-gradient-to-br from-fuchsia-500 to-indigo-500 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.3)] text-white border border-indigo-400/30">
                        <Wand2 size={28} />
                    </div>
                    <div>
                        <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Rewrite</h2>
                        <p className="text-gray-400 mt-2 text-lg">Grounded, brand-aligned OpenAI rewrite of your copy.</p>
                    </div>
                </div>
                <div className="flex flex-col items-start md:items-end gap-2 text-right">
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em]">Active Genome</span>
                    <span className="text-sm font-semibold text-gray-200">{profile?.designation || profile?.brand_name || 'Initialized genome'}</span>
                </div>
            </div>

            {errorCode && PRECONDITION_MESSAGES[errorCode] && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 mb-6 flex items-start gap-4">
                    <ShieldAlert className="text-amber-400 mt-0.5 shrink-0" size={20} />
                    <div>
                        <h4 className="text-amber-300 font-semibold mb-1">{PRECONDITION_MESSAGES[errorCode].title}</h4>
                        <p className="text-amber-200/80 text-sm leading-relaxed">{PRECONDITION_MESSAGES[errorCode].message}</p>
                    </div>
                </div>
            )}
            {(!errorCode || !PRECONDITION_MESSAGES[errorCode]) && <ErrorBanner error={error} />}

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-7 flex flex-col gap-6">
                    <Card delay={100} className="flex flex-col relative overflow-hidden">
                        <h3 className="text-xl font-bold text-white mb-6 tracking-wide flex items-center gap-3">
                            <Sparkles size={20} className="text-indigo-400" />
                            Copy To Rewrite
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
                            onClick={handleRewrite}
                            disabled={loading || genomeMissing || !copyText.trim()}
                        >
                            {loading ? <Loader2 className="animate-spin" /> : (
                                <>
                                    <Wand2 size={20} className="group-hover:scale-125 transition-transform" />
                                    Rewrite / Improve Copy
                                </>
                            )}
                        </Button>
                    </Card>

                    {result && (
                        <Card delay={150}>
                            <h3 className="text-xl font-bold text-white mb-6 tracking-wide">Before / After</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Original</p>
                                    <p className="text-sm text-gray-300 leading-relaxed">{result.original_text}</p>
                                </div>
                                <div className="p-4 bg-indigo-500/5 rounded-2xl border border-indigo-500/10">
                                    <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-2">Rewritten</p>
                                    <p className="text-sm text-indigo-50 leading-relaxed">{result.rewritten_text}</p>
                                </div>
                            </div>
                            <GroundingSnippets chunks={result.grounding_chunks} />
                        </Card>
                    )}

                    {result?.edit_plan && (
                        <Card delay={200}>
                            <h3 className="text-xl font-bold text-white mb-4 tracking-wide">Edit Plan</h3>
                            <div className="space-y-2 text-sm text-gray-300">
                                <div><span className="text-gray-500 font-bold uppercase tracking-widest text-[10px] mr-2">Goals</span>{(result.edit_plan.goals || []).join(', ') || '—'}</div>
                                <div><span className="text-gray-500 font-bold uppercase tracking-widest text-[10px] mr-2">Tone</span>{result.edit_plan.tone_direction || '—'}</div>
                                <div><span className="text-gray-500 font-bold uppercase tracking-widest text-[10px] mr-2">Style Rules</span>{(result.edit_plan.style_rules || []).join(', ') || '—'}</div>
                                <div><span className="text-gray-500 font-bold uppercase tracking-widest text-[10px] mr-2">Prefer Terms</span>{(result.edit_plan.prefer_terms || []).join(', ') || '—'}</div>
                            </div>
                        </Card>
                    )}
                </div>

                <div className="lg:col-span-5 flex flex-col gap-6">
                    {!result ? (
                        <Card delay={200} className="flex flex-col items-center justify-center p-16 text-center border-dashed border-white/10 bg-transparent h-full min-h-[500px]">
                            <div className="p-8 rounded-full bg-white/5 mb-8 animate-pulse text-indigo-500/20">
                                <Wand2 size={64} />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-200">Awaiting Submission</h3>
                            <p className="text-gray-500 mt-4 max-w-sm text-lg leading-relaxed">
                                Paste copy to rewrite it against the persisted genome, grounded in your own retrieved brand snippets.
                            </p>
                        </Card>
                    ) : (
                        <Card delay={300} className="h-full flex flex-col">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="text-xl font-bold text-white tracking-wide">Score Change</h3>
                                <div className="px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                                    {result.provider?.name === 'openai' ? `OpenAI · ${result.provider?.model}` : `Fallback · ${result.provider?.model}`}
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-3 mb-10 pb-8 border-b border-white/10">
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-center">
                                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Before</p>
                                    <div className="text-3xl font-black text-white">{clampScore(result.score_before)}</div>
                                </div>
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/5 text-center">
                                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">After</p>
                                    <div className="text-3xl font-black text-white">{clampScore(result.score_after)}</div>
                                </div>
                                <div className={`p-4 rounded-2xl border text-center ${result.score_delta >= 0 ? 'bg-emerald-500/5 border-emerald-500/10' : 'bg-red-500/5 border-red-500/10'}`}>
                                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Change</p>
                                    <div className={`text-3xl font-black flex items-center justify-center gap-1 ${result.score_delta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {result.score_delta >= 0 ? '+' : ''}{result.score_delta}
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-3 flex-1">
                                <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <AlertCircle size={12} /> Diagnostics / Drift (Before Rewrite)
                                </h4>
                                {(result.drift_report || []).length > 0 ? (
                                    result.drift_report.map((diag, index) => (
                                        <div key={`${diag.dimension}-${index}`} className={`flex flex-col gap-2 text-sm p-4 rounded-2xl border ${severityClass(diag.severity)}`}>
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="font-bold uppercase tracking-widest text-[10px]">{diag.dimension.replace(/_/g, ' ')}</span>
                                                <span className="text-[10px] font-black uppercase tracking-widest">{diag.severity}</span>
                                            </div>
                                            <div className="text-gray-100">{diag.message}</div>
                                            <div className="text-xs text-gray-300">Suggestion: {diag.suggestion}</div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="flex gap-3 text-sm text-emerald-300 bg-emerald-500/10 p-4 rounded-2xl border border-emerald-500/20">
                                        <div className="mt-1 w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                                        No meaningful drift detected before rewrite.
                                    </div>
                                )}
                            </div>

                            {featureEntries.length > 0 && (
                                <div className="mt-8">
                                    <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4">Feature Scores (Before → After)</h4>
                                    <div className="space-y-2">
                                        {featureEntries.map(({ name, before, after }) => (
                                            <div key={name} className="flex items-center justify-between text-sm bg-white/5 p-3 rounded-xl border border-white/5">
                                                <span className="capitalize text-gray-300">{name.replace(/_/g, ' ')}</span>
                                                <span className="font-bold text-white flex items-center gap-2">
                                                    {clampScore(before?.score)} <ArrowRight size={12} className="text-gray-500" /> {clampScore(after?.score)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
};
