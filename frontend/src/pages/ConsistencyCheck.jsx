import { useState } from 'react';
import { CheckCircle, Loader2, Sparkles } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { TextArea } from "../components/ui/TextArea";
import { Button } from "../components/ui/Button";
import { BrandSelector } from "../components/ui/BrandSelector";
import { ErrorBanner } from "../components/ui/ErrorBanner";
import { SuggestionsPanel } from "../components/ui/SuggestionsPanel";
import { RewritePanel } from "../components/ui/RewritePanel";
import { GroundingExamples } from "../components/ui/GroundingExamples";
import { API_BASE } from "../lib/constants";

const ENABLE_REWRITE_UI = false; // flip to true later

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

    const handleRewrite = async () => {
  if (!selectedBrand || !copyText) return;
  setLoading(true);
  setResult(null);

  try {
    const res = await fetch(`${API_BASE}/check-consistency`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        brand_id: selectedBrand.brand_id,
        text: copyText,
      }),
    });

    const data = await res.json();

    if (!res.ok || data?.error) {
      setResult({ error: data?.error || "Failed to score text." });
      setLoading(false);
      return;
    }

    // Shape into the old rewrite response so the existing UI doesn't crash
    setResult({
      score_before: data,
      score_after: data, // same for now (no rewrite in demo)
      rewritten_text: "",
      suggestions: [],
      grounding_chunks_used: [],
      error: null,
    });
  } catch (err) {
    console.error(err);
    setResult({ error: "Network error connecting to the engine." });
  }

  setLoading(false);
};

   return (
       <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-6xl mx-auto">

           <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-6">
               <div className="flex items-center gap-5">
                   <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.3)] text-white border border-indigo-400/30">
                       <Sparkles size={28} />
                   </div>
                    <div>
                        <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Analysis Engine</h2> {/*Later change to Rewrite Engine*/}
                        <p className="text-gray-400 mt-2 text-lg">Align and elevate copy perfectly to the brand genome.</p>
                    </div>
                </div>
                <div className="flex flex-col items-start md:items-end gap-2">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">Target Brand</span>
                    <BrandSelector selectedId={selectedBrand?.brand_id} onSelect={setSelectedBrand} />
                </div>
            </div>

            <ErrorBanner error={result?.error} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
                {/* Left Column: Input and Suggestions/Rewrite */}
                <div className="flex flex-col gap-6">
                    <Card delay={100} className="flex flex-col">
                        <h3 className="text-xl font-bold text-white mb-6 tracking-wide">Source Material</h3>
                        <TextArea
                            label="Paste Off-Brand Copy"
                            rows={6}
                            value={copyText}
                            onChange={(e) => setCopyText(e.target.value)}
                            className="font-mono text-sm leading-relaxed mb-6"
                        />
                        <Button primary className="w-full text-lg gap-3 py-4" onClick={handleRewrite} disabled={loading || !selectedBrand || copyText.length < 5}>
                            {loading ? <Loader2 className="animate-spin" /> : <><Sparkles size={20} /> Analyze Consistency </>} {/* Later change to Ground & Rewrite */}
                        </Button>

                        {/* RAG Examples Accordion */}
                        {ENABLE_REWRITE_UI && result && !result.error && (
                            <GroundingExamples chunks={result.grounding_chunks_used} />
                        )}
                    </Card>

                    {ENABLE_REWRITE_UI && result && !result.error && (
                        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <SuggestionsPanel suggestions={result.suggestions} />
                            <RewritePanel text={result.rewritten_text} />
                        </div>
                    )}
                </div>

                {/* Right Column: Scoring */}
                <div>
                    {!result || result.error ? (
                        <Card delay={200} className="flex flex-col items-center justify-center p-16 text-center border-dashed border-white/10 bg-transparent h-full min-h-[400px]">
                            <div className="p-6 rounded-full bg-white/5 mb-6">
                                <CheckCircle size={48} className="text-indigo-500/50" />
                            </div>
                            <h3 className="text-xl font-bold text-gray-300">Awaiting Submissions</h3>
                            <p className="text-gray-500 mt-4 max-w-sm text-lg leading-relaxed">Enter copy on the left to perform a deep semantic analysis and automatically rewrite it to standard.</p>
                        </Card>
                    ) : (
                        <Card delay={300} className="h-full">
                            <h3 className="text-xl font-bold text-white mb-8 tracking-wide">Alignment Shift</h3>

                            <div className="grid grid-cols-2 gap-4 mb-10 pb-8 border-b border-white/10">
                                <div>
                                    <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Before Score</p>
                                    <div className="text-4xl font-black text-gray-300">
                                        {result.score_before?.overall_score}<span className="text-xl text-gray-600">/100</span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-1">Analysis Score</p> {/*Later change to After Rewrite*/}
                                    <div className="text-5xl font-black text-white drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]">
                                        {result.score_after?.overall_score}<span className="text-xl text-indigo-400">/100</span>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-8">
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
