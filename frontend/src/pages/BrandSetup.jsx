import { useState, useEffect } from 'react';
import { Settings, Sparkles, Award, Loader2, ArrowRight, BookOpen, Fingerprint } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { TextArea } from "../components/ui/TextArea";
import { Button } from "../components/ui/Button";
import { Metric } from "../components/ui/Metric";
import { DNAAnimation } from "../components/ui/DNAAnimation";
import { API_BASE } from "../lib/constants";

export const BrandSetup = ({ profile, fetchProfile }) => {
    const defaultForm = { 
        brand_name: '', 
        mission: '', 
        tone: 'Sophisticated',
        snippets: ['', '', '', '', '', '', '']
    };
    const [form, setForm] = useState(defaultForm);
    const [loading, setLoading] = useState(false);
    const [initializing, setInitializing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [rebuilding, setRebuilding] = useState(false);
    const [rebuildStatus, setRebuildStatus] = useState(null);

    // Synchronize form with profile when profile is loaded/updated
    useEffect(() => {
        if (profile && profile.brand_id === 'user_brand') {
             console.log("Populating form from server profile:", profile);
             setForm({ 
                brand_name: profile.brand_name || profile.name || '', 
                mission: profile.mission || '', 
                tone: profile.tone_label || 'Sophisticated',
                snippets: profile.snippets || ['', '', '', '', '', '', '']
            });
        }
    }, [profile]);

    // save draft to localStorage ONLY if user has started typing
    useEffect(() => {
        const isDefault = JSON.stringify(form) === JSON.stringify(defaultForm);
        if (!initializing && !isDefault) {
            localStorage.setItem('genome_draft', JSON.stringify(form));
        }
    }, [form, initializing]);

    const handleRebuild = async () => {
        setRebuilding(true);
        setRebuildStatus("Rebuilding Vector Index...");
        try {
            const res = await fetch(`${API_BASE}/index/rebuild`, { method: 'POST' });
            if (res.ok) {
                setRebuildStatus("Success: Index Rebuilt");
            } else {
                setRebuildStatus("Failed to rebuild index");
            }
        } catch (err) {
            setRebuildStatus("Error: Network anomaly");
        }
        setRebuilding(false);
        setTimeout(() => setRebuildStatus(null), 3500);
    };

    const handleSave = async () => {
        // Validation
        if (!form.brand_name || !form.mission) {
            alert("Please provide a brand name and mission.");
            return;
        }
        if (form.snippets.some(s => s.trim().length < 10)) {
            alert("Please provide all 7 snippets (min 10 characters each) for accurate genome calibration.");
            return;
        }

        setInitializing(true);
        setLoading(true);
        
        // Progress simulation for "Initialising..." overlay
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 90) return prev;
                return prev + Math.random() * 15;
            });
        }, 600);

        try {
            const res = await fetch(`${API_BASE}/profile`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(form)
            });
            
            if (res.ok) {
                // Keep the overlay for a bit longer to look premium
                setProgress(100);
                setTimeout(async () => {
                    await fetchProfile();
                    setInitializing(false);
                    localStorage.removeItem('genome_draft');
                }, 1000);
            } else {
                alert("Failed to update profile.");
                setInitializing(false);
            }
        } catch (err) {
            console.error(err);
            setInitializing(false);
        } finally {
            clearInterval(interval);
            setLoading(false);
        }
    };

    const handleSnippetChange = (index, value) => {
        const newSnippets = [...form.snippets];
        newSnippets[index] = value;
        setForm({ ...form, snippets: newSnippets });
    };

    if (!profile && !initializing) return <div className="flex justify-center p-20"><Loader2 className="animate-spin text-indigo-500" size={48} /></div>;

    return (
        <div className="relative animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both">
            
            {/* Initialization Overlay */}
            {initializing && (
                <div className="fixed inset-0 z-[100] bg-[#09090B]/95 backdrop-blur-xl flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-500">
                    <div className="w-full max-w-md">
                        <div className="relative mb-12 flex justify-center">
                            <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full scale-150 animate-pulse"></div>
                            <DNAAnimation className="scale-125" />
                        </div>
                        <h2 className="text-3xl font-black text-white mb-4 tracking-tight">Initialising Genome...</h2>
                        <p className="text-gray-400 mb-8 max-w-sm mx-auto">
                            Extracting linguistic patterns and calibrating vector proximity across 384 dimensions.
                        </p>
                        
                        <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden mb-4 border border-white/10">
                            <div 
                                className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-500 ease-out shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <div className="flex justify-between text-[10px] font-bold uppercase tracking-[0.2em] text-gray-500">
                            <span>Vector Mapping</span>
                            <span>{Math.round(progress)}% Complete</span>
                        </div>
                    </div>
                </div>
            )}

            <div className="mb-12 flex items-center gap-5">
                <div className="p-3.5 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.3)] text-white border border-indigo-400/30">
                    <Settings size={28} />
                </div>
                <div>
                    <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Genome Setup</h2>
                    <p className="text-gray-400 mt-2 text-lg">Calibrate the core identity and linguistic parameters of your brand.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
                <Card delay={100} className="lg:row-span-2">
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-3 text-indigo-400">
                            <Sparkles size={20} />
                            <h3 className="text-xl font-bold text-white tracking-wide">Identity Matrix</h3>
                        </div>
                        {localStorage.getItem('genome_draft') && (
                            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-emerald-500/10 px-2 py-1 rounded border border-emerald-500/20">
                                Draft Saved
                            </span>
                        )}
                    </div>

                    <Input label="Brand Designation" placeholder="e.g. Rolex, Omega..." value={form.brand_name} onChange={(e) => setForm({ ...form, brand_name: e.target.value })} />
                    <TextArea label="Mission Statement" rows={3} placeholder="The core purpose and vision of the brand..." value={form.mission} onChange={(e) => setForm({ ...form, mission: e.target.value })} />

                    <div className="mb-10 group w-full">
                        <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-3 transition-colors group-focus-within:text-indigo-400">
                            Primary Tone Identifier
                        </label>
                        <div className="grid grid-cols-3 gap-3">
                            {['Sophisticated', 'Adventurous', 'Technical'].map(t => (
                                <button
                                    key={t}
                                    onClick={() => setForm({ ...form, tone: t })}
                                    className={`px-4 py-3 rounded-xl border text-xs font-bold transition-all ${
                                        form.tone === t 
                                        ? 'bg-indigo-500 border-indigo-400 text-white shadow-[0_0_15px_rgba(99,102,241,0.4)]' 
                                        : 'bg-[#09090B] border-white/10 text-gray-500 hover:border-white/20'
                                    }`}
                                >
                                    {t}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="flex items-center gap-3 text-violet-400 mb-2">
                            <BookOpen size={18} />
                            <h4 className="text-sm font-bold text-white uppercase tracking-widest">Grounding Snippets (7 Required)</h4>
                        </div>
                        <p className="text-xs text-gray-500 mb-6 leading-relaxed">
                            Provide 7 authentic snippets of brand copy (captions, website blurbs, ads) to map the DNA.
                        </p>
                        
                        <div className="space-y-4">
                            {form.snippets.map((s, i) => (
                                <div key={i} className="relative group">
                                    <div className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-[#111116] border border-white/10 flex items-center justify-center text-[10px] font-bold text-gray-500 group-focus-within:border-indigo-500 group-focus-within:text-indigo-400 transition-colors shadow-lg z-10">
                                        {i + 1}
                                    </div>
                                    <textarea
                                        value={s}
                                        onChange={(e) => handleSnippetChange(i, e.target.value)}
                                        placeholder={`Enter snippet ${i + 1}...`}
                                        className="w-full bg-[#09090B] border border-white/10 rounded-xl px-6 py-4 text-sm text-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all min-h-[80px] resize-none"
                                    />
                                </div>
                            ))}
                        </div>
                    </div>

                    <Button primary className="w-full text-lg mt-10 py-5 group" onClick={handleSave} disabled={loading}>
                        {loading ? <Loader2 className="animate-spin" /> : <>Compile Brand Genome <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" /></>}
                    </Button>
                </Card>

                <div className="space-y-8">
                    <Metric label="System Status" value="Online" trend="Calibrated" delay={200} />

                    <Card delay={300} className="relative overflow-hidden min-h-[360px] flex flex-col">
                        <div className="absolute inset-0 z-0 pointer-events-none opacity-50">
                            <DNAAnimation />
                        </div>
                        <div className="relative z-10 flex-1 flex flex-col">
                            <h3 className="text-xl font-bold text-white mb-8 flex items-center gap-3">
                                <Fingerprint className="text-violet-400" size={24} />
                                Active Blueprint
                            </h3>

                            <div className="space-y-8 flex-1">
                                <div>
                                    <span className="block text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Core Identifier</span>
                                    <div className="text-2xl font-black text-white tracking-tight">
                                        {profile.name || "Awaiting Definition"}
                                    </div>
                                </div>

                                <div>
                                    <span className="block text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Semantic Anchors</span>
                                    <div className="flex flex-wrap gap-2.5">
                                        {(profile.top_keywords || []).map((t, i) => (
                                            <span
                                                key={t}
                                                className="px-4 py-1.5 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded-full text-[11px] font-bold tracking-wide transition-transform hover:-translate-y-1 cursor-default shadow-sm"
                                            >
                                                {t}
                                            </span>
                                        ))}
                                        {(!profile.top_keywords || profile.top_keywords.length === 0) && <span className="text-gray-500 italic text-sm">Genome pending...</span>}
                                    </div>
                                </div>

                                <div className="pt-8 border-t border-white/5 mt-auto">
                                    <div className="flex items-center gap-6">
                                        <div>
                                            <span className="block text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-1">Snippets</span>
                                            <span className="text-xl font-black text-white">{profile.snippetsCount || 0}/7</span>
                                        </div>
                                        <div className="w-px h-10 bg-white/10" />
                                        <div>
                                            <span className="block text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-1">Tone Base</span>
                                            <span className="text-lg font-bold text-indigo-400">{profile.tone_label || 'Unset'}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card delay={400} className="flex flex-col border-indigo-500/10">
                        <div className="flex items-center gap-3 mb-4">
                            <Settings size={18} className="text-gray-400" />
                            <h3 className="text-sm font-bold text-white uppercase tracking-widest">System Controls</h3>
                        </div>
                        <p className="text-xs text-gray-500 mb-6">Manually trigger vector index rebuild and align system cache.</p>
                        
                        <Button 
                            className={`w-full py-3 text-sm font-bold transition-all ${rebuilding ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' : 'bg-[#111116] border border-white/10 text-gray-300 hover:border-white/30'}`}
                            onClick={handleRebuild} 
                            disabled={rebuilding}
                        >
                            {rebuilding ? <Loader2 className="animate-spin inline mr-2" size={16} /> : null}
                            {rebuilding ? "Rebuilding Index..." : "Force Index Rebuild"}
                        </Button>
                        
                        {rebuildStatus && (
                            <div className={`mt-4 text-xs font-bold px-4 py-3 rounded-xl border ${
                                rebuildStatus.includes('Error') || rebuildStatus.includes('Failed') 
                                    ? 'bg-red-500/10 border-red-500/20 text-red-400' 
                                    : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                            }`}>
                                {rebuildStatus}
                            </div>
                        )}
                    </Card>
                </div>
            </div>
        </div>
    );
};
