import { useState, useEffect } from 'react';
import { Settings, Sparkles, Award, Loader2, ArrowRight } from 'lucide-react';
import { Card } from "../components/ui/Card";
import { Input } from "../components/ui/Input";
import { TextArea } from "../components/ui/TextArea";
import { Button } from "../components/ui/Button";
import { Metric } from "../components/ui/Metric";
import { DNAAnimation } from "../components/ui/DNAAnimation";
import { API_BASE } from "../lib/constants";

export const BrandSetup = ({ profile, fetchProfile }) => {
    const defaultForm = { brand_name: '', mission: '', tone: 'Sophisticated' };
    const [form, setForm] = useState(defaultForm);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (profile) {
            setForm({ brand_name: profile.name, mission: profile.mission, tone: profile.tone });
        }
    }, [profile]);

    const handleSave = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/profile`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(form)
            });
            if (res.ok) {
                await fetchProfile();
            } else {
                alert("Failed to update profile.");
            }
        } catch (err) {
            console.error(err);
        }
        setLoading(false);
    };

    if (!profile) return <div className="flex justify-center p-20"><Loader2 className="animate-spin text-indigo-500" size={48} /></div>;

    return (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both">
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
                <Card delay={100}>
                    <div className="flex items-center gap-3 mb-8 text-indigo-400">
                        <Sparkles size={20} />
                        <h3 className="text-xl font-bold text-white tracking-wide">Configuration Matrix</h3>
                    </div>

                    <Input label="Brand Designation" value={form.brand_name} onChange={(e) => setForm({ ...form, brand_name: e.target.value })} />
                    <TextArea label="Mission / Core Vision" rows={5} value={form.mission} onChange={(e) => setForm({ ...form, mission: e.target.value })} />

                    <div className="mb-8 group w-full">
                        <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2 transition-colors group-focus-within:text-indigo-400">
                            Primary Tone Identifier
                        </label>
                        <div className="relative">
                            <select
                                value={form.tone}
                                onChange={(e) => setForm({ ...form, tone: e.target.value })}
                                className="w-full px-4 py-3.5 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all duration-300 text-gray-100 appearance-none cursor-pointer shadow-inner"
                            >
                                <option className="bg-[#111116] text-white">Sophisticated</option>
                                <option className="bg-[#111116] text-white">Adventurous</option>
                                <option className="bg-[#111116] text-white">Technical</option>
                            </select>
                            <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                            </div>
                        </div>
                    </div>

                    <Button primary className="w-full text-lg mt-4 py-4" onClick={handleSave} disabled={loading}>
                        {loading ? <Loader2 className="animate-spin" /> : <>Initialize Profile <ArrowRight size={18} /></>}
                    </Button>
                </Card>

                <div className="space-y-8">
                    <Metric label="System Status" value="Online" trend="Calibrated" delay={200} />

                    <Card delay={300} className="relative overflow-hidden min-h-[320px] flex flex-col">
                        <div className="absolute inset-0 z-0 pointer-events-none">
                            <DNAAnimation />
                        </div>
                        <div className="relative z-10 flex-1 flex flex-col">
                            <h3 className="text-xl font-bold text-white mb-8 flex items-center gap-3">
                                <Award className="text-violet-400" size={24} />
                                DNA Blueprint: {profile.name}
                            </h3>

                            <div className="space-y-8 flex-1">
                                <div>
                                    <span className="block text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Extracted Keywords</span>
                                    <div className="flex flex-wrap gap-2.5">
                                        {(profile.top_keywords || []).map((t, i) => (
                                            <span
                                                key={t}
                                                className="px-4 py-1.5 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 rounded-full text-sm font-semibold shadow-sm transition-transform hover:-translate-y-1 cursor-default"
                                                style={{ animationDelay: `${i * 100}ms` }}
                                            >
                                                {t}
                                            </span>
                                        ))}
                                        {(!profile.top_keywords || profile.top_keywords.length === 0) && <span className="text-gray-500 italic text-sm">Awaiting extraction...</span>}
                                    </div>
                                </div>

                                <div className="pt-8 border-t border-white/10 mt-auto">
                                    <div className="bg-emerald-500/10 p-5 rounded-xl border border-emerald-500/20 w-fit">
                                        <span className="block text-xs font-bold text-emerald-400/80 uppercase tracking-widest mb-1.5">Target Sentiment Base</span>
                                        <span className="text-xl md:text-2xl font-black tracking-tight text-emerald-400">
                                            {(profile.avg_sentiment > 0.1 ? "Positive (" : "Neutral (")} {profile.avg_sentiment.toFixed(2)})
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};
