import { useState } from 'react';
import { Sparkles, Copy, Check } from 'lucide-react';
import { TextArea } from './TextArea';

export const RewritePanel = ({ text }) => {
    const [editableText, setEditableText] = useState(text || "");
    const [copied, setCopied] = useState(false);

    if (!text) return null;

    const handleCopy = () => {
        navigator.clipboard.writeText(editableText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-xl p-6 relative group overflow-hidden mt-6">
            <div className="absolute top-0 right-0 p-4 opacity-50 pointer-events-none group-hover:opacity-100 transition-opacity">
                <Sparkles className="text-indigo-400 w-24 h-24 blur-2xl" />
            </div>

            <div className="flex items-center justify-between mb-4 relative z-10">
                <div className="flex items-center gap-2">
                    <Sparkles className="text-indigo-400" size={20} />
                    <h3 className="text-lg font-bold text-white tracking-wide">Brand-Aligned Rewrite</h3>
                </div>
                <button
                    onClick={handleCopy}
                    className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-400 hover:text-white transition-colors bg-indigo-500/10 hover:bg-indigo-500/30 px-3 py-1.5 rounded-md"
                >
                    {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy</>}
                </button>
            </div>

            <div className="relative z-10">
                <TextArea
                    value={editableText}
                    onChange={(e) => setEditableText(e.target.value)}
                    rows={6}
                    className="font-serif text-lg leading-relaxed text-indigo-50 bg-black/40 border-indigo-500/30 focus:border-indigo-400 shadow-[inset_0_2px_10px_rgba(0,0,0,0.2)]"
                />
            </div>
        </div>
    );
};
