import { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

export const GroundingExamples = ({ chunks }) => {
    const [isOpen, setIsOpen] = useState(false);

    if (!chunks || chunks.length === 0) return null;

    return (
        <div className="bg-black/20 border border-white/5 rounded-xl overflow-hidden mt-6 transition-all duration-300">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <BookOpen className="text-gray-400" size={18} />
                    <span className="text-sm font-semibold text-gray-200 tracking-wide uppercase">Grounding Examples ({chunks.length})</span>
                </div>
                {isOpen ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
            </button>

            {isOpen && (
                <div className="p-4 bg-black/40 border-t border-white/5 animate-in slide-in-from-top-2">
                    <div className="space-y-4">
                        {chunks.map((chunk, idx) => (
                            <div key={idx} className="flex gap-3 text-sm text-gray-400 leading-relaxed border-l-2 border-indigo-500/30 pl-3 py-1">
                                <p>"{chunk}"</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};
