import { Lightbulb } from 'lucide-react';

export const SuggestionsPanel = ({ suggestions }) => {
    if (!suggestions || suggestions.length === 0) return null;

    return (
        <div className="bg-[#1A1A24]/60 border border-white/10 rounded-xl p-6 mt-6">
            <div className="flex items-center gap-3 mb-4">
                <Lightbulb className="text-yellow-400" size={20} />
                <h3 className="text-lg font-semibold text-white">Improvement Plan</h3>
            </div>
            <ul className="space-y-3">
                {suggestions.map((s, i) => (
                    <li key={i} className="flex items-start gap-3 text-gray-300 text-sm leading-relaxed">
                        <span className="text-indigo-400 mt-1 flex-shrink-0 text-xs font-mono">{i + 1}.</span>
                        <span>{s}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
};
