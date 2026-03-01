import { AlertCircle } from 'lucide-react';

export const ErrorBanner = ({ error }) => {
    if (!error) return null;

    return (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 flex items-start gap-4 animate-in fade-in slide-in-from-top-4">
            <AlertCircle className="text-red-400 mt-0.5 shrink-0" size={20} />
            <div>
                <h4 className="text-red-400 font-semibold mb-1">Analysis Error</h4>
                <p className="text-red-300/80 text-sm leading-relaxed">{error}</p>
            </div>
        </div>
    );
};
