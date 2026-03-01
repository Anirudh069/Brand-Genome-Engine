import { cn } from "../../lib/utils";

export const TextArea = ({ label, placeholder, rows = 4, value, onChange, className }) => (
    <div className="mb-5 group w-full">
        {label && (
            <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2 transition-colors group-focus-within:text-indigo-400">
                {label}
            </label>
        )}
        <textarea
            placeholder={placeholder}
            rows={rows}
            value={value}
            onChange={onChange}
            className={cn(
                "w-full px-4 py-3.5 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all duration-300 text-gray-100 placeholder:text-gray-600 shadow-inner resize-none",
                className
            )}
        />
    </div>
);
