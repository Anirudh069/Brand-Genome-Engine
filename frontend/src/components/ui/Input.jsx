import { cn } from "../../lib/utils";

export const Input = ({ label, placeholder, type = "text", value, onChange, icon: Icon, className }) => (
    <div className="mb-5 group w-full">
        {label && (
            <label className="block text-xs font-semibold tracking-widest uppercase text-gray-400 mb-2 transition-colors group-focus-within:text-indigo-400">
                {label}
            </label>
        )}
        <div className="relative">
            {Icon && (
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-indigo-400 transition-colors">
                    <Icon size={18} />
                </div>
            )}
            <input
                type={type}
                placeholder={placeholder}
                value={value}
                onChange={onChange}
                className={cn(
                    "w-full py-3.5 bg-[#09090B] border border-white/10 rounded-xl focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all duration-300 text-gray-100 placeholder:text-gray-600 shadow-inner",
                    Icon ? "pl-11 pr-4" : "px-4",
                    className
                )}
            />
        </div>
    </div>
);
