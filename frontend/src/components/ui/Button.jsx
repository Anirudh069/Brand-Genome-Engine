import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export const Button = ({ children, primary = false, className = "", icon: Icon, onClick, disabled = false, ...props }) => {
    return (
        <motion.button
            whileHover={!disabled ? { scale: 1.02 } : {}}
            whileTap={!disabled ? { scale: 0.98 } : {}}
            onClick={onClick}
            disabled={disabled}
            className={cn(
                "relative overflow-hidden px-6 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all duration-300 group disabled:opacity-50 disabled:cursor-not-allowed",
                primary
                    ? "bg-gradient-to-r from-indigo-500 to-violet-600 text-white shadow-lg shadow-indigo-500/25 border border-indigo-400/30"
                    : "bg-[#18181B] text-gray-300 border border-white/10 hover:bg-[#27272A] hover:text-white",
                className
            )}
            {...props}
        >
            {Icon && <Icon size={18} className={primary ? "text-indigo-100" : "text-gray-400 group-hover:text-gray-200"} />}
            <span className="relative z-10 flex items-center gap-2 tracking-wide text-sm">{children}</span>

            {primary && !disabled && (
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite] skew-x-12" />
            )}
        </motion.button>
    );
};
