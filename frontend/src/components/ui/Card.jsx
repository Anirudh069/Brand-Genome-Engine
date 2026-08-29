import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export const Card = ({ children, className = "", delay = 0, ...props }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
                duration: 0.5,
                delay: delay / 1000,
                ease: [0.16, 1, 0.3, 1]
            }}
            className={cn(
                "bg-[#111116]/80 backdrop-blur-2xl border border-white/5 shadow-2xl rounded-2xl p-6 md:p-8 relative overflow-hidden group",
                "before:absolute before:inset-0 before:bg-gradient-to-br before:from-white/5 before:to-transparent before:opacity-0 before:transition-opacity hover:before:opacity-100",
                className
            )}
            {...props}
        >
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
            <div className="relative z-10">{children}</div>
        </motion.div>
    );
};
