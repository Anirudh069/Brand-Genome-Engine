import { motion } from "framer-motion";

export const NeuralNetworkSVG = () => (
    <svg className="absolute inset-0 w-full h-full opacity-[0.05] pointer-events-none" xmlns="http://www.w3.org/2000/svg">
        <motion.path
            d="M100 100 Q 300 500, 800 200 T 1500 500"
            fill="transparent"
            stroke="url(#orange-grad)"
            strokeWidth="2"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 3, ease: "easeInOut", repeat: Infinity, repeatType: "reverse" }}
        />
        <motion.path
            d="M200 800 Q 500 200, 1000 600 T 1800 300"
            fill="transparent"
            stroke="url(#rose-grad)"
            strokeWidth="2"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 4, ease: "easeInOut", repeat: Infinity, repeatType: "reverse", delay: 1 }}
        />
        <defs>
            <linearGradient id="orange-grad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#FB923C" />
                <stop offset="100%" stopColor="#F43F5E" />
            </linearGradient>
            <linearGradient id="rose-grad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#F43F5E" />
                <stop offset="100%" stopColor="#8B5CF6" />
            </linearGradient>
        </defs>
    </svg>
);
