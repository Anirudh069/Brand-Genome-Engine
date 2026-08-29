import { Card } from "./Card";
import { TrendingUp } from "lucide-react";

export const Metric = ({ label, value, trend, icon: Icon, delay }) => (
    <Card delay={delay} className="flex flex-col relative overflow-hidden group p-6">
        <div className="absolute -right-12 -top-12 w-32 h-32 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 rounded-full blur-3xl group-hover:bg-indigo-500/20 transition-colors duration-700" />

        <div className="flex justify-between items-start mb-6">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">{label}</span>
            {Icon && (
                <div className="p-2.5 bg-white/5 border border-white/10 text-indigo-400 rounded-xl group-hover:scale-110 transition-transform duration-500">
                    <Icon size={18} />
                </div>
            )}
        </div>

        <div className="flex flex-col gap-2 mt-auto relative z-10">
            <span className="text-4xl font-black tracking-tighter text-white drop-shadow-md">
                {value}
            </span>
            {trend && (
                <span className="inline-flex items-center text-xs font-bold text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-2.5 py-1 rounded-full w-fit">
                    <TrendingUp size={12} className="mr-1.5" />
                    {trend}
                </span>
            )}
        </div>
    </Card>
);
