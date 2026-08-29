export const DNAAnimation = () => (
    <svg className="w-full h-32 opacity-[0.03] animate-pulse" viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M10,50 Q30,10 50,50 T90,50 T130,50 T170,50" fill="none" stroke="url(#grad1)" strokeWidth="3">
            <animate attributeName="d" dur="4s" repeatCount="indefinite" values="
        M10,50 Q30,10 50,50 T90,50 T130,50 T170,50;
        M10,50 Q30,90 50,50 T90,50 T130,50 T170,50;
        M10,50 Q30,10 50,50 T90,50 T130,50 T170,50
      "/>
        </path>
        <path d="M10,50 Q30,90 50,50 T90,50 T130,50 T170,50" fill="none" stroke="url(#grad2)" strokeWidth="3">
            <animate attributeName="d" dur="4s" repeatCount="indefinite" values="
        M10,50 Q30,90 50,50 T90,50 T130,50 T170,50;
        M10,50 Q30,10 50,50 T90,50 T130,50 T170,50;
        M10,50 Q30,90 50,50 T90,50 T130,50 T170,50
      "/>
        </path>
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#818CF8" />
                <stop offset="100%" stopColor="#C084FC" />
            </linearGradient>
            <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#E879F9" />
                <stop offset="100%" stopColor="#818CF8" />
            </linearGradient>
        </defs>
    </svg>
);
