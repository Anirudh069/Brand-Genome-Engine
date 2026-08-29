import { useState, useEffect } from 'react';
import { API_BASE } from '../../lib/constants';

export const BrandSelector = ({ onSelect, selectedId }) => {
    const [brands, setBrands] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchBrands = async () => {
            try {
                const res = await fetch(`${API_BASE}/brands`);
                if (!res.ok) throw new Error("Failed to fetch brands");
                const data = await res.json();
                setBrands(data.brands || []);

                // Auto-select first if none selected
                if (!selectedId && data.brands?.length > 0) {
                    onSelect(data.brands[0]);
                }
            } catch (err) {
                console.error(err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchBrands();
    }, []);

    if (error) return <div className="text-red-400 text-sm">Failed to load brands</div>;

    return (
        <div className="relative">
            <select
                value={selectedId || ""}
                onChange={(e) => {
                    const brand = brands.find(b => b.brand_id === e.target.value);
                    if (brand) onSelect(brand);
                }}
                disabled={loading || brands.length === 0}
                className="appearance-none bg-[#1A1A24] text-white border border-white/10 rounded-lg px-4 py-3 min-w-[200px] outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 font-semibold tracking-wide transition-all uppercase text-sm disabled:opacity-50 hover:bg-white/5 cursor-pointer"
            >
                {loading && <option value="">Loading Brands...</option>}
                {!loading && brands.length === 0 && <option value="">No Brands Available</option>}
                {brands.map((b) => (
                    <option key={b.brand_id} value={b.brand_id} className="bg-[#111116] text-white">
                        {b.brand_name}
                    </option>
                ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-gray-400">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
            </div>
        </div>
    );
};
