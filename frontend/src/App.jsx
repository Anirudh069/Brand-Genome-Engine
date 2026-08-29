import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { MainLayout } from './layouts/MainLayout';
import { BrandSetup } from './pages/BrandSetup';
import { ConsistencyCheck } from './pages/ConsistencyCheck';
import { Benchmarking } from './pages/Benchmarking';
import { Analytics } from './pages/Analytics';
import { API_BASE } from './lib/constants';

function App() {
  const [activeTab, setActiveTab] = useState('setup');
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API_BASE}/profile`);
      if (res.ok) {
        const data = await res.json();
        console.log("Global Profile State Fetched:", data);
        setProfile(data);
      }
    } catch (err) {
      console.error("Failed to fetch profile", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#09090B] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="animate-spin text-indigo-500" size={48} />
          <span className="text-gray-500 font-bold uppercase tracking-[0.2em] text-xs">Initialising Engine...</span>
        </div>
      </div>
    );
  }

  return (
    <MainLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      {activeTab === 'setup' && <BrandSetup profile={profile} fetchProfile={fetchProfile} />}
      {activeTab === 'check' && <ConsistencyCheck profile={profile} />}
      {activeTab === 'bench' && <Benchmarking profile={profile} />}
      {activeTab === 'analytics' && <Analytics profile={profile} />}
    </MainLayout>
  );
}

export default App;
