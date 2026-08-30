import { useState, useEffect } from 'react';
<<<<<<< HEAD
import { Loader2 } from 'lucide-react';
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
import { MainLayout } from './layouts/MainLayout';
import { BrandSetup } from './pages/BrandSetup';
import { ConsistencyCheck } from './pages/ConsistencyCheck';
import { Benchmarking } from './pages/Benchmarking';
import { Analytics } from './pages/Analytics';
import { API_BASE } from './lib/constants';

function App() {
  const [activeTab, setActiveTab] = useState('setup');
  const [profile, setProfile] = useState(null);
<<<<<<< HEAD
  const [loading, setLoading] = useState(true);
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API_BASE}/profile`);
      if (res.ok) {
        const data = await res.json();
<<<<<<< HEAD
        console.log("Global Profile State Fetched:", data);
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
        setProfile(data);
      }
    } catch (err) {
      console.error("Failed to fetch profile", err);
<<<<<<< HEAD
    } finally {
      setLoading(false);
=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

<<<<<<< HEAD
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

=======
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
  return (
    <MainLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      {activeTab === 'setup' && <BrandSetup profile={profile} fetchProfile={fetchProfile} />}
      {activeTab === 'check' && <ConsistencyCheck profile={profile} />}
      {activeTab === 'bench' && <Benchmarking profile={profile} />}
<<<<<<< HEAD
      {activeTab === 'analytics' && <Analytics profile={profile} />}
=======
      {activeTab === 'analytics' && <Analytics />}
>>>>>>> 5c33a850df3e0cde7b2d472ca31c397ba19febcb
    </MainLayout>
  );
}

export default App;
