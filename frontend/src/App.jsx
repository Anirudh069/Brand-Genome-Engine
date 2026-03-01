import { useState, useEffect } from 'react';
import { MainLayout } from './layouts/MainLayout';
import { BrandSetup } from './pages/BrandSetup';
import { ConsistencyCheck } from './pages/ConsistencyCheck';
import { Benchmarking } from './pages/Benchmarking';
import { Analytics } from './pages/Analytics';
import { API_BASE } from './lib/constants';

function App() {
  const [activeTab, setActiveTab] = useState('setup');
  const [profile, setProfile] = useState(null);

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API_BASE}/profile`);
      if (res.ok) {
        const data = await res.json();
        setProfile(data);
      }
    } catch (err) {
      console.error("Failed to fetch profile", err);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  return (
    <MainLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      {activeTab === 'setup' && <BrandSetup profile={profile} fetchProfile={fetchProfile} />}
      {activeTab === 'check' && <ConsistencyCheck profile={profile} />}
      {activeTab === 'bench' && <Benchmarking profile={profile} />}
      {activeTab === 'analytics' && <Analytics />}
    </MainLayout>
  );
}

export default App;
