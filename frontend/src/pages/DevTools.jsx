import { useState } from 'react';
import { AlertTriangle, Loader2, RefreshCcw, Shield, SplitSquareHorizontal, Layers3, Fingerprint } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { rebuildChunks, rebuildIndex, rebuildProfile } from '../lib/rebuildApi';

const actions = {
  profile: {
    title: 'Rebuild User Profile',
    description: 'Recompute the active user profile from the saved mission and snippet texts.',
    confirm: 'Rebuild the active user profile from the current saved mission and snippets?',
    icon: Fingerprint,
  },
  chunks: {
    title: 'Rebuild Chunks',
    description: 'Rebuild canonical brand_chunks and refresh the RAG index from the new corpus.',
    confirm: 'Rebuild all chunks from the current canonical texts? This will also refresh the RAG index.',
    icon: SplitSquareHorizontal,
  },
  index: {
    title: 'Rebuild RAG Index',
    description: 'Rebuild the FAISS index directly from the current canonical chunks.',
    confirm: 'Rebuild the RAG index from the current chunks?',
    icon: Layers3,
  },
};

const summaryFor = (action, result) => {
  if (!result) return null;
  if (action === 'profile') {
    return [
      `status: ${result.status}`,
      `sources: ${result.source_texts}`,
      `embedding_dim: ${result.embedding_dim}`,
      `version: ${result.genome_version}`,
    ];
  }
  if (action === 'chunks') {
    return [
      `status: ${result.status}`,
      `total: ${result.total_chunks}`,
      `user: ${result.user_chunks}`,
      `competitor: ${result.competitor_chunks}`,
      `index_rebuilt: ${String(result.index_rebuilt)}`,
    ];
  }
  return [
    `status: ${result.status}`,
    `indexed: ${result.total_chunks}`,
    `brands: ${result.brands_indexed}`,
    `fingerprint: ${String(result.fingerprint || '').slice(0, 12)}`,
    `model: ${result.model_name} / ${result.embedding_dim}`,
  ];
};

export const DevTools = () => {
  const [loadingAction, setLoadingAction] = useState(null);
  const [results, setResults] = useState({});
  const [errors, setErrors] = useState({});

  const runAction = async (action) => {
    const meta = actions[action];
    if (!window.confirm(meta.confirm)) return;

    setLoadingAction(action);
    setErrors((current) => ({ ...current, [action]: null }));

    try {
      const result = action === 'profile'
        ? await rebuildProfile()
        : action === 'chunks'
          ? await rebuildChunks()
          : await rebuildIndex();
      setResults((current) => ({ ...current, [action]: result }));
    } catch (err) {
      setErrors((current) => ({ ...current, [action]: err.message || 'Rebuild failed.' }));
    } finally {
      setLoadingAction(null);
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out fill-mode-both max-w-6xl mx-auto px-4 sm:px-6">
      <div className="mb-10 flex items-center gap-5">
        <div className="p-3.5 bg-gradient-to-br from-slate-700 to-indigo-600 rounded-2xl shadow-[0_0_30px_rgba(99,102,241,0.2)] text-white border border-indigo-400/30">
          <Shield size={28} />
        </div>
        <div>
          <h2 className="text-3xl md:text-4xl font-black text-white tracking-tight">Dev Tools</h2>
          <p className="text-gray-400 mt-2 text-lg">Controlled rebuild actions for the local academic PoC.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {Object.entries(actions).map(([key, meta]) => {
          const Icon = meta.icon;
          const result = results[key];
          const error = errors[key];
          const summary = summaryFor(key, result);
          const isLoading = loadingAction === key;

          return (
            <Card key={key} className="flex flex-col gap-5 min-h-[280px]">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-2xl bg-white/5 border border-white/10 text-indigo-300 shrink-0">
                  <Icon size={22} />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">{meta.title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed">{meta.description}</p>
                </div>
              </div>

              <Button primary className="w-full py-4" onClick={() => runAction(key)} disabled={!!loadingAction}>
                {isLoading ? <Loader2 className="animate-spin" /> : <RefreshCcw size={18} />}
                {isLoading ? 'Running...' : meta.title}
              </Button>

              {error && (
                <div className="text-xs font-bold px-4 py-3 rounded-xl border bg-red-500/10 border-red-500/20 text-red-300 flex items-start gap-2">
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {summary && !error && (
                <div className="grid grid-cols-1 gap-2 text-xs font-semibold text-gray-300 bg-black/20 border border-white/5 rounded-xl p-4">
                  {summary.map((line) => (
                    <div key={line} className="truncate">
                      {line}
                    </div>
                  ))}
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
};
