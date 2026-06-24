import { useState } from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, Search, Activity } from 'lucide-react';

export default function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/analyze/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Veritas AI
          </h1>
          <p className="text-slate-400 text-lg">
            Advanced Fake News Detection & Verification Platform
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 p-6 rounded-2xl shadow-xl"
        >
          <div className="space-y-4">
            <textarea
              className="w-full h-40 bg-slate-900/50 border border-slate-600 rounded-xl p-4 text-white placeholder:text-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              placeholder="Paste a news article or claim here to verify its authenticity..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button
              onClick={analyze}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <Activity className="animate-spin" />
              ) : (
                <Search />
              )}
              {loading ? 'Analyzing...' : 'Analyze Content'}
            </button>
          </div>
        </motion.div>

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`p-6 rounded-2xl border backdrop-blur-xl flex flex-col gap-4 ${
              result.is_fake 
                ? 'bg-red-900/20 border-red-500/50' 
                : 'bg-green-900/20 border-green-500/50'
            }`}
          >
            <div className="flex items-center gap-4">
              {result.is_fake ? (
                <ShieldAlert className="w-12 h-12 text-red-400" />
              ) : (
                <ShieldCheck className="w-12 h-12 text-green-400" />
              )}
              <div>
                <h2 className="text-2xl font-bold">
                  {result.is_fake ? 'Potential Fake News' : 'Likely Reliable'}
                </h2>
                <p className="text-slate-300">
                  Confidence Score: <span className="font-mono font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                </p>
              </div>
            </div>
            
            <div className="bg-slate-900/50 rounded-xl p-4 mt-2">
              <h3 className="font-semibold text-slate-300 mb-2">Explanation</h3>
              <p>{result.explanation}</p>
            </div>

            <div className="mt-2">
              <h3 className="font-semibold text-slate-300 mb-2">Key Indicators</h3>
              <div className="flex gap-2 flex-wrap">
                {result.keywords.map((kw: string, i: number) => (
                  <span key={i} className="px-3 py-1 bg-slate-800 rounded-full text-sm border border-slate-700">
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
