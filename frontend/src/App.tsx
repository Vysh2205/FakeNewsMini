import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, Search, Activity, Link as LinkIcon, Download, BarChart2 } from 'lucide-react';
import { jsPDF } from 'jspdf';

export default function App() {
  const [activeTab, setActiveTab] = useState<'text' | 'url' | 'dashboard'>('text');
  const [inputData, setInputData] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/analytics');
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchStats();
    }
  }, [activeTab]);

  const analyze = async () => {
    if (!inputData.trim()) return;
    setLoading(true);
    setErrorMsg(null);
    setResult(null);
    try {
      const endpoint = activeTab === 'text' ? '/api/analyze/text' : '/api/analyze/url';
      const body = activeTab === 'text' ? { text: inputData } : { url: inputData };
      
      const res = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`);
      }
      
      const data = await res.json();
      setResult(data);
    } catch (error: any) {
      console.error(error);
      setErrorMsg("Failed to connect to the backend server. Please make sure the FastAPI server is running on port 8000.");
    }
    setLoading(false);
  };

  const downloadPDF = () => {
    if (!result) return;
    const doc = new jsPDF();
    doc.setFontSize(22);
    doc.text('Bumblebee AI Analysis Report', 20, 20);
    doc.setFontSize(16);
    doc.text(`Status: ${result.is_fake ? 'Fake News Detected' : 'Reliable Source'}`, 20, 40);
    doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 20, 50);
    doc.setFontSize(12);
    doc.text('Explanation:', 20, 70);
    doc.text(result.explanation, 20, 80, { maxWidth: 170 });
    doc.text(`Keywords: ${result.keywords.join(', ')}`, 20, 110);
    doc.save('bumblebee-report.pdf');
  };

  return (
    <div className="min-h-screen bg-[#08080a] text-zinc-100 p-8 font-sans selection:bg-amber-500 selection:text-black">
      <div className="max-w-5xl mx-auto space-y-8">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-semibold uppercase tracking-wider mb-2">
            🐝 Next-Gen Verification
          </div>
          <h1 className="text-6xl font-black bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 bg-clip-text text-transparent tracking-tight">
            Bumblebee AI
          </h1>
          <p className="text-zinc-400 text-lg max-w-xl mx-auto">Ultra-fast AI-powered verification engine for detecting misinformation and validating claims.</p>
        </motion.div>

        <div className="flex justify-center gap-3 mb-8">
          <button onClick={() => { setActiveTab('text'); setResult(null); }} className={`px-6 py-2.5 rounded-xl font-bold transition-all duration-300 ${activeTab === 'text' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700'}`}>Text Analysis</button>
          <button onClick={() => { setActiveTab('url'); setResult(null); }} className={`px-6 py-2.5 rounded-xl font-bold transition-all duration-300 ${activeTab === 'url' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700'}`}>URL Analysis</button>
          <button onClick={() => setActiveTab('dashboard')} className={`px-6 py-2.5 rounded-xl font-bold transition-all duration-300 ${activeTab === 'dashboard' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:border-zinc-700'}`}>Analytics Dashboard</button>
        </div>

        {activeTab !== 'dashboard' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-zinc-900/40 backdrop-blur-xl border border-zinc-800 p-8 rounded-2xl shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-amber-500/50 to-transparent" />
            <div className="space-y-5">
              {activeTab === 'text' ? (
                <textarea
                  className="w-full h-44 bg-zinc-950/60 border border-zinc-800 rounded-xl p-4 text-white focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/50 outline-none transition placeholder-zinc-600"
                  placeholder="Paste a news article, social media post, or claim here..."
                  value={inputData} onChange={(e) => setInputData(e.target.value)}
                />
              ) : (
                <div className="flex items-center bg-zinc-950/60 border border-zinc-800 focus-within:ring-2 focus-within:ring-amber-500/50 focus-within:border-amber-500/50 rounded-xl px-4 py-2.5 transition">
                  <LinkIcon className="text-zinc-500 mr-2" size={18} />
                  <input
                    type="url"
                    className="w-full bg-transparent p-1.5 text-white outline-none placeholder-zinc-600"
                    placeholder="https://example-news.com/article-url"
                    value={inputData} onChange={(e) => setInputData(e.target.value)}
                  />
                </div>
              )}
              <button onClick={analyze} disabled={loading} className="w-full bg-gradient-to-r from-amber-500 to-yellow-400 hover:from-amber-400 hover:to-yellow-300 text-black py-3.5 rounded-xl flex items-center justify-center gap-2 font-black transition-all duration-300 shadow-lg shadow-amber-500/10 active:scale-[0.99] disabled:opacity-50">
                {loading ? <Activity className="animate-spin" /> : <Search size={20} />}
                {loading ? 'Analyzing with Bumblebee AI...' : 'Verify Content'}
              </button>
              
              {errorMsg && (
                <div className="mt-4 p-4 bg-red-950/40 border border-red-500/30 rounded-xl text-red-200 flex items-start gap-2">
                  <ShieldAlert className="text-red-400 shrink-0 mt-0.5" size={20} />
                  <div>
                    <h4 className="font-bold text-red-400">Connection Error</h4>
                    <p className="text-sm text-red-300/90 mt-0.5">{errorMsg}</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {result && activeTab !== 'dashboard' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`p-8 rounded-2xl border backdrop-blur-xl relative overflow-hidden ${result.is_fake ? 'bg-red-950/20 border-red-500/30' : 'bg-emerald-950/20 border-emerald-500/30'}`}>
            <button onClick={downloadPDF} className="absolute top-4 right-4 flex items-center gap-2 bg-zinc-900/80 hover:bg-zinc-800 px-4 py-2 rounded-xl text-xs font-bold transition border border-zinc-800">
              <Download size={14} /> Export Report
            </button>
            <div className="flex items-center gap-6">
              {result.is_fake ? <ShieldAlert className="w-16 h-16 text-red-500" /> : <ShieldCheck className="w-16 h-16 text-emerald-500" />}
              <div>
                <h2 className="text-3xl font-black">{result.is_fake ? 'Potential Fake News' : 'Likely Reliable Source'}</h2>
                <p className="text-zinc-400 text-lg mt-1">Confidence Score: <span className={`font-mono font-bold ${result.is_fake ? 'text-red-400' : 'text-emerald-400'}`}>{(result.confidence * 100).toFixed(1)}%</span></p>
              </div>
            </div>
            <div className="bg-zinc-950/60 border border-zinc-800/80 rounded-xl p-6 mt-6">
              <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-400 mb-2">Bumblebee Verdict Explanation</h3>
              <p className="text-zinc-300 leading-relaxed">{result.explanation}</p>
            </div>
            <div className="mt-6">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-3">Key Risk Indicators</h3>
              <div className="flex gap-2 flex-wrap">
                {result.keywords.map((kw: string, i: number) => (
                  <span key={i} className="px-3.5 py-1.5 bg-zinc-900 border border-zinc-800 rounded-full text-xs font-semibold text-zinc-300">{kw}</span>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'dashboard' && stats && (
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-zinc-900/40 border border-zinc-800 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[2px] bg-amber-500/30" />
              <BarChart2 className="w-10 h-10 mx-auto text-amber-400 mb-2" />
              <h3 className="text-zinc-400 font-semibold text-sm">Total Analyzed</h3>
              <p className="text-4xl font-extrabold text-white mt-2">{stats.total_analyzed}</p>
            </div>
            <div className="bg-zinc-900/40 border border-red-500/20 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[2px] bg-red-500/30" />
              <ShieldAlert className="w-10 h-10 mx-auto text-red-400 mb-2" />
              <h3 className="text-zinc-400 font-semibold text-sm">Fake News Detected</h3>
              <p className="text-4xl font-extrabold text-red-400 mt-2">{stats.fake_detected}</p>
            </div>
            <div className="bg-zinc-900/40 border border-emerald-500/20 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[2px] bg-emerald-500/30" />
              <ShieldCheck className="w-10 h-10 mx-auto text-emerald-400 mb-2" />
              <h3 className="text-zinc-400 font-semibold text-sm">Reliable Sources</h3>
              <p className="text-4xl font-extrabold text-emerald-400 mt-2">{stats.real_detected}</p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
}
