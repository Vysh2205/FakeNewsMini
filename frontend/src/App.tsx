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
      const res = await fetch('http://localhost:8000/api/analytics');
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
      
      const res = await fetch(`http://localhost:8000${endpoint}`, {
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
    doc.text('Veritas AI Analysis Report', 20, 20);
    doc.setFontSize(16);
    doc.text(`Status: ${result.is_fake ? 'Fake News Detected' : 'Reliable Source'}`, 20, 40);
    doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 20, 50);
    doc.setFontSize(12);
    doc.text('Explanation:', 20, 70);
    doc.text(result.explanation, 20, 80, { maxWidth: 170 });
    doc.text(`Keywords: ${result.keywords.join(', ')}`, 20, 110);
    doc.save('veritas-report.pdf');
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-8">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center space-y-4">
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            Veritas AI
          </h1>
          <p className="text-slate-400 text-lg">AI-Powered Fake News Detection & Verification Platform</p>
        </motion.div>

        <div className="flex justify-center gap-4 mb-8">
          <button onClick={() => { setActiveTab('text'); setResult(null); }} className={`px-6 py-2 rounded-full font-semibold transition ${activeTab === 'text' ? 'bg-blue-600' : 'bg-slate-800'}`}>Text Analysis</button>
          <button onClick={() => { setActiveTab('url'); setResult(null); }} className={`px-6 py-2 rounded-full font-semibold transition ${activeTab === 'url' ? 'bg-blue-600' : 'bg-slate-800'}`}>URL Analysis</button>
          <button onClick={() => setActiveTab('dashboard')} className={`px-6 py-2 rounded-full font-semibold transition ${activeTab === 'dashboard' ? 'bg-blue-600' : 'bg-slate-800'}`}>Analytics Dashboard</button>
        </div>

        {activeTab !== 'dashboard' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 p-8 rounded-2xl shadow-2xl">
            <div className="space-y-4">
              {activeTab === 'text' ? (
                <textarea
                  className="w-full h-40 bg-slate-900/50 border border-slate-600 rounded-xl p-4 text-white focus:ring-2 focus:ring-blue-500 outline-none"
                  placeholder="Paste a news article or claim here..."
                  value={inputData} onChange={(e) => setInputData(e.target.value)}
                />
              ) : (
                <div className="flex items-center bg-slate-900/50 border border-slate-600 rounded-xl px-4 py-2">
                  <LinkIcon className="text-slate-500 mr-2" />
                  <input
                    type="url"
                    className="w-full bg-transparent p-2 text-white outline-none"
                    placeholder="https://news-site.com/article"
                    value={inputData} onChange={(e) => setInputData(e.target.value)}
                  />
                </div>
              )}
              <button onClick={analyze} disabled={loading} className="w-full bg-indigo-600 hover:bg-indigo-500 py-3 rounded-xl flex items-center justify-center gap-2 font-bold transition">
                {loading ? <Activity className="animate-spin" /> : <Search />}
                {loading ? 'Analyzing with AI...' : 'Verify Content'}
              </button>
              
              {errorMsg && (
                <div className="mt-4 p-4 bg-red-900/40 border border-red-500/50 rounded-xl text-red-200">
                  <ShieldAlert className="inline mr-2" size={20} />
                  {errorMsg}
                </div>
              )}
            </div>
          </motion.div>
        )}

        {result && activeTab !== 'dashboard' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`p-8 rounded-2xl border backdrop-blur-xl relative ${result.is_fake ? 'bg-red-900/20 border-red-500/50' : 'bg-emerald-900/20 border-emerald-500/50'}`}>
            <button onClick={downloadPDF} className="absolute top-4 right-4 flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg text-sm transition border border-slate-600">
              <Download size={16} /> Export PDF
            </button>
            <div className="flex items-center gap-6">
              {result.is_fake ? <ShieldAlert className="w-16 h-16 text-red-500" /> : <ShieldCheck className="w-16 h-16 text-emerald-500" />}
              <div>
                <h2 className="text-3xl font-bold">{result.is_fake ? 'Potential Fake News' : 'Likely Reliable Source'}</h2>
                <p className="text-slate-300 text-lg mt-1">Confidence Score: <span className="font-mono font-bold text-white">{(result.confidence * 100).toFixed(1)}%</span></p>
              </div>
            </div>
            <div className="bg-slate-900/60 rounded-xl p-6 mt-6">
              <h3 className="text-lg font-bold text-slate-200 mb-2">AI Explanation</h3>
              <p className="text-slate-300 leading-relaxed">{result.explanation}</p>
            </div>
            <div className="mt-6">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3">Key Indicators</h3>
              <div className="flex gap-3 flex-wrap">
                {result.keywords.map((kw: string, i: number) => (
                  <span key={i} className="px-4 py-1.5 bg-slate-800/80 rounded-full text-sm font-medium border border-slate-600">{kw}</span>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'dashboard' && stats && (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 text-center shadow-lg">
              <BarChart2 className="w-10 h-10 mx-auto text-blue-400 mb-2" />
              <h3 className="text-slate-400 font-semibold">Total Analyzed</h3>
              <p className="text-4xl font-bold text-white mt-2">{stats.total_analyzed}</p>
            </div>
            <div className="bg-slate-800 p-6 rounded-2xl border border-red-500/30 text-center shadow-lg">
              <ShieldAlert className="w-10 h-10 mx-auto text-red-400 mb-2" />
              <h3 className="text-red-300 font-semibold">Fake News Detected</h3>
              <p className="text-4xl font-bold text-red-400 mt-2">{stats.fake_detected}</p>
            </div>
            <div className="bg-slate-800 p-6 rounded-2xl border border-emerald-500/30 text-center shadow-lg">
              <ShieldCheck className="w-10 h-10 mx-auto text-emerald-400 mb-2" />
              <h3 className="text-emerald-300 font-semibold">Reliable Sources</h3>
              <p className="text-4xl font-bold text-emerald-400 mt-2">{stats.real_detected}</p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
