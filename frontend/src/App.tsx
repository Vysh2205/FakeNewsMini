import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ShieldAlert, ShieldCheck, Search, Activity, Link as LinkIcon, Download, 
  BarChart2, FileText, Image as ImageIcon, History as HistoryIcon,
  ExternalLink, Globe, AlertTriangle, Info, Upload, Volume2
} from 'lucide-react';
import { jsPDF } from 'jspdf';

const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
  }
  return 'https://mario-redeem-backup-texts.trycloudflare.com';
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'news' | 'media' | 'history' | 'dashboard' | 'workflow'>('news');
  const [subTab, setSubTab] = useState<'text' | 'url' | 'image' | 'video' | 'audio'>('text');
  
  const [inputData, setInputData] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [historyList, setHistoryList] = useState<any[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/analytics`, {
        headers: { 'Bypass-Tunnel-Reminder': 'true' }
      });
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchHistory = async () => {
    try {
      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}/api/history`, {
        headers: { 'Bypass-Tunnel-Reminder': 'true' }
      });
      const data = await res.json();
      setHistoryList(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchStats();
    } else if (activeTab === 'history') {
      fetchHistory();
    }
  }, [activeTab]);

  const analyze = async () => {
    setLoading(true);
    setErrorMsg(null);
    setResult(null);
    try {
      let endpoint = '';
      let options: RequestInit = {};

      if (subTab === 'text') {
        if (!inputData.trim()) throw new Error("Please enter text or a claim to analyze.");
        endpoint = '/api/verify/text';
        options = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: JSON.stringify({ text: inputData })
        };
      } else if (subTab === 'url') {
        if (!inputData.trim()) throw new Error("Please enter a news URL to analyze.");
        endpoint = '/api/verify/url';
        options = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: JSON.stringify({ url: inputData })
        };
      } else if (subTab === 'image' || subTab === 'video' || subTab === 'audio') {
        if (!selectedFile) throw new Error(`Please select an ${subTab} file to upload.`);
        endpoint = subTab === 'audio' ? '/api/verify/audio' : (subTab === 'image' ? '/api/verify/image' : '/api/verify/video');
        const formData = new FormData();
        formData.append('file', selectedFile);
        options = {
          method: 'POST',
          headers: {
            'Bypass-Tunnel-Reminder': 'true'
          },
          body: formData
        };
      }

      const baseUrl = getApiBaseUrl();
      const res = await fetch(`${baseUrl}${endpoint}`, options);
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status}): ${res.statusText}`);
      }
      
      const data = await res.json();
      setResult(data);
    } catch (error: any) {
      console.error(error);
      const isLocalHost = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
      if (error.message === "Failed to fetch" || error.name === "TypeError") {
        if (isLocalHost) {
          setErrorMsg("Unable to connect to the verification server. Please check that the Python FastAPI backend is running at http://localhost:8000.");
        } else {
          setErrorMsg("Verification service is currently unavailable. The FastAPI backend must be deployed to a cloud server (e.g. Render or Railway) and configured as VITE_API_URL.");
        }
      } else {
        setErrorMsg(error.message || "Failed to connect to the backend server.");
      }
    }
    setLoading(false);
  };

  const downloadPDF = () => {
    if (!result) return;
    const doc = new jsPDF();
    doc.setFontSize(22);
    doc.text('FakeBuster AI Analysis Report', 20, 20);
    doc.setFontSize(16);
    doc.text(`Verdict: ${result.verdict_type || (result.is_fake ? 'FAKE' : 'REAL')}`, 20, 40);
    doc.text(`Confidence Score: ${(result.confidence * 100).toFixed(1)}%`, 20, 50);
    doc.text(`Overall Risk: ${result.overall_risk}% (${result.verdict})`, 20, 60);
    doc.setFontSize(12);
    doc.text('AI Explanation:', 20, 80);
    doc.text(result.explanation || '', 20, 90, { maxWidth: 170 });
    if (result.claims && result.claims.length > 0) {
      doc.text(`Extracted Claims: ${result.claims.join(' | ')}`, 20, 120, { maxWidth: 170 });
    }
    doc.save(`fakebuster-report-${result.verification_id || 'result'}.pdf`);
  };

  return (
    <div className="min-h-screen bg-[#08080a] text-zinc-100 p-4 md:p-8 font-sans selection:bg-amber-500 selection:text-black">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-semibold uppercase tracking-wider mb-1">
            🛡️ Multimodal Misinformation Engine
          </div>
          <h1 className="text-5xl md:text-6xl font-black bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-500 bg-clip-text text-transparent tracking-tight">
            FakeBuster AI
          </h1>
          <p className="text-zinc-400 text-base md:text-lg max-w-2xl mx-auto">
            Real-time AI verification platform for news text, URLs, images, and videos with multilingual detection and fact-checking integration.
          </p>
        </motion.div>

        {/* Primary Navigation Tabs */}
        <div className="flex flex-wrap justify-center gap-2 md:gap-3 bg-zinc-950 p-2 rounded-2xl border border-zinc-800/80 shadow-lg">
          <button 
            onClick={() => { setActiveTab('news'); setSubTab('text'); setResult(null); setErrorMsg(null); }} 
            className={`px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 transition-all ${activeTab === 'news' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'text-zinc-400 hover:text-white'}`}
          >
            <FileText size={16} /> Verify News
          </button>
          
          <button 
            onClick={() => { setActiveTab('media'); setSubTab('image'); setResult(null); setErrorMsg(null); }} 
            className={`px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 transition-all ${activeTab === 'media' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'text-zinc-400 hover:text-white'}`}
          >
            <ImageIcon size={16} /> Verify Media
          </button>
          
          <button 
            onClick={() => setActiveTab('history')} 
            className={`px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 transition-all ${activeTab === 'history' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'text-zinc-400 hover:text-white'}`}
          >
            <HistoryIcon size={16} /> History
          </button>
          
          <button 
            onClick={() => setActiveTab('dashboard')} 
            className={`px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 transition-all ${activeTab === 'dashboard' ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/20' : 'text-zinc-400 hover:text-white'}`}
          >
            <BarChart2 size={16} /> Analytics
          </button>
        </div>

        {/* Input Card Container */}
        {(activeTab === 'news' || activeTab === 'media') && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-zinc-900/40 backdrop-blur-xl border border-zinc-800 p-6 md:p-8 rounded-2xl shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-amber-500/50 to-transparent" />
            
            {/* Sub-tabs */}
            <div className="flex gap-4 border-b border-zinc-800/80 pb-4 mb-6">
              {activeTab === 'news' ? (
                <>
                  <button onClick={() => setSubTab('text')} className={`text-sm font-bold pb-1 transition ${subTab === 'text' ? 'text-amber-400 border-b-2 border-amber-400' : 'text-zinc-500 hover:text-zinc-300'}`}>Text / Article</button>
                  <button onClick={() => setSubTab('url')} className={`text-sm font-bold pb-1 transition ${subTab === 'url' ? 'text-amber-400 border-b-2 border-amber-400' : 'text-zinc-500 hover:text-zinc-300'}`}>News URL</button>
                </>
              ) : (
                <>
                  <button onClick={() => setSubTab('image')} className={`text-sm font-bold pb-1 transition ${subTab === 'image' ? 'text-amber-400 border-b-2 border-amber-400' : 'text-zinc-500 hover:text-zinc-300'}`}>Image Verification</button>
                  <button onClick={() => setSubTab('video')} className={`text-sm font-bold pb-1 transition ${subTab === 'video' ? 'text-amber-400 border-b-2 border-amber-400' : 'text-zinc-500 hover:text-zinc-300'}`}>Video Verification</button>
                  <button onClick={() => setSubTab('audio')} className={`text-sm font-bold pb-1 transition ${subTab === 'audio' ? 'text-amber-400 border-b-2 border-amber-400' : 'text-zinc-500 hover:text-zinc-300'}`}>Audio Deepfake</button>
                </>
              )}
            </div>

            <div className="space-y-5">
              {subTab === 'text' && (
                <textarea
                  className="w-full h-44 bg-zinc-950/70 border border-zinc-800 rounded-xl p-4 text-white focus:ring-2 focus:ring-amber-500/50 outline-none transition placeholder-zinc-600 font-sans"
                  placeholder="Paste news text, headline, or claim in English, Hindi (हिंदी), or Telugu (తెలుగు)..."
                  value={inputData} onChange={(e) => setInputData(e.target.value)}
                />
              )}

              {subTab === 'url' && (
                <div className="flex items-center bg-zinc-950/70 border border-zinc-800 focus-within:ring-2 focus-within:ring-amber-500/50 rounded-xl px-4 py-3">
                  <LinkIcon className="text-zinc-500 mr-3 shrink-0" size={20} />
                  <input
                    type="url"
                    className="w-full bg-transparent text-white outline-none placeholder-zinc-600"
                    placeholder="https://example-news.com/article-to-verify"
                    value={inputData} onChange={(e) => setInputData(e.target.value)}
                  />
                </div>
              )}

              {(subTab === 'image' || subTab === 'video' || subTab === 'audio') && (
                <div className="border-2 border-dashed border-zinc-800 rounded-xl p-8 text-center bg-zinc-950/40 hover:border-amber-500/40 transition">
                  {subTab === 'audio' ? <Volume2 className="mx-auto text-amber-400 mb-3" size={36} /> : <Upload className="mx-auto text-amber-400 mb-3" size={36} />}
                  <input
                    type="file"
                    accept={
                      subTab === 'image' ? "image/jpeg,image/jpg,image/png,image/webp" :
                      subTab === 'video' ? "video/mp4,video/mov,video/avi" :
                      "audio/mp3,audio/mpeg,audio/wav,audio/m4a,audio/aac,audio/x-m4a,audio/x-aac,.mp3,.wav,.m4a,.aac"
                    }
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    className="hidden" id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-white px-5 py-2.5 rounded-xl font-bold text-sm inline-block">
                    {selectedFile ? selectedFile.name : `Select ${subTab === 'audio' ? 'Audio' : (subTab === 'image' ? 'Image' : 'Video')} File`}
                  </label>
                  <p className="text-xs text-zinc-500 mt-2">
                    {subTab === 'audio' ? "Supported: MP3, WAV, M4A, AAC (Max 25MB)" :
                     subTab === 'image' ? "Supported: JPG, JPEG, PNG, WEBP (Max 10MB)" : "Supported: MP4, MOV, AVI (Max 50MB)"}
                  </p>
                </div>
              )}

              <button 
                onClick={analyze} 
                disabled={loading} 
                className="w-full bg-gradient-to-r from-amber-500 to-yellow-400 hover:from-amber-400 hover:to-yellow-300 text-black py-4 rounded-xl flex items-center justify-center gap-2 font-black transition-all duration-300 shadow-lg shadow-amber-500/10 active:scale-[0.99] disabled:opacity-50"
              >
                {loading ? <Activity className="animate-spin" /> : <Search size={20} />}
                {loading ? 'Analyzing with FakeBuster AI Engine...' : 'Run FakeBuster Verification'}
              </button>
              
              {errorMsg && (
                <div className="mt-4 p-4 bg-red-950/40 border border-red-500/30 rounded-xl text-red-200 flex items-start gap-3">
                  <ShieldAlert className="text-red-400 shrink-0 mt-0.5" size={22} />
                  <div>
                    <h4 className="font-bold text-red-400">Verification Alert</h4>
                    <p className="text-sm text-red-300/90 mt-0.5">{errorMsg}</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* Verification Result Card */}
        {result && (activeTab === 'news' || activeTab === 'media') && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            
            {/* PROMINENT TOP VERDICT BANNER */}
            <div className={`p-8 rounded-2xl border backdrop-blur-xl relative overflow-hidden ${
              (result.verdict_type === 'UNAVAILABLE' || result.status === 'unavailable') ? 'bg-amber-950/40 border-amber-500/60' :
              result.verdict_type === 'FAKE' ? 'bg-red-950/30 border-red-500/50' :
              result.verdict_type === 'UNCERTAIN' ? 'bg-amber-950/30 border-amber-500/50' :
              'bg-emerald-950/30 border-emerald-500/50'
            }`}>
              <button onClick={downloadPDF} className="absolute top-4 right-4 flex items-center gap-2 bg-zinc-900/90 hover:bg-zinc-800 px-4 py-2 rounded-xl text-xs font-bold transition border border-zinc-800">
                <Download size={14} /> Export Report
              </button>

              <div className="flex items-start md:items-center gap-6 flex-col md:flex-row">
                {(result.verdict_type === 'UNAVAILABLE' || result.status === 'unavailable') && <AlertTriangle className="w-16 h-16 md:w-20 md:h-20 text-amber-400 shrink-0" />}
                {result.verdict_type === 'FAKE' && <ShieldAlert className="w-16 h-16 md:w-20 md:h-20 text-red-500 shrink-0" />}
                {result.verdict_type === 'REAL' && <ShieldCheck className="w-16 h-16 md:w-20 md:h-20 text-emerald-500 shrink-0" />}
                {result.verdict_type === 'UNCERTAIN' && <AlertTriangle className="w-16 h-16 md:w-20 md:h-20 text-amber-500 shrink-0" />}

                <div className="flex-1">
                  <div className="flex gap-2 mb-2 flex-wrap">
                    <span className="px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider border" style={{
                      borderColor: (result.verdict_type === 'UNAVAILABLE' || result.status === 'unavailable') ? '#f59e0b' : result.verdict_type === 'FAKE' ? '#ef4444' : result.verdict_type === 'UNCERTAIN' ? '#f59e0b' : '#10b981',
                      color: (result.verdict_type === 'UNAVAILABLE' || result.status === 'unavailable') ? '#fde68a' : result.verdict_type === 'FAKE' ? '#fca5a5' : result.verdict_type === 'UNCERTAIN' ? '#fde68a' : '#a7f3d0'
                    }}>
                      PREDICTION: {result.status === 'unavailable' || result.verdict_type === 'UNAVAILABLE' ? 'MODEL UNAVAILABLE' : (result.prediction ? result.prediction.toUpperCase() : (result.is_fake ? 'FAKE' : 'REAL'))}
                    </span>
                    
                    {result.model_used && (
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        MODEL: {result.model_used}
                      </span>
                    )}
                  </div>

                  <h2 className="text-2xl md:text-4xl font-black">
                    {(result.verdict_type === 'UNAVAILABLE' || result.status === 'unavailable') ? (
                      result.video_metadata ? 'Video Verification Inconclusive' : 'Image Verification Model Unavailable'
                    ) : result.video_metadata ? (
                      result.is_fake ? 'Manipulated / AI-Generated Video Detected' : 'Likely Authentic Video'
                    ) : result.file_type && result.duration_formatted ? (
                      result.is_fake ? 'AI Generated Deepfake Audio Detected' : 'Real Human Voice Audio'
                    ) : (result.forensic_indicators || result.image_metadata) ? (
                      result.is_fake ? 'Manipulated / AI Deepfake Image Detected' : 'Real Genuine Camera Photo'
                    ) : (
                      <>
                        {result.verdict_type === 'FAKE' && 'Potential Fake News Detected'}
                        {result.verdict_type === 'REAL' && 'Likely Reliable Source / Real News'}
                        {result.verdict_type === 'UNCERTAIN' && 'UNCERTAIN – Additional Verification Recommended'}
                      </>
                    )}
                  </h2>
                  
                  {result.status === 'unavailable' ? (
                    <div className="mt-3 p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-200 text-sm">
                      <p className="font-medium">{result.message || "Video verification model is currently unavailable."}</p>
                    </div>
                  ) : (
                    <p className="text-zinc-300 text-lg mt-2">
                      Confidence Score: <span className="font-mono font-bold text-white">{(result.confidence * 100).toFixed(1)}%</span>
                      <span className="mx-3 text-zinc-600">|</span>
                      Model Used: <span className="font-semibold text-amber-400">{result.model_used || 'Trained Classifier'}</span>
                      {result.frames_analyzed !== undefined && (
                        <>
                          <span className="mx-3 text-zinc-600">|</span>
                          Frames Analyzed: <span className="font-mono font-bold text-amber-300">{result.frames_analyzed}</span>
                        </>
                      )}
                      {result.duration_formatted && (
                        <>
                          <span className="mx-3 text-zinc-600">|</span>
                          Duration: <span className="font-mono font-bold text-amber-300">{result.duration_formatted}</span>
                          <span className="mx-3 text-zinc-600">|</span>
                          Format: <span className="font-mono font-bold uppercase text-amber-300">{result.file_type}</span>
                        </>
                      )}
                      {result.overall_risk !== undefined && (
                        <>
                          <span className="mx-3 text-zinc-600">|</span>
                          Risk Level: <span className={`font-mono font-bold ${result.overall_risk >= 70 ? 'text-red-400' : 'text-emerald-400'}`}>{result.overall_risk}% ({result.verdict})</span>
                        </>
                      )}
                    </p>
                  )}

                  {result.media_url && result.file_type && (
                    <div className="mt-3">
                      <audio controls src={`${getApiBaseUrl()}${result.media_url}`} className="w-full h-10 rounded-lg border border-zinc-800" />
                    </div>
                  )}

                  {result.media_url && !result.file_type && !result.video_metadata && (
                    <div className="mt-4 flex flex-col md:flex-row gap-4 items-start">
                      <div className="max-w-xs rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 shrink-0">
                        <img src={`${getApiBaseUrl()}${result.media_url}`} alt="Uploaded Verification Preview" className="w-full h-auto object-cover max-h-56" />
                      </div>
                      {result.image_metadata && (
                        <div className="flex-1 bg-zinc-950/70 border border-zinc-800 p-4 rounded-xl text-xs space-y-2">
                          <h4 className="font-bold text-amber-400 uppercase tracking-wider text-[11px]">Uploaded Image Technical Metadata</h4>
                          <div className="grid grid-cols-2 gap-2 text-zinc-300">
                            <div><span className="text-zinc-500">Dimensions:</span> {result.image_metadata.dimensions}</div>
                            <div><span className="text-zinc-500">Format:</span> {result.image_metadata.format}</div>
                            <div><span className="text-zinc-500">Color Mode:</span> {result.image_metadata.mode}</div>
                            <div><span className="text-zinc-500">File Size:</span> {result.image_metadata.file_size_mb} MB</div>
                            {result.image_metadata.Make && <div><span className="text-zinc-500">Camera Make:</span> {result.image_metadata.Make}</div>}
                            {result.image_metadata.Model && <div><span className="text-zinc-500">Camera Model:</span> {result.image_metadata.Model}</div>}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Video Verification Player & Metadata Card */}
                  {result.media_url && result.video_metadata && (
                    <div className="mt-4 flex flex-col md:flex-row gap-4 items-start">
                      <div className="max-w-xs rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 shrink-0 w-full md:w-auto">
                        <video controls src={`${getApiBaseUrl()}${result.media_url}`} className="w-full h-auto object-cover max-h-56 rounded-lg" />
                      </div>
                      <div className="flex-1 bg-zinc-950/70 border border-zinc-800 p-4 rounded-xl text-xs space-y-3 w-full">
                        <h4 className="font-bold text-amber-400 uppercase tracking-wider text-[11px]">Analyzed Video Technical Metadata</h4>
                        <div className="grid grid-cols-2 gap-2 text-zinc-300">
                          <div><span className="text-zinc-500">File Name:</span> {result.video_metadata.file_name}</div>
                          <div><span className="text-zinc-500">File Size:</span> {result.video_metadata.file_size}</div>
                          <div><span className="text-zinc-500">Dimensions:</span> {result.video_metadata.dimensions || 'N/A'}</div>
                          <div><span className="text-zinc-500">Duration:</span> {result.video_metadata.duration_seconds || 0} sec</div>
                          <div><span className="text-zinc-500">FPS:</span> {result.video_metadata.fps || 'N/A'}</div>
                          <div><span className="text-zinc-500">Total Frames:</span> {result.video_metadata.total_frames || 'N/A'}</div>
                          <div><span className="text-zinc-500">Frames Analyzed:</span> {result.frames_analyzed || 0}</div>
                          <div><span className="text-zinc-500">Format:</span> {result.video_metadata.format}</div>
                        </div>
                        {result.extracted_frames && result.extracted_frames.length > 0 && (
                          <div className="pt-2 border-t border-zinc-800">
                            <span className="text-zinc-400 font-bold block mb-1.5 text-[11px]">Extracted Keyframe Analysis Previews:</span>
                            <div className="flex gap-2 overflow-x-auto pb-1">
                              {result.extracted_frames.map((kf: string, i: number) => (
                                <img key={i} src={`${getApiBaseUrl()}${kf}`} alt={`Keyframe ${i+1}`} className="h-16 w-24 object-cover rounded border border-zinc-800 shrink-0" />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-zinc-400 bg-zinc-950/60 border border-zinc-800/80 p-2.5 rounded-xl mt-3">
                    <strong>Disclaimer:</strong> FakeBuster is an AI-based decision-support tool. Predictions and confidence scores represent classification probabilities based on training patterns and do not guarantee absolute factual truth.
                  </p>
                </div>
              </div>

              {/* Trained ML Models Evaluation Table */}
              {result.all_metrics && Object.keys(result.all_metrics).length > 0 && (
                <div className="mt-6 pt-4 border-t border-zinc-800/80">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-amber-400 mb-2">Trained ML Classifiers Benchmark Comparison</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left text-zinc-300 border border-zinc-800/80 rounded-lg overflow-hidden">
                      <thead className="bg-zinc-950 text-zinc-400 uppercase text-[10px]">
                        <tr>
                          <th className="p-2">Model</th>
                          <th className="p-2 text-right">Accuracy</th>
                          <th className="p-2 text-right">Precision</th>
                          <th className="p-2 text-right">Recall</th>
                          <th className="p-2 text-right">F1-Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-800/60 font-mono">
                        {Object.entries(result.all_metrics).map(([mName, mVal]: [string, any]) => (
                          <tr key={mName} className={mName === result.model_used ? 'bg-amber-500/10 text-amber-300 font-bold' : ''}>
                            <td className="p-2 font-sans font-semibold flex items-center gap-1">
                              {mName} {mName === result.model_used && <span className="text-[9px] bg-amber-500 text-black px-1.5 py-0.5 rounded font-black uppercase">Selected</span>}
                            </td>
                            <td className="p-2 text-right">{(mVal.accuracy * 100).toFixed(1)}%</td>
                            <td className="p-2 text-right">{(mVal.precision * 100).toFixed(1)}%</td>
                            <td className="p-2 text-right">{(mVal.recall * 100).toFixed(1)}%</td>
                            <td className="p-2 text-right font-bold">{(mVal.f1_score * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* Fake News Risk Meter - Suppressed for image / unavailable status */}
            {result.clickbait !== undefined && result.status !== 'unavailable' && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 space-y-6">
                <div className="flex justify-between items-center border-b border-zinc-800/60 pb-3">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-amber-400">1. Supporting Risk Indicators (Heuristic Measures)</h3>
                  <span className="text-xs text-zinc-500">Heuristic Signal Indicators</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Clickbait Probability</span>
                      <span className="text-zinc-200">{result.clickbait}%</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-lg h-2.5 overflow-hidden border border-zinc-800">
                      <div className="bg-amber-500 h-full rounded-lg" style={{ width: `${result.clickbait}%` }} />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Source Reliability</span>
                      <span className="text-zinc-200">{result.source_reliability}%</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-lg h-2.5 overflow-hidden border border-zinc-800">
                      <div className="bg-blue-500 h-full rounded-lg" style={{ width: `${result.source_reliability}%` }} />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Emotional Language Intensity</span>
                      <span className="text-zinc-200">{result.emotional_language}%</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-lg h-2.5 overflow-hidden border border-zinc-800">
                      <div className="bg-rose-500 h-full rounded-lg" style={{ width: `${result.emotional_language}%` }} />
                    </div>
                  </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-zinc-400">Evidence Quality</span>
                    <span className="text-zinc-200">{result.evidence_quality}%</span>
                  </div>
                  <div className="w-full bg-zinc-950 rounded-lg h-2.5 overflow-hidden border border-zinc-800">
                    <div className="bg-emerald-500 h-full rounded-lg" style={{ width: `${result.evidence_quality}%` }} />
                  </div>
                </div>
              </div>
            </div>
            )}

            {/* Language & Translation Info */}
            {result.detected_language && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6">
                <div className="flex items-center gap-2 text-sm font-bold text-zinc-300 mb-2">
                  <Globe size={18} className="text-amber-400" />
                  Language Detection: <span className="text-amber-400 font-mono">{result.detected_language}</span>
                </div>
                {result.was_translated && (
                  <div className="p-4 bg-zinc-950/80 border border-zinc-800 rounded-xl mt-3 text-sm text-zinc-300">
                    <span className="text-xs font-bold text-amber-400 uppercase tracking-wider block mb-1">Translated to English for AI Analysis:</span>
                    "{result.translated_text}"
                  </div>
                )}
              </div>
            )}

            {/* Extracted Factual Claims */}
            {result.claims && result.claims.length > 0 && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6">
                <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-300 mb-3">Extracted Factual Claims</h3>
                <div className="space-y-2">
                  {result.claims.map((c: string, idx: number) => (
                    <div key={idx} className="p-3.5 bg-zinc-950/60 border border-zinc-800/80 rounded-xl text-sm text-zinc-300 flex items-start gap-3">
                      <span className="text-amber-400 font-bold shrink-0">#{idx + 1}</span>
                      <p>{c}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* External Fact-Checks (Only for Text/URL or when fact checks exist) */}
            {(result.content_type === 'text' || result.content_type === 'url' || (result.fact_checks && result.fact_checks.length > 0)) && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6">
                <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-300 mb-3">Reputable Fact-Checks</h3>
                {result.fact_checks && result.fact_checks.length > 0 ? (
                  <div className="space-y-3">
                    {result.fact_checks.map((fc: any, i: number) => (
                      <div key={i} className="p-4 bg-zinc-950/60 border border-zinc-800 rounded-xl space-y-1.5">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-bold text-amber-400">{fc.organization}</span>
                          <span className="px-2 py-0.5 rounded bg-zinc-800 font-mono text-zinc-300">{fc.verdict}</span>
                        </div>
                        <p className="text-sm font-semibold text-white">{fc.summary}</p>
                        <a href={fc.source_link} target="_blank" rel="noreferrer" className="text-xs text-blue-400 hover:underline inline-flex items-center gap-1">
                          Read Fact-Check <ExternalLink size={12} />
                        </a>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-zinc-500 italic p-4 bg-zinc-950/40 rounded-xl border border-zinc-800/50">
                    {result.fact_check_message || "No matching fact-check found."}
                  </p>
                )}
              </div>
            )}

            {/* Web Evidence (Only for Text/URL or when evidence exists) */}
            {(result.content_type === 'text' || result.content_type === 'url' || (result.evidence && result.evidence.length > 0)) && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6">
                <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-300 mb-3">Retrieved Evidence Sources</h3>
                {result.evidence && result.evidence.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.evidence.map((ev: any, i: number) => (
                      <div key={i} className="p-4 bg-zinc-950/60 border border-zinc-800 rounded-xl space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-bold text-zinc-400">{ev.source}</span>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${ev.category === 'Contradicting' ? 'bg-red-950 text-red-400' : 'bg-emerald-950 text-emerald-400'}`}>{ev.category}</span>
                        </div>
                        <h4 className="text-sm font-bold text-white line-clamp-1">{ev.title}</h4>
                        <p className="text-xs text-zinc-400 line-clamp-2">{ev.snippet}</p>
                        <a href={ev.url} target="_blank" rel="noreferrer" className="text-xs text-blue-400 hover:underline inline-flex items-center gap-1">
                          Source Link <ExternalLink size={12} />
                        </a>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-zinc-500 italic p-4 bg-zinc-950/40 rounded-xl border border-zinc-800/50">
                    {result.evidence_message || "No relevant evidence found."}
                  </p>
                )}
              </div>
            )}

            {/* Source Credibility Card */}
            {result.source_info && (
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 flex justify-between items-center">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">Source Credibility</h3>
                  <p className="text-lg font-bold text-white mt-1">{result.source_info.source_name} ({result.source_info.domain})</p>
                  <p className="text-sm text-zinc-400">{result.source_info.credibility_status}</p>
                </div>
                <div className="text-right">
                  <span className="text-xs text-zinc-500 font-semibold uppercase">Reliability Score</span>
                  <p className="text-3xl font-black text-amber-400">{result.source_info.reliability_score}/100</p>
                </div>
              </div>
            )}

            {/* Limitations Notice */}
            {result.limitations && (
              <div className="p-4 bg-zinc-950/80 border border-zinc-800 rounded-2xl text-xs text-zinc-400 flex items-start gap-3">
                <Info size={18} className="text-amber-400 shrink-0 mt-0.5" />
                <p><strong className="text-zinc-300">System Disclaimer & Limitations:</strong> {result.limitations}</p>
              </div>
            )}

          </motion.div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-zinc-900/40 border border-zinc-800 p-6 md:p-8 rounded-2xl shadow-2xl space-y-6">
            <h2 className="text-2xl font-black text-white flex items-center gap-2">
              <HistoryIcon className="text-amber-400" /> Recent Verification Logs
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-zinc-300">
                <thead className="text-xs uppercase bg-zinc-950 text-zinc-400 border-b border-zinc-800">
                  <tr>
                    <th className="p-3">ID</th>
                    <th className="p-3">Type</th>
                    <th className="p-3">Claim / Title</th>
                    <th className="p-3">Verdict</th>
                    <th className="p-3">Risk</th>
                    <th className="p-3">Language</th>
                    <th className="p-3">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60">
                  {historyList.map((h, i) => (
                    <tr key={i} className="hover:bg-zinc-950/40 transition">
                      <td className="p-3 font-mono text-xs text-amber-400">{h.verification_id}</td>
                      <td className="p-3 uppercase text-xs font-bold">{h.content_type}</td>
                      <td className="p-3 max-w-xs truncate">{h.title || h.content_snippet}</td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded text-xs font-black uppercase ${
                          h.verdict === 'FAKE' ? 'bg-red-950 text-red-400' :
                          h.verdict === 'UNCERTAIN' ? 'bg-amber-950 text-amber-400' :
                          'bg-emerald-950 text-emerald-400'
                        }`}>
                          {h.verdict}
                        </span>
                      </td>
                      <td className="p-3 font-mono">{h.overall_risk}%</td>
                      <td className="p-3">{h.language}</td>
                      <td className="p-3 text-xs text-zinc-500">{h.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Analytics Dashboard */}
        {activeTab === 'dashboard' && stats && (
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-zinc-900/40 border border-zinc-800 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
                <BarChart2 className="w-8 h-8 mx-auto text-amber-400 mb-2" />
                <h3 className="text-zinc-400 font-semibold text-xs uppercase">Total Analyzed</h3>
                <p className="text-4xl font-black text-white mt-2">{stats.total_analyzed}</p>
              </div>

              <div className="bg-zinc-900/40 border border-red-500/30 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
                <ShieldAlert className="w-8 h-8 mx-auto text-red-400 mb-2" />
                <h3 className="text-red-300 font-semibold text-xs uppercase">Fake News Detected</h3>
                <p className="text-4xl font-black text-red-400 mt-2">{stats.fake_detected}</p>
              </div>

              <div className="bg-zinc-900/40 border border-emerald-500/30 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
                <ShieldCheck className="w-8 h-8 mx-auto text-emerald-400 mb-2" />
                <h3 className="text-emerald-300 font-semibold text-xs uppercase">Real / Reliable</h3>
                <p className="text-4xl font-black text-emerald-400 mt-2">{stats.real_detected}</p>
              </div>

              <div className="bg-zinc-900/40 border border-amber-500/30 p-6 rounded-2xl text-center shadow-lg relative overflow-hidden">
                <AlertTriangle className="w-8 h-8 mx-auto text-amber-400 mb-2" />
                <h3 className="text-amber-300 font-semibold text-xs uppercase">Uncertain Claims</h3>
                <p className="text-4xl font-black text-amber-400 mt-2">{stats.uncertain_detected || 0}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-zinc-900/40 border border-zinc-800 p-6 rounded-2xl space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-300">Input Type Breakdown</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>Text Articles</span>
                      <span>{stats.input_breakdown?.text || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-amber-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.input_breakdown?.text / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>News URLs</span>
                      <span>{stats.input_breakdown?.url || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.input_breakdown?.url / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>Image Media</span>
                      <span>{stats.input_breakdown?.image || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-purple-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.input_breakdown?.image / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>Video Clips</span>
                      <span>{stats.input_breakdown?.video || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-emerald-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.input_breakdown?.video / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-zinc-900/40 border border-zinc-800 p-6 rounded-2xl space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-300">Language Distribution</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>English</span>
                      <span>{stats.language_breakdown?.English || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-amber-400 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.language_breakdown?.English / stats.total_analyzed) * 100 : 100}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>Hindi (हिंदी)</span>
                      <span>{stats.language_breakdown?.Hindi || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-rose-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.language_breakdown?.Hindi / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span>Telugu (తెలుగు)</span>
                      <span>{stats.language_breakdown?.Telugu || 0}</span>
                    </div>
                    <div className="w-full bg-zinc-950 rounded-full h-2">
                      <div className="bg-indigo-500 h-2 rounded-full" style={{ width: `${stats.total_analyzed > 0 ? (stats.language_breakdown?.Telugu / stats.total_analyzed) * 100 : 0}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

      </div>
    </div>
  );
}
