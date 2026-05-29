import { useState, useEffect, useRef, useCallback, FC } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Play, Shield, Activity, AlertTriangle, RefreshCw, Layers, BarChart3, TrendingUp, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import './index.css';

// ── Types ───────────────────────────────────────────────────────────────
interface GraphNode {
  id: string; status: string; risk_score: number;
  is_fraud: boolean; in_loop: boolean; degree: number;
  reasons?: string[]; x?: number; y?: number;
}
interface GraphLink {
  source: string | GraphNode; target: string | GraphNode;
  tx_id: string; amount: number; type: string;
  is_alert: boolean; in_loop: boolean; risk_score: number;
  review_status?: string;
}
interface GraphData { nodes: GraphNode[]; links: GraphLink[]; }
interface AlertData {
  case_id: string; tx_id: string; sender: string; receiver: string;
  risk_score: number; status: string; tx_type: string; amount: number;
  created_at?: string;
}

const API = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/inference';

const PATTERN_META: Record<string, any> = {
  Circular: { icon: RefreshCw, color: 'text-red-400', label: 'Circular Loop' },
  Layering: { icon: Layers, color: 'text-amber-400', label: 'Rapid Layering' },
  Structuring: { icon: BarChart3, color: 'text-purple-400', label: 'Structuring' },
  Normal: { icon: TrendingUp, color: 'text-slate-400', label: 'Normal Traffic' },
};

const NODE_COLORS: Record<string, string> = {
  STABLE: '#06b6d4', SUSPICIOUS: '#f59e0b', CRITICAL: '#ef4444', CONFIRMED_FRAUD: '#dc2626'
};

export default function App() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [cases, setCases] = useState<AlertData[]>([]);

  const [patterns, setPatterns] = useState<Record<string, number>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState<'feed' | 'queue'>('feed');
  const [selectedCase, setSelectedCase] = useState<AlertData | null>(null);

  const fullGraphRef = useRef<GraphData>({ nodes: [], links: [] });
  const [progress, setProgress] = useState({ processed: 0, total: 0 });
  const [currentConfidence, setCurrentConfidence] = useState(0);

  const graphRef = useRef<any>(undefined);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      setDimensions({ width: entries[0].contentRect.width, height: entries[0].contentRect.height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const fetchGraph = () => {
    fetch(`${API}/api/graph`).then(r => r.json()).then(data => {
      fullGraphRef.current = data;
      setGraphData({ nodes: data.nodes, links: [] }); // Start with empty links
      setIsLoaded(true);
    });
  };

  const fetchCases = () => {
    fetch(`${API}/api/cases`).then(r => r.json()).then(data => {
      setCases(data);
    });
  };

  useEffect(() => {
    fetchGraph();
    setTimeout(() => graphRef.current?.zoomToFit(400, 60), 500);
  }, []);

  // Poll for cases when running
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(fetchCases, 2000);
      return () => clearInterval(interval);
    }
  }, [isRunning]);

  const runDemo = () => {
    if (isRunning) return;
    setIsRunning(true);
    setAlerts([]);
    setCases([]);
    setPatterns({});
    setSelectedCase(null);
    setCurrentConfidence(0);

    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'inference_start') {
        setProgress({ processed: 0, total: msg.data.total });
      } else if (msg.type === 'alert' || msg.type === 'transaction') {
        const d = msg.data;
        if (msg.type === 'alert') {
          setCurrentConfidence(d.risk_score);
          setAlerts(prev => [d, ...prev].slice(0, 50));
          setPatterns(prev => ({ ...prev, [d.tx_type]: (prev[d.tx_type] || 0) + 1 }));
        }
        
        // Add the link to the graph incrementally
        setGraphData(prev => {
          const nodes = [...prev.nodes];
          const links = [...prev.links];
          
          if (msg.type === 'alert') {
            const sender = nodes.find(n => n.id === d.sender);
            if (sender) { sender.status = d.status; sender.is_fraud = true; }
          }
          
          // Find link in full graph and add it if not already there
          const fullLink = fullGraphRef.current.links.find(l => l.tx_id === d.tx_id);
          if (fullLink && !links.find(l => (typeof l.source === 'object' ? (l as any).tx_id : l.tx_id) === d.tx_id)) {
            const newLink = { ...fullLink };
            if (msg.type === 'alert') {
              newLink.is_alert = true;
              newLink.risk_score = d.risk_score;
            }
            links.push(newLink);
          }
          return { nodes, links };
        });

        if (msg.type === 'alert') {
          // Zoom to alert
          setTimeout(() => {
              const senderNode = graphData.nodes.find(n => n.id === d.sender);
              if (senderNode && senderNode.x && senderNode.y && graphRef.current) {
                  graphRef.current.centerAt(senderNode.x, senderNode.y, 800);
                  graphRef.current.zoom(3, 800);
              }
          }, 100);
        }

      } else if (msg.type === 'progress') {
        setProgress(msg.data);
      } else if (msg.type === 'inference_complete') {
        setIsRunning(false);
        fetchCases();
        setTimeout(() => graphRef.current?.zoomToFit(1000, 60), 1000);
      }
    };
    ws.onclose = () => setIsRunning(false);
  };

  const handleReview = async (caseId: string, decision: string) => {
    await fetch(`${API}/api/cases/${caseId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision })
    });
    setSelectedCase(null);
    fetchCases();
    fetchGraph(); // Refresh graph to show updated status colors
  };

  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
    const size = Math.max(3, Math.min(8, Math.sqrt(node.degree || 1) * 1.5 + 2));
    let color = NODE_COLORS[node.status] || NODE_COLORS.STABLE;

    if (node.status === 'CONFIRMED_FRAUD' || node.is_fraud) {
      const grad = ctx.createRadialGradient(node.x, node.y, size, node.x, node.y, size * 4);
      grad.addColorStop(0, color + '60');
      grad.addColorStop(1, color + '00');
      ctx.beginPath(); ctx.arc(node.x, node.y, size * 4, 0, 2 * Math.PI);
      ctx.fillStyle = grad; ctx.fill();
    }

    ctx.beginPath(); ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 0.5; ctx.stroke();

    if (globalScale > 2 || (node.is_fraud && globalScale > 1.2)) {
      ctx.font = `${node.is_fraud ? 'bold' : 'normal'} ${Math.max(12 / globalScale, 3)}px Inter`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillStyle = node.is_fraud ? '#fff' : '#94a3b8';
      ctx.fillText(node.id, node.x, node.y + size + 2);
    }
  }, []);

  return (
    <div className="h-full flex flex-col grid-bg text-slate-200">
      <header className="h-16 flex items-center justify-between px-6 border-b border-white/5 bg-black/40 backdrop-blur-md z-20">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Shield size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight text-white flex items-center gap-2">
              FCCI <span className="text-cyan-400">TGNN</span>
              {isRunning && <span className="flex h-2 w-2 relative ml-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span><span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span></span>}
            </h1>
            <p className="text-xs text-slate-400 font-medium">Real-Time k-hop Inference (Neo4j Aura)</p>
          </div>
        </div>
        
        <button onClick={runDemo} disabled={!isLoaded || isRunning} className={`start-button px-6 py-2.5 rounded-xl font-bold tracking-wide flex items-center gap-2 ${!isLoaded || isRunning ? 'bg-white/5 text-slate-500' : 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg'}`}>
          {isRunning ? <Activity size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
          {isRunning ? 'INFERENCE ACTIVE' : 'START DEMO'}
        </button>
      </header>

      <div className="flex-1 flex overflow-hidden relative">
        <main ref={containerRef} className="flex-1 relative">
          {isLoaded && (
            <ForceGraph2D
              ref={graphRef as any}
              graphData={graphData}
              width={dimensions.width}
              height={dimensions.height}
              backgroundColor="transparent"
              nodeId="id"
              nodeCanvasObject={nodeCanvasObject}
              linkColor={(link: any) => link.is_alert ? 'rgba(239, 68, 68, 0.8)' : 'rgba(6, 182, 212, 0.3)'}
              linkWidth={(link: any) => link.is_alert ? 2 : 1}
              linkDirectionalParticles={(link: any) => link.is_alert ? 3 : 1}
              linkDirectionalParticleWidth={(link: any) => link.is_alert ? 2 : 1}
              linkDirectionalParticleSpeed={0.005}
              linkDirectionalParticleColor={(link: any) => link.is_alert ? '#f87171' : '#67e8f9'}
            />
          )}
        </main>

        <aside className="w-[400px] flex flex-col border-l border-white/5 bg-black/40 backdrop-blur-md z-10">
          
          <div className="flex border-b border-white/5">
            <button onClick={() => setActiveTab('feed')} className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider border-b-2 transition-colors ${activeTab === 'feed' ? 'border-cyan-500 text-white' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>Live Feed</button>
            <button onClick={() => setActiveTab('queue')} className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-2 ${activeTab === 'queue' ? 'border-cyan-500 text-white' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
              Case Queue {cases.length > 0 && <span className="bg-red-500 text-white text-[10px] px-1.5 rounded-full">{cases.length}</span>}
            </button>
          </div>

          <div className="flex-1 overflow-hidden flex flex-col">
            {activeTab === 'feed' ? (
              <div className="p-4 flex-1 overflow-y-auto space-y-3">
                {alerts.map((a, i) => (
                  <div key={i} className="glass-card p-3 rounded-lg border-l-2 border-l-red-500 animate-slide-up">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-[10px] font-bold text-red-400 uppercase">{a.tx_type}</span>
                      <span className="text-[10px] font-mono text-slate-400">{a.risk_score}% Risk</span>
                    </div>
                    <div className="text-[11px] font-semibold text-white truncate">{a.sender} → {a.receiver}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex-1 overflow-hidden flex flex-col">
                {selectedCase ? (
                  <div className="p-6 flex-1 flex flex-col">
                    <button onClick={() => setSelectedCase(null)} className="text-xs text-slate-400 hover:text-white mb-4">← Back to Queue</button>
                    <h2 className="text-lg font-black text-white mb-1">Case Review: {selectedCase.tx_id}</h2>
                    <div className="text-xs text-red-400 font-bold uppercase mb-6">{selectedCase.tx_type} Detected</div>
                    
                    <div className="glass-card p-4 rounded-xl mb-6 space-y-3">
                      <div className="flex justify-between"><span className="text-slate-400">Sender</span><span className="text-white">{selectedCase.sender}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Receiver</span><span className="text-white">{selectedCase.receiver}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Amount</span><span className="text-cyan-400 font-mono">${selectedCase.amount}</span></div>
                      <div className="flex justify-between border-t border-white/5 pt-3"><span className="text-slate-400">TGNN Risk Score</span><span className="text-red-400 font-bold font-mono">{selectedCase.risk_score}%</span></div>
                    </div>

                    <div className="mt-auto space-y-3">
                      <button onClick={() => handleReview(selectedCase.case_id, 'REJECTED')} className="w-full py-3 rounded-lg bg-red-500/20 text-red-400 font-bold flex items-center justify-center gap-2 hover:bg-red-500/30 transition-colors border border-red-500/30">
                        <XCircle size={16} /> Confirm Fraud (Reject)
                      </button>
                      <button onClick={() => handleReview(selectedCase.case_id, 'APPROVED')} className="w-full py-3 rounded-lg bg-emerald-500/10 text-emerald-400 font-bold flex items-center justify-center gap-2 hover:bg-emerald-500/20 transition-colors border border-emerald-500/20">
                        <CheckCircle size={16} /> False Positive (Approve)
                      </button>
                      <button onClick={() => handleReview(selectedCase.case_id, 'ESCALATED')} className="w-full py-3 rounded-lg bg-amber-500/10 text-amber-400 font-bold flex items-center justify-center gap-2 hover:bg-amber-500/20 transition-colors border border-amber-500/20">
                        <AlertCircle size={16} /> Escalate to Tier 2
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 flex-1 overflow-y-auto space-y-3">
                    {cases.length === 0 && !isRunning && <div className="text-center text-slate-500 text-sm mt-10">No pending cases.</div>}
                    {cases.map((c, i) => (
                      <div key={i} onClick={() => setSelectedCase(c)} className="glass-card p-4 rounded-lg border border-white/5 hover:border-cyan-500/50 cursor-pointer transition-colors group">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-xs font-bold text-red-400">{c.risk_score}% RISK</span>
                          <span className="text-[10px] text-slate-500">{c.tx_id}</span>
                        </div>
                        <div className="text-sm font-semibold text-white mb-1 group-hover:text-cyan-400 transition-colors">{c.tx_type}</div>
                        <div className="text-xs text-slate-400">{c.sender} → {c.receiver}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
