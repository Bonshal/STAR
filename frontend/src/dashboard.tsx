import { useState, useEffect, useRef, useCallback, type FC, type DragEvent } from 'react';
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d';
import {
  Upload, Play, Search, Shield, Activity, AlertTriangle,
  RefreshCw, Layers, BarChart3, Clock, TrendingUp, X, Crosshair, Eye
} from 'lucide-react';
import './index.css';

// ── Types ───────────────────────────────────────────────────────────────

interface GraphNode {
  id: string; status: string; risk_score: number;
  is_fraud: boolean; in_loop: boolean; degree: number;
  reasons?: string[];
  x?: number; y?: number;
  __highlighted?: boolean;
}
interface GraphLink {
  source: string | GraphNode; target: string | GraphNode;
  tx_id: string; amount: number; type: string;
  is_alert: boolean; in_loop: boolean; risk_score: number;
}
interface GraphData { nodes: GraphNode[]; links: GraphLink[]; }
interface AlertData {
  tx_id: string; sender: string; receiver: string;
  risk_score: number; status: string; tx_type: string; amount: number;
}
interface PatternCount { [key: string]: number; }

const API = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/inference';
const NEO4J_URI = 'neo4j://127.0.0.1:7687';

const PATTERN_META: Record<string, { icon: typeof RefreshCw; color: string; label: string; border: string }> = {
  Circular:    { icon: RefreshCw,   color: 'text-red-400',    label: 'Circular Round-Tripping', border: 'glow-border-red' },
  Layering:    { icon: Layers,      color: 'text-amber-400',  label: 'Rapid Layering',          border: 'glow-border-amber' },
  Structuring: { icon: BarChart3,   color: 'text-purple-400', label: 'Structuring (Smurfing)',   border: 'glow-border-purple' },
  Normal:      { icon: TrendingUp,  color: 'text-slate-400',  label: 'Anomalous Normal',        border: 'glow-border-cyan' },
};

const NODE_COLORS: Record<string, string> = {
  STABLE: '#06b6d4',
  SUSPICIOUS: '#f59e0b',
  CRITICAL: '#ef4444',
};

// ── Sub-components ──────────────────────────────────────────────────────

const PatternCard: FC<{
  type: string; count: number; total: number;
  onClick: () => void; active: boolean;
}> = ({ type, count, total, onClick, active }) => {
  const meta = PATTERN_META[type] || PATTERN_META.Normal;
  const Icon = meta.icon;
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;

  return (
    <button
      onClick={onClick}
      className={`pattern-card w-full text-left p-3.5 rounded-xl glass-card ${meta.border} ${
        active ? 'ring-1 ring-white/20' : ''
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon size={14} className={meta.color} />
          <span className="text-xs font-semibold text-slate-300">{meta.label}</span>
        </div>
        <span className={`text-lg font-bold font-mono ${meta.color}`}>{count}</span>
      </div>
      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${NODE_COLORS.SUSPICIOUS}, ${NODE_COLORS.CRITICAL})`,
        }} />
      </div>
      <p className="text-[10px] text-slate-500 mt-1.5 font-medium">{pct}% of total alerts</p>
    </button>
  );
};

const AlertItem: FC<{ alert: AlertData; index: number }> = ({ alert, index }) => (
  <div className="animate-slide-up p-3 glass-card rounded-lg mb-2" style={{ animationDelay: `${index * 30}ms` }}>
    <div className="flex items-center justify-between mb-1.5">
      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md uppercase tracking-wider ${
        alert.status === 'CRITICAL'
          ? 'bg-red-500/15 text-red-400 border border-red-500/20'
          : 'bg-amber-500/15 text-amber-400 border border-amber-500/20'
      }`}>{alert.status}</span>
      <span className="text-[10px] text-slate-500 font-mono">{alert.risk_score.toFixed(1)}%</span>
    </div>
    <p className="text-xs font-semibold text-slate-200 truncate">{alert.sender} → {alert.receiver}</p>
    <div className="flex items-center gap-3 mt-1.5 text-[10px] text-slate-500">
      <span className="font-mono">${alert.amount.toLocaleString()}</span>
      <span className="px-1.5 py-0.5 rounded bg-white/5 font-medium">{alert.tx_type}</span>
    </div>
  </div>
);

// ── Main Dashboard ──────────────────────────────────────────────────────

export default function App() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [patterns, setPatterns] = useState<PatternCount>({});
  const [loopNodes, setLoopNodes] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [progress, setProgress] = useState({ processed: 0, total: 0 });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterPattern, setFilterPattern] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [stats, setStats] = useState({ nodes: 0, edges: 0, flagged: 0 });

  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
  const wsRef = useRef<WebSocket | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // ── Resize observer ──
  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // ── Try loading existing graph on mount ──
  useEffect(() => {
    fetch(`${API}/api/graph`)
      .then((r) => r.json())
      .then((data: GraphData) => {
        if (data.nodes.length > 0) {
          setGraphData(data);
          setIsLoaded(true);
          setStats({ nodes: data.nodes.length, edges: data.links.length, flagged: data.nodes.filter(n => n.is_fraud).length });
          setTimeout(() => graphRef.current?.zoomToFit(400, 60), 500);
        }
      })
      .catch(() => {});
  }, []);

  // ── Upload CSV ──
  const handleFile = async (file: File) => {
    setUploadName(file.name);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
      const data: GraphData = await res.json();
      setGraphData(data);
      setIsLoaded(true);
      setAlerts([]);
      setPatterns({});
      setLoopNodes(0);
      setFilterPattern(null);
      setStats({ nodes: data.nodes.length, edges: data.links.length, flagged: 0 });
      setTimeout(() => graphRef.current?.zoomToFit(400, 60), 500);
    } catch (e) {
      console.error('Upload failed:', e);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith('.csv')) handleFile(file);
  };

  // ── Run Inference ──
  const runInference = () => {
    if (!isLoaded || isRunning) return;
    setIsRunning(true);
    setAlerts([]);
    setPatterns({});
    setLoopNodes(0);
    setFilterPattern(null);

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case 'inference_start':
          setProgress({ processed: 0, total: msg.data.total });
          break;

        case 'alert': {
          const d: AlertData = msg.data;
          setGraphData((prev) => {
            const nextNodes = [...prev.nodes];
            const senderNode = nextNodes.find((n) => n.id === d.sender);
            if (senderNode) {
              senderNode.status = d.status;
              senderNode.risk_score = d.risk_score;
              senderNode.is_fraud = true;
              if (!senderNode.reasons) senderNode.reasons = [];
              if (!senderNode.reasons.includes(d.tx_type)) {
                senderNode.reasons.push(d.tx_type);
              }
            }

            const nextLinks = [...prev.links];
            const alertLink = nextLinks.find((l) => {
              const lid = typeof l.source === 'object' ? (l as any).tx_id : l.tx_id;
              return lid === d.tx_id;
            });
            if (alertLink) {
              alertLink.is_alert = true;
              alertLink.risk_score = d.risk_score;
            }

            return { nodes: nextNodes, links: nextLinks };
          });
          setAlerts((prev) => [d, ...prev].slice(0, 200));
          setPatterns((prev) => ({ ...prev, [d.tx_type]: (prev[d.tx_type] || 0) + 1 }));
          setStats((prev) => ({ ...prev, flagged: prev.flagged + 1 }));
          break;
        }

        case 'progress':
          setProgress(msg.data);
          break;

        case 'inference_complete':
          setIsRunning(false);
          setProgress({ processed: msg.data.processed, total: msg.data.processed });
          setLoopNodes(msg.data.loop_nodes || 0);
          // Refresh full graph to get loop data but mutate to preserve all physics engine object references
          fetch(`${API}/api/graph`).then(r => r.json()).then((data: GraphData) => {
            setGraphData(prev => {
              const nextNodes = [...prev.nodes];
              const prevNodeMap = new Map(nextNodes.map(n => [n.id, n]));
              data.nodes.forEach(n => {
                const p = prevNodeMap.get(n.id);
                if (p) {
                  p.risk_score = n.risk_score;
                  p.is_fraud = n.is_fraud;
                  p.in_loop = n.in_loop;
                  p.status = n.status;
                  p.reasons = n.reasons;
                } else {
                  nextNodes.push(n);
                }
              });

              const nextLinks = [...prev.links];
              const prevLinkMap = new Map(nextLinks.map(l => {
                const lid = typeof l.source === 'object' ? (l as any).tx_id : l.tx_id;
                return [lid, l];
              }));
              data.links.forEach(l => {
                const p = prevLinkMap.get(l.tx_id);
                if (p) {
                  p.is_alert = l.is_alert;
                  p.in_loop = l.in_loop;
                  p.risk_score = l.risk_score;
                } else {
                  nextLinks.push(l);
                }
              });

              return { nodes: nextNodes, links: nextLinks };
            });
            setStats({ nodes: data.nodes.length, edges: data.links.length, flagged: data.nodes.filter(n => n.is_fraud).length });
          });
          break;

        case 'error':
          setIsRunning(false);
          console.error('Inference error:', msg.data.message);
          break;
      }
    };

    ws.onclose = () => setIsRunning(false);
    ws.onerror = () => setIsRunning(false);
  };

  // ── Search ──
  const handleSearch = () => {
    if (!searchTerm.trim()) return;
    const node = graphData.nodes.find(n => n.id.toLowerCase().includes(searchTerm.toLowerCase()));
    if (node && node.x != null && node.y != null) {
      graphRef.current?.centerAt(node.x, node.y, 600);
      graphRef.current?.zoom(5, 600);
    }
  };

  // ── Graph Rendering Callbacks ──
  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
    const deg = node.degree || 1;
    const size = Math.max(2.5, Math.min(7, Math.sqrt(deg) * 1.5 + 1.5));
    let color = NODE_COLORS[node.status] || NODE_COLORS.STABLE;
    if (node.in_loop) color = '#8b5cf6';

    // Filter dimming
    if (filterPattern) {
      const isRelevant = node.is_fraud;
      if (!isRelevant) {
        ctx.globalAlpha = 0.08;
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
        ctx.fillStyle = '#475569';
        ctx.fill();
        ctx.globalAlpha = 1;
        return;
      }
    }

    // Outer glow for flagged
    if (node.is_fraud) {
      const grad = ctx.createRadialGradient(node.x, node.y, size * 0.5, node.x, node.y, size * 3.5);
      grad.addColorStop(0, color + '50');
      grad.addColorStop(1, color + '00');
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Node body
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 0.4;
    ctx.stroke();

    // Label
    if (globalScale > 3 || (node.is_fraud && globalScale > 1.5)) {
      const fs = Math.max(10 / globalScale, 2.5);
      ctx.font = `${node.is_fraud ? '700' : '500'} ${fs}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = node.is_fraud ? '#ffffff' : '#94a3b8';
      ctx.fillText(node.id, node.x, node.y + size + 1.5);
    }
  }, [filterPattern]);

  const linkColor = useCallback((link: any) => {
    if (link.in_loop) return 'rgba(139, 92, 246, 0.85)';
    if (link.is_alert) return 'rgba(239, 68, 68, 0.8)';
    return 'rgba(100, 160, 200, 0.55)';
  }, []);

  const linkWidth = useCallback((link: any) => {
    if (link.in_loop) return 2.5;
    if (link.is_alert) return 2;
    return 1;
  }, []);

  const linkParticles = useCallback((link: any) => {
    if (link.in_loop) return 3;
    if (link.is_alert) return 2;
    return 0;
  }, []);

  const totalAlerts = Object.values(patterns).reduce((a, b) => a + b, 0);
  const pctDone = progress.total > 0 ? Math.round((progress.processed / progress.total) * 100) : 0;

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-void)' }}>
      {/* ─── Header ─── */}
      <header className="h-14 flex items-center justify-between px-5 border-b"
        style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-primary)' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Shield size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white">IFFT Command Center</h1>
            <p className="text-[10px] text-slate-500 font-medium">Intelligent Fund Flow Tracking</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {isLoaded && (
            <div className="flex items-center gap-5 text-[11px] font-mono text-slate-400">
              <span><span className="text-cyan-400 font-bold">{stats.nodes}</span> nodes</span>
              <span><span className="text-slate-300 font-bold">{stats.edges.toLocaleString()}</span> edges</span>
              <span><span className="text-red-400 font-bold">{stats.flagged}</span> flagged</span>
            </div>
          )}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold ${
            isRunning
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : isLoaded
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                : 'bg-slate-500/10 text-slate-400 border border-slate-500/20'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${
              isRunning ? 'bg-emerald-400 animate-pulse' : isLoaded ? 'bg-cyan-400' : 'bg-slate-500'
            }`} />
            {isRunning ? 'INFERENCE ACTIVE' : isLoaded ? 'GRAPH LOADED' : 'AWAITING DATA'}
          </div>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <div className="flex-1 flex overflow-hidden">

        {/* ─── Left Sidebar ─── */}
        <aside className="w-[260px] flex flex-col border-r"
          style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-primary)' }}>

          <div className="p-4 space-y-3 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            {/* Upload Zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={onDrop}
              className={`relative rounded-xl p-4 border-2 border-dashed transition-all cursor-pointer text-center ${
                isDragOver ? 'drop-zone-active' : 'border-white/10 hover:border-white/20'
              }`}
              style={{ background: 'var(--bg-card)' }}
            >
              <input
                type="file"
                accept=".csv"
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <Upload size={20} className="mx-auto mb-2 text-slate-500" />
              <p className="text-xs font-semibold text-slate-400">
                {uploadName || 'Drop CSV or click to upload'}
              </p>
              {uploadName && (
                <p className="text-[10px] text-cyan-400 mt-1 font-mono truncate">{uploadName}</p>
              )}
            </div>

            {/* Run Inference */}
            <button
              onClick={runInference}
              disabled={!isLoaded || isRunning}
              className={`w-full py-2.5 rounded-xl text-xs font-bold tracking-wide flex items-center justify-center gap-2 transition-all ${
                !isLoaded || isRunning
                  ? 'bg-white/5 text-slate-600 cursor-not-allowed border border-white/5'
                  : 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30 active:scale-[0.98]'
              }`}
            >
              {isRunning ? <Activity size={14} className="animate-spin" /> : <Play size={14} />}
              {isRunning ? 'ANALYZING...' : 'RUN INFERENCE'}
            </button>

            {/* Progress */}
            {isRunning && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>{progress.processed.toLocaleString()} / {progress.total.toLocaleString()}</span>
                  <span>{pctDone}%</span>
                </div>
                <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="progress-bar h-full rounded-full transition-all duration-300"
                    style={{ width: `${pctDone}%` }} />
                </div>
              </div>
            )}
          </div>

          {/* Search */}
          <div className="p-4 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-white/5 focus-within:border-cyan-500/30 transition-colors">
              <Search size={13} className="text-slate-500" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search node..."
                className="bg-transparent text-xs text-slate-200 placeholder:text-slate-600 outline-none w-full font-medium"
              />
            </div>
          </div>

          {/* Legend */}
          <div className="p-4 space-y-3">
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.15em]">Node Legend</p>
            {[
              { color: 'bg-cyan-500', label: 'Stable Account', shadow: 'shadow-cyan-500/20' },
              { color: 'bg-amber-500', label: 'Suspicious', shadow: 'shadow-amber-500/20' },
              { color: 'bg-red-500', label: 'Critical Risk', shadow: 'shadow-red-500/20' },
              { color: 'bg-purple-500', label: 'In Circular Loop', shadow: 'shadow-purple-500/20' },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${item.color} shadow-sm ${item.shadow}`} />
                <span className="text-xs text-slate-400 font-medium">{item.label}</span>
              </div>
            ))}
            <div className="flex items-center gap-3 mt-2 pt-2 border-t border-white/5">
              <div className="w-6 h-0.5 bg-slate-500 rounded-full" />
              <span className="text-xs text-slate-500 font-medium">Clean Transfer</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-0.5 bg-red-500 rounded-full" />
              <span className="text-xs text-slate-500 font-medium">Flagged Transfer</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-0.5 bg-purple-500 rounded-full" />
              <span className="text-xs text-slate-500 font-medium">Loop Edge</span>
            </div>
          </div>

          {/* Hover tooltip in sidebar */}
          {hoveredNode && (
            <div className="mt-auto p-4 border-t animate-fade-in" style={{ borderColor: 'var(--border-subtle)' }}>
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.15em] mb-2">
                <Eye size={10} className="inline mr-1" />Node Inspector
              </p>
              <div className="glass-card rounded-lg p-3 space-y-1.5">
                <p className="text-sm font-bold text-white truncate">{hoveredNode.id}</p>
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    hoveredNode.status === 'CRITICAL' ? 'bg-red-500/15 text-red-400' :
                    hoveredNode.status === 'SUSPICIOUS' ? 'bg-amber-500/15 text-amber-400' :
                    'bg-cyan-500/15 text-cyan-400'
                  }`}>{hoveredNode.status}</span>
                  {hoveredNode.in_loop && (
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-purple-500/15 text-purple-400">LOOP</span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 font-mono">Risk: {hoveredNode.risk_score}%</p>
                <p className="text-[11px] text-slate-400 font-mono">Connections: {hoveredNode.degree}</p>
                {hoveredNode.reasons && hoveredNode.reasons.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-white/5">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Triggers</p>
                    <div className="flex flex-wrap gap-1">
                      {hoveredNode.reasons.map((r: string) => (
                        <span key={r} className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${
                          r === 'Circular Loop' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                          'bg-amber-500/10 text-amber-400 border-amber-500/20'
                        }`}>
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </aside>

        {/* ─── Graph Canvas ─── */}
        <main ref={containerRef} className="flex-1 relative" style={{ background: 'var(--bg-void)' }}>
          {isLoaded ? (
            <ForceGraph2D
              ref={graphRef as any}
              graphData={graphData}
              width={dimensions.width}
              height={dimensions.height}
              backgroundColor="#050a18"
              nodeId="id"
              nodeCanvasObject={nodeCanvasObject}
              nodePointerAreaPaint={(node: any, color, ctx) => {
                if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
                const s = Math.max(3, Math.min(8, Math.sqrt(node.degree || 1) * 1.5 + 2));
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x, node.y, s + 2, 0, 2 * Math.PI);
                ctx.fill();
              }}
              linkColor={linkColor}
              linkWidth={linkWidth}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              linkDirectionalParticles={linkParticles}
              linkDirectionalParticleWidth={1.5}
              linkDirectionalParticleSpeed={0.004}
              linkDirectionalParticleColor={(link: any) =>
                link.in_loop ? '#8b5cf6' : link.is_alert ? '#ef4444' : '#06b6d4'
              }
              onNodeHover={(node: any) => setHoveredNode(node || null)}
              onNodeClick={(node: any) => {
                if (node) {
                  graphRef.current?.centerAt(node.x, node.y, 400);
                  graphRef.current?.zoom(6, 400);
                }
              }}
              cooldownTicks={200}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.25}
              enableNodeDrag={true}
              enableZoomInteraction={true}
              enablePanInteraction={true}
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="w-20 h-20 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-5">
                <Crosshair size={32} className="text-slate-600" />
              </div>
              <p className="text-sm font-semibold text-slate-500">No Graph Data</p>
              <p className="text-xs text-slate-600 mt-1.5 max-w-[240px] text-center leading-relaxed">
                Upload a transaction CSV to load the network graph and begin investigation.
              </p>
            </div>
          )}

          {/* Graph overlay controls */}
          {isLoaded && (
            <div className="absolute top-4 left-4 flex items-center gap-2 z-10">
              <button
                onClick={() => graphRef.current?.zoomToFit(400, 60)}
                className="glass-card px-3 py-1.5 rounded-lg text-[11px] font-semibold text-slate-300 hover:text-white transition-colors"
              >
                Fit View
              </button>
              {filterPattern && (
                <button
                  onClick={() => setFilterPattern(null)}
                  className="glass-card px-3 py-1.5 rounded-lg text-[11px] font-semibold text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
                >
                  <X size={10} /> Clear Filter
                </button>
              )}
            </div>
          )}
        </main>

        {/* ─── Right Panel ─── */}
        <aside className="w-[300px] flex flex-col border-l overflow-hidden"
          style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-primary)' }}>

          {/* Fraud Patterns Summary */}
          <div className="p-4 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <AlertTriangle size={13} className="text-red-400" />
                <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.15em]">
                  Detected Patterns
                </h2>
              </div>
              {totalAlerts > 0 && (
                <span className="text-xs font-bold text-red-400 font-mono">{totalAlerts}</span>
              )}
            </div>

            {totalAlerts > 0 ? (
              <div className="space-y-2">
                {Object.entries(patterns).map(([type, count]) => (
                  <PatternCard
                    key={type}
                    type={type}
                    count={count}
                    total={totalAlerts}
                    active={filterPattern === type}
                    onClick={() => setFilterPattern(filterPattern === type ? null : type)}
                  />
                ))}
                {loopNodes > 0 && (
                  <div className="glass-card glow-border-purple rounded-xl p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RefreshCw size={13} className="text-purple-400" />
                        <span className="text-xs font-semibold text-slate-300">Circular Loops</span>
                      </div>
                      <span className="text-lg font-bold text-purple-400 font-mono">{loopNodes}</span>
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">Nodes in detected round-trip cycles</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-6">
                <Shield size={24} className="mx-auto text-slate-700 mb-2" />
                <p className="text-xs text-slate-500 font-medium">
                  {isLoaded ? 'Run inference to detect patterns' : 'Upload data to begin'}
                </p>
              </div>
            )}
          </div>

          {/* Alert Feed */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="px-4 py-3 flex items-center justify-between border-b"
              style={{ borderColor: 'var(--border-subtle)' }}>
              <div className="flex items-center gap-2">
                <Activity size={13} className={isRunning ? 'text-emerald-400 animate-pulse' : 'text-slate-500'} />
                <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.15em]">
                  Live Alert Feed
                </h2>
              </div>
              <span className="text-[10px] font-mono text-slate-500">{alerts.length} alerts</span>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-0">
              {alerts.length > 0 ? (
                alerts.map((a, i) => <AlertItem key={a.tx_id + i} alert={a} index={i} />)
              ) : (
                <div className="flex flex-col items-center justify-center h-full opacity-60">
                  <Clock size={20} className="text-slate-600 mb-2" />
                  <p className="text-xs text-slate-500 text-center font-medium">
                    Alerts will appear here<br />during inference
                  </p>
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* ─── Bottom Status Bar ─── */}
      <footer className="h-7 flex items-center justify-between px-5 text-[10px] font-mono border-t"
        style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-primary)' }}>
        <span className="text-slate-600">
          IFFT v1.0 — Temporal Graph Neural Network Inference Engine
        </span>
        <div className="flex items-center gap-5 text-slate-500">
          {isRunning && (
            <span className="text-cyan-400">
              Processing: {progress.processed}/{progress.total} ({pctDone}%)
            </span>
          )}
          <span>Neo4j: {NEO4J_URI}</span>
        </div>
      </footer>
    </div>
  );
}
