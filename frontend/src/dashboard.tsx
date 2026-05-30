import { useState, useEffect, useRef, useCallback, FC } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Play, Shield, Activity, RefreshCw, Layers, BarChart3, TrendingUp, CheckCircle, XCircle, AlertCircle, ChevronDown, ChevronUp, Search, X } from 'lucide-react';
import './index.css';

interface GraphNode { id: string; status: string; risk_score: number; is_fraud: boolean; in_loop: boolean; degree: number; reasons?: string[]; x?: number; y?: number; }
interface GraphLink { source: string | GraphNode; target: string | GraphNode; tx_id: string; amount: number; type: string; is_alert: boolean; in_loop: boolean; risk_score: number; review_status?: string; }
interface GraphData { nodes: GraphNode[]; links: GraphLink[]; }
interface TxRow {
  tx_id: string; sender: string; receiver: string; amount: number;
  tx_type: string; risk_score: number; currency: string; payment_format: string;
  timestamp: number; status: string; case_id?: string; isNew?: boolean;
}

const API = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/inference';
const NODE_COLORS: Record<string, string> = { STABLE: '#4a4a5a', SUSPICIOUS: '#9a7030', CRITICAL: '#8a3030', CONFIRMED_FRAUD: '#c44040' };

function riskClass(r: number) { if (r >= 70) return 'critical'; if (r >= 50) return 'high'; if (r >= 30) return 'medium'; return 'low'; }
function fmtAmount(n: number) { return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }

const TypeBadge: FC<{ type: string }> = ({ type }) => {
  const cls = type === 'Circular' ? 'circular' : type === 'Layering' ? 'layering' : type === 'Structuring' ? 'structuring' : 'normal';
  const icon = type === 'Circular' ? <RefreshCw size={9} /> : type === 'Layering' ? <Layers size={9} /> : type === 'Structuring' ? <BarChart3 size={9} /> : <TrendingUp size={9} />;
  return <span className={`badge-type ${cls}`}>{icon}{type}</span>;
};

const StatusBadge: FC<{ status: string }> = ({ status }) => {
  const map: Record<string, string> = { NORMAL: 'badge-normal', SUSPICIOUS: 'badge-flagged', PENDING_REVIEW: 'badge-pending', APPROVED: 'badge-approved', REJECTED: 'badge-rejected', ESCALATED: 'badge-escalated' };
  const label: Record<string, string> = { NORMAL: 'Normal', SUSPICIOUS: 'Flagged', PENDING_REVIEW: 'Pending', APPROVED: 'Cleared', REJECTED: 'Rejected', ESCALATED: 'Escalated' };
  return <span className={`badge ${map[status] || 'badge-normal'}`}>{label[status] || status}</span>;
};

const RiskCell: FC<{ score: number }> = ({ score }) => {
  const cls = riskClass(score);
  return (
    <div className="risk-cell">
      <div className="risk-bar-track"><div className={`risk-bar-fill ${cls}`} style={{ width: `${score}%` }} /></div>
      <span className={`risk-text ${cls}`}>{score > 0 ? `${score.toFixed(1)}%` : '—'}</span>
    </div>
  );
};

export default function App() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [transactions, setTransactions] = useState<TxRow[]>([]);
  const [cases, setCases] = useState<TxRow[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'cases'>('dashboard');
  const [tableFilter, setTableFilter] = useState<'all' | 'flagged'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTx, setSelectedTx] = useState<TxRow | null>(null);
  const [graphCollapsed, setGraphCollapsed] = useState(false);
  const [progress, setProgress] = useState({ processed: 0, total: 0 });
  const [alertCount, setAlertCount] = useState(0);
  const [patternCount, setPatternCount] = useState(0);

  const [sortCol, setSortCol] = useState<string>('');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const fullGraphRef = useRef<GraphData>({ nodes: [], links: [] });
  const graphRef = useRef<any>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });
  const tableBodyRef = useRef<HTMLTableSectionElement>(null);
  const seenPatterns = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(e => setDimensions({ width: e[0].contentRect.width, height: e[0].contentRect.height }));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const fetchGraph = () => fetch(`${API}/api/graph`).then(r => r.json()).then(data => { fullGraphRef.current = data; setGraphData({ nodes: data.nodes, links: [] }); setIsLoaded(true); });
  const fetchCases = () => fetch(`${API}/api/cases`).then(r => r.json()).then(data => setCases(data));

  useEffect(() => { fetchGraph(); setTimeout(() => graphRef.current?.zoomToFit(400, 60), 500); }, []);
  useEffect(() => { if (graphRef.current && isLoaded) { graphRef.current.d3Force('charge').strength(-400); graphRef.current.d3Force('link').distance(100); graphRef.current.d3Force('center').strength(0.05); } }, [isLoaded, graphData.nodes.length]);
  useEffect(() => { if (isRunning) { const iv = setInterval(fetchCases, 2000); return () => clearInterval(iv); } }, [isRunning]);

  const runDemo = () => {
    if (isRunning) return;
    setIsRunning(true); setTransactions([]); setCases([]); setSelectedTx(null);
    setAlertCount(0); setPatternCount(0); seenPatterns.current.clear();
    setProgress({ processed: 0, total: 0 });
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'inference_start') { setProgress({ processed: 0, total: msg.data.total }); }
      else if (msg.type === 'alert' || msg.type === 'transaction') {
        const d = msg.data;
        const isAlert = msg.type === 'alert';
        const row: TxRow = { tx_id: d.tx_id, sender: d.sender, receiver: d.receiver, amount: d.amount || 0, tx_type: d.tx_type || 'Normal', risk_score: d.risk_score || 0, currency: d.currency || 'USD', payment_format: d.payment_format || 'Wire', timestamp: d.timestamp || 0, status: isAlert ? 'SUSPICIOUS' : 'NORMAL', case_id: d.case_id, isNew: true };
        if (isAlert) {
          setAlertCount(p => p + 1);
          if (!seenPatterns.current.has(d.tx_type) && d.tx_type !== 'Normal') { seenPatterns.current.add(d.tx_type); setPatternCount(p => p + 1); }
        }
        setTransactions(prev => { const next = [...prev, row]; setTimeout(() => setTransactions(tx => tx.map(t => t.tx_id === row.tx_id ? { ...t, isNew: false } : t)), 500); return next; });
        setGraphData(prev => {
          const nodes = [...prev.nodes]; const links = [...prev.links];
          if (isAlert) { const s = nodes.find(n => n.id === d.sender); if (s) { s.status = 'SUSPICIOUS'; s.is_fraud = true; } }
          const fullLink = fullGraphRef.current.links.find(l => l.tx_id === d.tx_id);
          if (fullLink && !links.find(l => (l as any).tx_id === d.tx_id)) { links.push({ ...fullLink, is_alert: isAlert, risk_score: d.risk_score || 0 }); }
          return { nodes, links };
        });
        if (isAlert && graphRef.current) {
          setTimeout(() => { const sn = graphData.nodes.find(n => n.id === d.sender); if (sn?.x && sn?.y) { graphRef.current.centerAt(sn.x, sn.y, 800); graphRef.current.zoom(3, 800); } }, 100);
        }
      } else if (msg.type === 'progress') {
        setProgress(msg.data);
      } else if (msg.type === 'inference_complete') {
        setIsRunning(false); fetchCases(); setTimeout(() => graphRef.current?.zoomToFit(1000, 60), 1000);
      }
    };
    ws.onclose = () => setIsRunning(false);
  };

  const handleReview = async (caseId: string, decision: string) => {
    await fetch(`${API}/api/cases/${caseId}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }) });
    setSelectedTx(null); fetchCases(); fetchGraph();
    setTransactions(prev => prev.map(t => t.case_id === caseId ? { ...t, status: decision === 'APPROVED' ? 'APPROVED' : decision === 'REJECTED' ? 'REJECTED' : 'ESCALATED' } : t));
  };

  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') return;
    const size = Math.max(3, Math.min(8, Math.sqrt(node.degree || 1) * 1.5 + 2));
    const color = NODE_COLORS[node.status] || NODE_COLORS.STABLE;
    if (node.is_fraud) { const g = ctx.createRadialGradient(node.x, node.y, size, node.x, node.y, size * 3); g.addColorStop(0, color + '50'); g.addColorStop(1, color + '00'); ctx.beginPath(); ctx.arc(node.x, node.y, size * 3, 0, 2 * Math.PI); ctx.fillStyle = g; ctx.fill(); }
    ctx.beginPath(); ctx.arc(node.x, node.y, size, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 0.5; ctx.stroke();
    if (globalScale > 2 || (node.is_fraud && globalScale > 1.2)) { ctx.font = `${node.is_fraud ? '600' : '400'} ${Math.max(10 / globalScale, 3)}px Inter`; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = node.is_fraud ? '#ecedee' : '#565868'; ctx.fillText(node.id, node.x, node.y + size + 2); }
  }, []);

  const handleSort = (col: string) => { if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortCol(col); setSortDir('desc'); } };

  const filteredTx = transactions.filter(t => {
    if (tableFilter === 'flagged' && t.status === 'NORMAL') return false;
    if (searchQuery) { const q = searchQuery.toLowerCase(); return t.tx_id.toLowerCase().includes(q) || t.sender.toLowerCase().includes(q) || t.receiver.toLowerCase().includes(q) || t.tx_type.toLowerCase().includes(q); }
    return true;
  }).sort((a, b) => {
    if (!sortCol) return 0;
    const v = (x: TxRow) => sortCol === 'amount' ? x.amount : sortCol === 'risk' ? x.risk_score : 0;
    return sortDir === 'asc' ? v(a) - v(b) : v(b) - v(a);
  });

  const progPct = progress.total > 0 ? (progress.processed / progress.total) * 100 : 0;
  const isDone = !isRunning && progress.processed > 0 && progress.processed === progress.total;

  const SortIcon = ({ col }: { col: string }) => (
    <span className={`sort-icon ${sortCol === col ? 'sorted' : ''}`}>
      {sortCol === col ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
    </span>
  );

  return (
    <div className="app-root">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="header-logo"><Shield size={16} color="#fff" /></div>
          <div>
            <div className="header-title">FCCI <span>TGNN</span></div>
            <div className="header-subtitle">Real-Time AML Detection · Neo4j Aura</div>
          </div>
        </div>
        <nav className="header-nav">
          <button className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>Dashboard</button>
          <button className={`nav-tab ${activeTab === 'cases' ? 'active' : ''}`} onClick={() => setActiveTab('cases')}>
            Case Queue {cases.length > 0 && <span className="badge-count">{cases.length}</span>}
          </button>
        </nav>
        <div className="header-right">
          <div className={`status-chip ${isRunning ? 'running' : ''}`}>
            {isRunning ? <><span className="status-dot-live" />LIVE</> : isDone ? '✓ Complete' : '● Idle'}
          </div>
          <button className={`btn-start ${!isLoaded || isRunning ? '' : 'ready'}`} onClick={runDemo} disabled={!isLoaded || isRunning}>
            {isRunning ? <><Activity size={13} />Running…</> : <><Play size={13} fill="currentColor" />Start Demo</>}
          </button>
        </div>
      </header>

      {/* Progress */}
      <div className="progress-strip">
        <div className={`progress-fill ${isDone ? 'complete' : ''}`} style={{ width: `${progPct}%` }} />
      </div>

      {/* Stats */}
      <div className="stats-row">
        <div className="stat-card">
          <Activity size={14} className="stat-icon" />
          <div className="stat-body">
            <div className="stat-label">Processed</div>
            <div className="stat-value">{progress.processed || 0}<span className="frac"> / {progress.total || '—'}</span></div>
          </div>
        </div>
        <div className="stat-card">
          <AlertCircle size={14} className="stat-icon" style={{ color: alertCount > 0 ? 'var(--red)' : undefined }} />
          <div className="stat-body">
            <div className="stat-label">Alerts</div>
            <div className={`stat-value ${alertCount > 0 ? 'v-red' : ''}`}>{alertCount}</div>
          </div>
        </div>
        <div className="stat-card">
          <Layers size={14} className="stat-icon" style={{ color: patternCount > 0 ? 'var(--purple)' : undefined }} />
          <div className="stat-body">
            <div className="stat-label">Patterns</div>
            <div className={`stat-value ${patternCount > 0 ? 'v-purple' : ''}`}>{patternCount}</div>
          </div>
        </div>
        <div className="stat-card">
          <Shield size={14} className="stat-icon" style={{ color: cases.length > 0 ? 'var(--amber)' : undefined }} />
          <div className="stat-body">
            <div className="stat-label">Pending Review</div>
            <div className={`stat-value ${cases.length > 0 ? 'v-amber' : ''}`}>{cases.length}</div>
          </div>
        </div>
      </div>

      {/* Body */}
      {activeTab === 'dashboard' ? (
        <div className="main-content">
          {/* Left: Table */}
          <div className="table-panel">
            <div className="table-toolbar">
              <div className="filter-tabs">
                <button className={`filter-tab ${tableFilter === 'all' ? 'active' : ''}`} onClick={() => setTableFilter('all')}>All <span className="count">{transactions.length}</span></button>
                <button className={`filter-tab ${tableFilter === 'flagged' ? 'active' : ''}`} onClick={() => setTableFilter('flagged')}>Flagged <span className="count">{alertCount}</span></button>
              </div>
              <div className="search-box">
                <Search size={12} />
                <input placeholder="Search ID, entity, type…" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
                {searchQuery && <button style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex' }} onClick={() => setSearchQuery('')}><X size={12} /></button>}
              </div>
            </div>
            <div className="table-container">
              {filteredTx.length === 0 ? (
                <div className="empty-state">
                  <Shield size={32} color="var(--text-muted)" />
                  <p>{isLoaded ? 'Click "Start Demo" to begin inference' : 'Loading scenario…'}</p>
                </div>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: 36 }} className="align-center">#</th>
                      <th style={{ width: 80 }}>TX ID</th>
                      <th>Sender</th>
                      <th>Receiver</th>
                      <th className={`align-right ${sortCol === 'amount' ? 'sorted' : ''}`} onClick={() => handleSort('amount')}>Amount <SortIcon col="amount" /></th>
                      <th style={{ width: 70 }}>Currency</th>
                      <th style={{ width: 80 }}>Format</th>
                      <th style={{ width: 100 }}>Type</th>
                      <th className={`align-right ${sortCol === 'risk' ? 'sorted' : ''}`} onClick={() => handleSort('risk')} style={{ width: 110 }}>Risk Score <SortIcon col="risk" /></th>
                      <th className="align-center" style={{ width: 90 }}>Status</th>
                    </tr>
                  </thead>
                  <tbody ref={tableBodyRef}>
                    {filteredTx.map((t, i) => (
                      <tr key={t.tx_id} className={`${t.status !== 'NORMAL' ? 'row-alert' : ''} ${t.isNew ? 'row-new' : ''} ${selectedTx?.tx_id === t.tx_id ? 'selected' : ''}`} onClick={() => setSelectedTx(t.status !== 'NORMAL' ? t : null)}>
                        <td className="align-center text-muted" style={{ fontSize: 11 }}>{i + 1}</td>
                        <td className="mono" style={{ color: 'var(--accent)' }}>{t.tx_id}</td>
                        <td style={{ maxWidth: 130 }}>{t.sender}</td>
                        <td style={{ maxWidth: 130 }}>{t.receiver}</td>
                        <td className="align-right mono">{fmtAmount(t.amount)}</td>
                        <td className="mono text-muted" style={{ fontSize: 11 }}>{t.currency}</td>
                        <td className="text-muted" style={{ fontSize: 11 }}>{t.payment_format}</td>
                        <td><TypeBadge type={t.tx_type} /></td>
                        <td className="align-right"><RiskCell score={t.risk_score} /></td>
                        <td className="align-center"><StatusBadge status={t.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>

          {/* Right: Graph + Case Review */}
          <div className="graph-panel">
            <div className="panel-header">
              <span className="panel-title">Transaction Graph</span>
              <button className="panel-toggle" onClick={() => setGraphCollapsed(c => !c)}>
                {graphCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
              </button>
            </div>
            <div ref={containerRef} className={`graph-container ${graphCollapsed ? 'collapsed' : ''}`}>
              {isLoaded && !graphCollapsed && (
                <ForceGraph2D ref={graphRef as any} graphData={graphData} width={dimensions.width} height={dimensions.height} backgroundColor="transparent" nodeId="id" nodeCanvasObject={nodeCanvasObject} linkColor={(l: any) => l.is_alert ? 'rgba(196,64,64,0.65)' : 'rgba(255,255,255,0.1)'} linkWidth={(l: any) => l.is_alert ? 1.5 : 0.8} linkDirectionalParticles={(l: any) => l.is_alert ? 2 : 0} linkDirectionalParticleWidth={2} linkDirectionalParticleSpeed={0.004} linkDirectionalParticleColor={() => '#c44040'} d3VelocityDecay={0.8} d3AlphaDecay={0.05} />
              )}
            </div>

            {/* Case Review */}
            {selectedTx && (
              <div className="case-review">
                <div className="case-review-header">
                  <div>
                    <div className="case-review-title">{selectedTx.tx_id}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="case-review-badge"><TypeBadge type={selectedTx.tx_type} /></span>
                    <button className="close-btn" onClick={() => setSelectedTx(null)}><X size={14} /></button>
                  </div>
                </div>
                <div className="case-details">
                  <div className="detail-item"><div className="detail-label">Sender</div><div className="detail-value">{selectedTx.sender}</div></div>
                  <div className="detail-item"><div className="detail-label">Receiver</div><div className="detail-value">{selectedTx.receiver}</div></div>
                  <div className="detail-item"><div className="detail-label">Amount</div><div className="detail-value mono">{fmtAmount(selectedTx.amount)}</div></div>
                  <div className="detail-item"><div className="detail-label">Currency</div><div className="detail-value mono">{selectedTx.currency}</div></div>
                  <div className="detail-item"><div className="detail-label">Format</div><div className="detail-value">{selectedTx.payment_format}</div></div>
                  <div className="detail-item"><div className="detail-label">TGNN Risk</div><div className="detail-value risk-highlight">{selectedTx.risk_score.toFixed(1)}%</div></div>
                </div>
                {selectedTx.case_id && (
                  <div className="case-actions">
                    <button className="action-btn action-reject" onClick={() => handleReview(selectedTx.case_id!, 'REJECTED')}><XCircle size={12} />Confirm Fraud</button>
                    <button className="action-btn action-approve" onClick={() => handleReview(selectedTx.case_id!, 'APPROVED')}><CheckCircle size={12} />False Positive</button>
                    <button className="action-btn action-escalate" onClick={() => handleReview(selectedTx.case_id!, 'ESCALATED')}><AlertCircle size={12} />Escalate</button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Cases Tab — proper table */
        <div className="cases-view">
          <div className="cases-list-panel">
            {cases.length === 0 ? (
              <div className="empty-cases"><Shield size={28} color="var(--text-muted)" /><p>No pending cases. Run the demo to generate alerts.</p></div>
            ) : (
              <table className="cases-table">
                <thead>
                  <tr>
                    <th style={{ width: 90 }}>TX ID</th>
                    <th>Sender</th>
                    <th>Receiver</th>
                    <th className="align-right">Amount</th>
                    <th style={{ width: 100 }}>Type</th>
                    <th className="align-right" style={{ width: 100 }}>Risk</th>
                    <th style={{ width: 120 }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((c: any) => (
                    <>
                      <tr key={c.case_id} className={selectedTx?.case_id === c.case_id ? 'selected' : ''} onClick={() => setSelectedTx(selectedTx?.case_id === c.case_id ? null : { ...c, status: 'SUSPICIOUS' })}>
                        <td className="mono" style={{ color: 'var(--accent)' }}>{c.tx_id}</td>
                        <td>{c.sender}</td>
                        <td className="dim">{c.receiver}</td>
                        <td className="align-right mono">{fmtAmount(c.amount)}</td>
                        <td><TypeBadge type={c.tx_type} /></td>
                        <td className="align-right"><RiskCell score={c.risk_score} /></td>
                        <td>
                          <div style={{ display: 'flex', gap: 4 }}>
                            <button className="action-btn action-reject" style={{ flex: 'none', padding: '3px 7px', fontSize: 10 }} onClick={e => { e.stopPropagation(); handleReview(c.case_id, 'REJECTED'); }}><XCircle size={10} />Fraud</button>
                            <button className="action-btn action-approve" style={{ flex: 'none', padding: '3px 7px', fontSize: 10 }} onClick={e => { e.stopPropagation(); handleReview(c.case_id, 'APPROVED'); }}><CheckCircle size={10} />Clear</button>
                            <button className="action-btn action-escalate" style={{ flex: 'none', padding: '3px 7px', fontSize: 10 }} onClick={e => { e.stopPropagation(); handleReview(c.case_id, 'ESCALATED'); }}><AlertCircle size={10} />Esc.</button>
                          </div>
                        </td>
                      </tr>
                    </>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
