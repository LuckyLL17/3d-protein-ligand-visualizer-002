import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { ContactShadows, Line, OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

type DisplayMode = 'cartoon' | 'surface' | 'atoms';
type Residue = { id: number; code: string; name: string; x: number; y: number; z: number; color: string; role: 'active' | 'support' | 'loop' };
type Element = 'C' | 'N' | 'O' | 'S';
type Atom = { id: string; element: Element; x: number; y: number; z: number; color: string; radius: number };

const residues: Residue[] = [
  { id: 41, code: 'GLU41', name: '谷氨酸', x: -1.6, y: .4, z: .5, color: '#ef8e77', role: 'active' },
  { id: 78, code: 'TYR78', name: '酪氨酸', x: -.7, y: 1.15, z: -.25, color: '#f0c76e', role: 'support' },
  { id: 102, code: 'HIS102', name: '组氨酸', x: .4, y: .75, z: .2, color: '#71c9e6', role: 'active' },
  { id: 146, code: 'ASP146', name: '天冬氨酸', x: 1.2, y: .15, z: -.45, color: '#e88389', role: 'active' },
  { id: 183, code: 'PHE183', name: '苯丙氨酸', x: 1.8, y: .7, z: .35, color: '#c28ce8', role: 'support' },
  { id: 219, code: 'SER219', name: '丝氨酸', x: .55, y: -.3, z: -.65, color: '#8fd6a2', role: 'loop' },
  { id: 244, code: 'GLY244', name: '甘氨酸', x: -1.05, y: -.55, z: -.3, color: '#9aabb4', role: 'loop' }
];
const ligandAtoms: Atom[] = [
  { id: 'C1', element: 'C', x: 0, y: 0, z: 0, color: '#d6e1e4', radius: .17 },
  { id: 'C2', element: 'C', x: .55, y: .26, z: .08, color: '#d6e1e4', radius: .17 },
  { id: 'N1', element: 'N', x: .98, y: -.18, z: .18, color: '#54b9e3', radius: .2 },
  { id: 'O1', element: 'O', x: -.48, y: .35, z: .05, color: '#ec7474', radius: .18 },
  { id: 'O2', element: 'O', x: -.15, y: -.5, z: -.1, color: '#ec7474', radius: .18 },
  { id: 'S1', element: 'S', x: .42, y: .05, z: -.62, color: '#e7c35f', radius: .24 }
];
const bonds: [string, string][] = [['C1','C2'],['C1','O1'],['C1','O2'],['C2','N1'],['C2','S1']];
const helixPoints = Array.from({ length: 46 }, (_, index) => { const t = index / 45; return new THREE.Vector3(-3.2 + t * 6.4, Math.sin(t * Math.PI * 4) * .72 + Math.cos(t * Math.PI * 2) * .18, Math.cos(t * Math.PI * 3) * .74); });
const sheetPoints = Array.from({ length: 22 }, (_, index) => new THREE.Vector3(-3.5 + index * .33, 1.4 + Math.sin(index * .55) * .16, -.9 + Math.sin(index * .55) * .25));

/* ---------- 相互作用分析 ---------- */
type Category = 'hbond' | 'polar' | 'hydrophobic' | 'vdw';
type Contact = {
  id: string;
  residueId: number;
  residueCode: string;
  residueName: string;
  atomId: string;
  element: Element;
  distance: number;
  category: Category;
};
type WorldPos = [number, number, number];
type AnalysisResult = {
  ok: true;
  contacts: Contact[];
  residueWorld: Record<number, WorldPos>;
  atomWorld: Record<string, WorldPos>;
} | { ok: false };

const CATEGORIES: { key: Category; label: string; short: string; color: string; hint: string }[] = [
  { key: 'hbond', label: '氢键', short: 'H', color: '#56b6e8', hint: 'N/O 供体-受体，≤ 3.5 Å' },
  { key: 'polar', label: '极性接触', short: 'P', color: '#7fd6a2', hint: '极性原子间，≤ 4.0 Å' },
  { key: 'hydrophobic', label: '疏水接触', short: 'HΦ', color: '#e3b341', hint: '碳骨架与疏水残基，≤ 4.5 Å' },
  { key: 'vdw', label: '范德华接触', short: 'vdW', color: '#b78cf0', hint: '范德华半径重叠，≤ 4.0 Å' }
];
const CATEGORY_META: Record<Category, { label: string; color: string }> = Object.fromEntries(CATEGORIES.map(c => [c.key, { label: c.label, color: c.color }])) as Record<Category, { label: string; color: string }>;

const POLAR_RESIDUES = new Set(['GLU41', 'TYR78', 'HIS102', 'ASP146', 'SER219']);
const HYDROPHOBIC_RESIDUES = new Set(['PHE183', 'GLY244']);
const VDW_RADII: Record<Element, number> = { C: 1.7, N: 1.55, O: 1.52, S: 1.80 };
const PROTEIN_VDW_RADIUS = 1.7; // 蛋白重原子平均范德华半径
const VDW_TOLERANCE = 0.5;
const COMPUTE_RADIUS = 6.0; // 预计算半径，阈值滑块只做过滤
const H_BOND_CUTOFF = 3.5;
const POLAR_CUTOFF = 4.0;
const HYDROPHOBIC_CUTOFF = 4.5;

function readWorldPositions(residueRefs: React.MutableRefObject<Record<number, THREE.Group | null>>, atomRefs: React.MutableRefObject<Record<string, THREE.Mesh | null>>) {
  const residueWorld: Record<number, WorldPos> = {};
  for (const r of residues) {
    const obj = residueRefs.current[r.id];
    if (!obj) return null;
    const v = new THREE.Vector3();
    obj.getWorldPosition(v);
    if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) return null;
    residueWorld[r.id] = [v.x, v.y, v.z];
  }
  const atomWorld: Record<string, WorldPos> = {};
  for (const a of ligandAtoms) {
    const obj = atomRefs.current[a.id];
    if (!obj) return null;
    const v = new THREE.Vector3();
    obj.getWorldPosition(v);
    if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) return null;
    atomWorld[a.id] = [v.x, v.y, v.z];
  }
  return { residueWorld, atomWorld };
}

function computeContacts(residueRefs: React.MutableRefObject<Record<number, THREE.Group | null>>, atomRefs: React.MutableRefObject<Record<string, THREE.Mesh | null>>): AnalysisResult {
  const coords = readWorldPositions(residueRefs, atomRefs);
  if (!coords) return { ok: false };
  const { residueWorld, atomWorld } = coords;
  const contacts: Contact[] = [];
  for (const r of residues) {
    const rp = new THREE.Vector3(...residueWorld[r.id]);
    for (const a of ligandAtoms) {
      const ap = new THREE.Vector3(...atomWorld[a.id]);
      const distance = rp.distanceTo(ap);
      if (distance > COMPUTE_RADIUS) continue;
      const polar = POLAR_RESIDUES.has(r.code);
      const hydrophobic = HYDROPHOBIC_RESIDUES.has(r.code);
      let category: Category | null = null;
      if ((a.element === 'N' || a.element === 'O') && polar && distance <= H_BOND_CUTOFF) category = 'hbond';
      else if ((a.element === 'N' || a.element === 'O' || a.element === 'S') && polar && distance <= POLAR_CUTOFF) category = 'polar';
      else if (a.element === 'C' && hydrophobic && distance <= HYDROPHOBIC_CUTOFF) category = 'hydrophobic';
      else if (distance <= VDW_RADII[a.element] + PROTEIN_VDW_RADIUS + VDW_TOLERANCE) category = 'vdw';
      if (category) {
        contacts.push({
          id: `${r.id}-${a.id}`,
          residueId: r.id,
          residueCode: r.code,
          residueName: r.name,
          atomId: a.id,
          element: a.element,
          distance,
          category
        });
      }
    }
  }
  contacts.sort((a, b) => a.distance - b.distance);
  return { ok: true, contacts, residueWorld, atomWorld };
}

/* ---------- 3D 组件 ---------- */
function Ligand({ rotation, selectedAtom, onSelect, atomRefs }: { rotation: number; selectedAtom: string; onSelect: (id: string) => void; atomRefs: React.MutableRefObject<Record<string, THREE.Mesh | null>> }) {
  const positions = useMemo(() => Object.fromEntries(ligandAtoms.map(atom => [atom.id, new THREE.Vector3(atom.x, atom.y, atom.z)])), []);
  return <group position={[.1, .45, .15]} rotation={[0, rotation, 0]}>
    {bonds.map(([from, to]) => <Line key={`${from}-${to}`} points={[positions[from], positions[to]]} color="#cbd9dc" lineWidth={3} />)}
    {ligandAtoms.map(atom => <mesh
      key={atom.id}
      ref={el => { atomRefs.current[atom.id] = el; }}
      position={[atom.x, atom.y, atom.z]}
      onClick={(event) => { event.stopPropagation(); onSelect(atom.id); }}>
      <sphereGeometry args={[atom.radius * (selectedAtom === atom.id ? 1.35 : 1), 16, 12]} />
      <meshStandardMaterial color={atom.color} emissive={selectedAtom === atom.id ? atom.color : '#000000'} emissiveIntensity={selectedAtom === atom.id ? .7 : 0} />
    </mesh>)}
  </group>;
}

type ContactLine = { key: string; from: WorldPos; to: WorldPos; color: string; active: boolean; dimmed: boolean };

function ProteinModel({ displayMode, selectedResidue, setSelectedResidue, ligandRotation, selectedAtom, setSelectedAtom, residueRefs, atomRefs, contactLines }: {
  displayMode: DisplayMode;
  selectedResidue: number;
  setSelectedResidue: (id: number) => void;
  ligandRotation: number;
  selectedAtom: string;
  setSelectedAtom: (id: string) => void;
  residueRefs: React.MutableRefObject<Record<number, THREE.Group | null>>;
  atomRefs: React.MutableRefObject<Record<string, THREE.Mesh | null>>;
  contactLines: ContactLine[];
}) {
  const showAtoms = displayMode === 'atoms';
  const surface = displayMode === 'surface';
  return <Canvas shadows camera={{ position: [8, 5.5, 9], fov: 40 }}>
    <color attach="background" args={['#09151d']} />
    <ambientLight intensity={1.2} />
    <directionalLight position={[5, 8, 5]} intensity={2.4} castShadow />
    <pointLight position={[-4, 3, 3]} color="#7fe0ff" intensity={18} distance={11} />
    <group rotation={[.08, -.25, 0]}>
      {!showAtoms && <><Line points={helixPoints} color="#6b9fe9" lineWidth={surface ? 12 : 5} /> <Line points={sheetPoints} color="#d58acb" lineWidth={surface ? 13 : 6} /></>}
      {showAtoms && residues.map(residue => <mesh key={residue.id} position={[residue.x, residue.y, residue.z]}><sphereGeometry args={[.22, 12, 10]} /><meshStandardMaterial color={residue.color} /></mesh>)}
      {residues.map(residue => <group
        key={`res-${residue.id}`}
        ref={el => { residueRefs.current[residue.id] = el; }}
        position={[residue.x, residue.y, residue.z]}>
        <mesh onClick={(event) => { event.stopPropagation(); setSelectedResidue(residue.id); }}>
          <sphereGeometry args={[selectedResidue === residue.id ? .28 : .18, 16, 12]} />
          <meshStandardMaterial color={residue.color} emissive={selectedResidue === residue.id ? residue.color : '#000'} emissiveIntensity={selectedResidue === residue.id ? .8 : 0} transparent opacity={surface ? .9 : 1} />
        </mesh>
        <Text position={[0, .43, 0]} fontSize={.14} color={selectedResidue === residue.id ? '#ffe9a2' : '#8baeb9'} anchorX="center">{residue.code}</Text>
      </group>)}
      <Ligand rotation={ligandRotation} selectedAtom={selectedAtom} onSelect={setSelectedAtom} atomRefs={atomRefs} />
    </group>
    {contactLines.map(line => <Line
      key={line.key}
      points={[line.from, line.to]}
      color={line.color}
      lineWidth={line.active ? 3.5 : 1.4}
      transparent
      opacity={line.dimmed ? .12 : line.active ? 1 : .85}
    />)}
    <ContactShadows position={[0, -1.5, 0]} opacity={.32} scale={10} blur={2.5} />
    <OrbitControls makeDefault minDistance={5} maxDistance={16} />
  </Canvas>;
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>; }

const EyeIcon = ({ off }: { off: boolean }) => off
  ? <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 3l18 18" /><path d="M10.6 10.7a2 2 0 002.8 2.6" /><path d="M9.9 4.2A9.1 9.1 0 0112 4c7 0 10 8 10 8a13 13 0 01-3.2 4M6.6 6.6A12.6 12.6 0 002 12s3 8 10 8a9.2 9.2 0 005.4-1.7" /></svg>
  : <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M2 12s3-8 10-8 10 8 10 8-3 8-10 8-10-8-10-8z" /><circle cx="12" cy="12" r="3" /></svg>;

export default function App() {
  const [mode, setMode] = useState<DisplayMode>('cartoon');
  const [selectedResidue, setSelectedResidue] = useState(102);
  const [selectedAtom, setSelectedAtom] = useState('N1');
  const [ligandRotation, setLigandRotation] = useState(0.2);
  const [threshold, setThreshold] = useState(5.0);
  const [hiddenCategories, setHiddenCategories] = useState<Set<Category>>(new Set());
  const [hoveredContact, setHoveredContact] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'computing' | 'done' | 'missing'>('computing');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [recomputeNonce, setRecomputeNonce] = useState(0);

  const residueRefs = useRef<Record<number, THREE.Group | null>>({});
  const atomRefs = useRef<Record<string, THREE.Mesh | null>>({});
  const retryCount = useRef(0);

  const residue = residues.find(item => item.id === selectedResidue) ?? residues[2];

  /* 基于当前残基/配体世界坐标计算接触；配体旋转后自动重算 */
  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;
    setAnalysisStatus('computing');
    const timer = window.setTimeout(() => {
      const result = computeContacts(residueRefs, atomRefs);
      if (cancelled) return;
      if (!result.ok) {
        setAnalysisStatus('missing');
        if (retryCount.current < 8) {
          retryCount.current += 1;
          retryTimer = window.setTimeout(() => { if (!cancelled) setRecomputeNonce(n => n + 1); }, 700);
        }
        return;
      }
      retryCount.current = 0;
      setAnalysis(result);
      setAnalysisStatus('done');
    }, 450);
    return () => { cancelled = true; window.clearTimeout(timer); window.clearTimeout(retryTimer); };
  }, [ligandRotation, recomputeNonce]);

  /* 阈值与类别筛选：3D 连线、残基列表标记、详情面板三处共用同一份结果 */
  const visibleContacts = useMemo(() => {
    if (!analysis || !analysis.ok) return [];
    return analysis.contacts.filter(c => c.distance <= threshold && !hiddenCategories.has(c.category));
  }, [analysis, threshold, hiddenCategories]);

  const visibleByResidue = useMemo(() => {
    const map = new Map<number, Set<Category>>();
    for (const c of visibleContacts) {
      if (!map.has(c.residueId)) map.set(c.residueId, new Set());
      map.get(c.residueId)!.add(c.category);
    }
    return map;
  }, [visibleContacts]);

  const visibleByAtom = useMemo(() => new Set(visibleContacts.map(c => c.atomId)), [visibleContacts]);

  const contactLines = useMemo<ContactLine[]>(() => {
    if (!analysis || !analysis.ok) return [];
    return visibleContacts.map(c => ({
      key: c.id,
      from: analysis.residueWorld[c.residueId],
      to: analysis.atomWorld[c.atomId],
      color: CATEGORY_META[c.category].color,
      active: hoveredContact === c.id,
      dimmed: hoveredContact !== null && hoveredContact !== c.id
    }));
  }, [analysis, visibleContacts, hoveredContact]);

  const contactsByCategory = useMemo(() => {
    const groups: Record<Category, Contact[]> = { hbond: [], polar: [], hydrophobic: [], vdw: [] };
    for (const c of visibleContacts) groups[c.category].push(c);
    return CATEGORIES.map(meta => ({ meta, items: groups[meta.key] })).filter(g => g.items.length > 0);
  }, [visibleContacts]);

  const residueContacts = useMemo(() => visibleContacts.filter(c => c.residueId === residue.id), [visibleContacts, residue.id]);

  const toggleCategory = (cat: Category) => {
    setHiddenCategories(prev => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat); else next.add(cat);
      return next;
    });
  };
  const allHidden = hiddenCategories.size === CATEGORIES.length;
  const toggleAll = () => setHiddenCategories(allHidden ? new Set() : new Set(CATEGORIES.map(c => c.key)));

  const locate = (c: Contact) => {
    setSelectedResidue(c.residueId);
    setSelectedAtom(c.atomId);
    window.setTimeout(() => {
      document.getElementById(`residue-btn-${c.residueId}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 60);
  };

  return <div className="app">
    <header className="topbar">
      <div className="brand">
        <div className="brand-icon">P</div>
        <div><span className="eyebrow">STRUCTURAL BIOLOGY / ANALYSIS WORKSPACE</span><h1>Protein Atlas <span>·</span> 配体结合可视化</h1></div>
      </div>
      <div className="header-meta">
        <span className="dataset">PDB-LITE / 1MOL</span>
        <button onClick={toggleAll}>{allHidden ? '显示全部相互作用' : '隐藏全部相互作用'}</button>
        <button onClick={() => setLigandRotation(value => value + .35)}>旋转配体 ↻</button>
      </div>
    </header>
    <main className="layout">
      <section className="model-card">
        <div className="model-head">
          <div><span className="eyebrow">MOLECULAR VIEWER</span><h2>蛋白质结合口袋 · Ligand LIG-17</h2></div>
          <div className="mode-switch">{(['cartoon','surface','atoms'] as DisplayMode[]).map(item => <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{item === 'cartoon' ? '卡通带状' : item === 'surface' ? '表面模型' : '原子视图'}</button>)}</div>
        </div>
        <div className="canvas-wrap">
          <ProteinModel
            displayMode={mode}
            selectedResidue={selectedResidue}
            setSelectedResidue={setSelectedResidue}
            ligandRotation={ligandRotation}
            selectedAtom={selectedAtom}
            setSelectedAtom={setSelectedAtom}
            residueRefs={residueRefs}
            atomRefs={atomRefs}
            contactLines={contactLines}
          />
          <div className="viewer-legend">
            <span><i className="dot protein" />蛋白骨架</span>
            <span><i className="dot ligand" />配体原子</span>
            <span><i className="dot interaction" />相互作用</span>
          </div>
          <div className="viewer-hint">左键旋转 · 滚轮缩放 · 点击残基或原子</div>
        </div>
      </section>
      <aside className="sidebar">
        <section className="panel intro-panel">
          <span className="eyebrow">ACTIVE SITE</span>
          <h3>结合口袋分析</h3>
          <p>当前模型显示 7 个关键残基和一个小分子配体。通过选择残基查看空间关系与作用类型。</p>
          <div className="metrics">
            <Metric label="口袋体积" value="486 Å³" note="预测区域" />
            <Metric label="结合残基" value="5" note="3 active / 2 support" />
            <Metric label="氢键" value={String(analysisStatus === 'done' ? (analysis?.ok ? analysis.contacts.filter(c => c.category === 'hbond').length : 0) : 0)} note="几何筛选" />
          </div>
        </section>

        {/* 相互作用分析 */}
        <section className="panel analysis-panel">
          <div className="panel-title">
            <div><span className="eyebrow">CONTACT ANALYSIS</span><h3>相互作用分析</h3></div>
            <span className="count">{analysisStatus === 'done' ? visibleContacts.length : '–'}</span>
          </div>

          <div className="threshold-row">
            <label htmlFor="threshold">距离阈值</label>
            <input id="threshold" type="range" min={3.0} max={6.0} step={.1} value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} />
            <span className="threshold-value">{threshold.toFixed(1)} Å</span>
          </div>

          <div className="cat-chips">
            {CATEGORIES.map(cat => {
              const off = hiddenCategories.has(cat.key);
              const count = analysisStatus === 'done' && analysis?.ok ? analysis.contacts.filter(c => c.category === cat.key && c.distance <= threshold).length : 0;
              return <button key={cat.key} className={off ? 'cat-chip off' : 'cat-chip'} onClick={() => toggleCategory(cat.key)} title={cat.hint}>
                <i className="cat-dot" style={{ background: off ? '#3a525b' : cat.color }} />
                <span>{cat.label}</span>
                <em className="cat-count">{count}</em>
                <span className="eye"><EyeIcon off={off} /></span>
              </button>;
            })}
          </div>

          {analysisStatus === 'computing' && <div className="analysis-status"><span className="spinner" />正在基于当前坐标计算近距离接触…</div>}
          {analysisStatus === 'missing' && <div className="analysis-status warn"><span>⚠</span><span>缺少坐标数据：3D 场景中的残基或配体原子坐标尚未就绪，无法计算接触。</span><button onClick={() => setRecomputeNonce(n => n + 1)}>重新计算</button></div>}
          {analysisStatus === 'done' && visibleContacts.length === 0 && <div className="analysis-status">{allHidden ? '所有相互作用类别均已隐藏。' : '当前距离阈值与筛选条件下没有接触。'}<small>尝试调大距离阈值或显示更多类别</small></div>}

          {analysisStatus === 'done' && visibleContacts.length > 0 && <div className="contact-groups">
            {contactsByCategory.map(group => <div key={group.meta.key} className="contact-group">
              <div className="group-head"><i className="cat-dot" style={{ background: group.meta.color }} />{group.meta.label}<em className="cat-count">{group.items.length} 条</em></div>
              {group.items.map(c => {
                const active = selectedResidue === c.residueId && selectedAtom === c.atomId;
                return <button
                  key={c.id}
                  className={active ? 'contact-row active' : 'contact-row'}
                  onClick={() => locate(c)}
                  onMouseEnter={() => setHoveredContact(c.id)}
                  onMouseLeave={() => setHoveredContact(null)}>
                  <i className="cat-dot" style={{ background: group.meta.color }} />
                  <span className="res">{c.residueCode}<small>{c.residueName}</small></span>
                  <span className="atom"><i className="atom-element-badge" style={{ background: ligandAtoms.find(a => a.id === c.atomId)?.color }}>{c.element}</i>{c.atomId}</span>
                  <span className="dist">{c.distance.toFixed(2)} Å</span>
                  <em className="cat-tag" style={{ color: group.meta.color, borderColor: group.meta.color }}>{group.meta.short}</em>
                </button>;
              })}
            </div>)}
          </div>}
        </section>

        <section className="panel">
          <div className="panel-title">
            <div><span className="eyebrow">RESIDUES</span><h3>关键残基</h3></div>
            <span className="count">{residues.length}</span>
          </div>
          <div className="residue-list">
            {residues.map(item => {
              const cats = visibleByResidue.get(item.id);
              return <button key={item.id} id={`residue-btn-${item.id}`} className={selectedResidue === item.id ? 'residue selected' : 'residue'} onClick={() => setSelectedResidue(item.id)}>
                <span className="residue-color" style={{ background: item.color }} />
                <span className="residue-main">
                  <b>{item.code}</b><small>{item.name}</small>
                  <span className="residue-cats">
                    {CATEGORIES.map(cat => <i key={cat.key} className={cats?.has(cat.key) ? 'on' : ''} style={cats?.has(cat.key) ? { background: cat.color } : undefined} title={cat.label} />)}
                  </span>
                </span>
                <em>{item.role === 'active' ? 'ACTIVE' : item.role === 'support' ? 'SUPPORT' : 'LOOP'}</em>
              </button>;
            })}
          </div>
        </section>

        <section className="panel detail-panel">
          <span className="eyebrow">SELECTED RESIDUE</span>
          <div className="detail-title"><h3>{residue.code}</h3><span>{residue.name}</span></div>
          <div className="detail-grid">
            <div><span>空间坐标</span><b>{residue.x.toFixed(2)} / {residue.y.toFixed(2)} / {residue.z.toFixed(2)} Å</b></div>
            <div><span>残基角色</span><b>{residue.role === 'active' ? '结合活性位点' : residue.role === 'support' ? '口袋支撑' : '柔性环区'}</b></div>
            <div><span>配体距离</span><b>{(1.9 + residue.id % 6 * .22).toFixed(2)} Å</b></div>
            <div><span>相互作用</span><b>{residue.role === 'active' ? '氢键 / 极性' : '范德华力'}</b></div>
          </div>
          <div className="detail-contacts">
            <span>相互作用连线（{residueContacts.length}）</span>
            {analysisStatus !== 'done' && <em className="detail-empty">{analysisStatus === 'computing' ? '正在计算…' : '坐标缺失，暂不可用'}</em>}
            {analysisStatus === 'done' && residueContacts.length === 0 && <em className="detail-empty">当前阈值下该残基无可见接触</em>}
            {analysisStatus === 'done' && residueContacts.map(c => <button key={c.id} className="detail-contact" onClick={() => setSelectedAtom(c.atomId)} onMouseEnter={() => setHoveredContact(c.id)} onMouseLeave={() => setHoveredContact(null)}>
              <i className="cat-dot" style={{ background: CATEGORY_META[c.category].color }} />
              <i className="atom-element-badge" style={{ background: ligandAtoms.find(a => a.id === c.atomId)?.color }}>{c.element}</i>
              <span className="atom">{c.atomId}</span>
              <span className="dist">{c.distance.toFixed(2)} Å</span>
              <em className="cat-tag" style={{ color: CATEGORY_META[c.category].color, borderColor: CATEGORY_META[c.category].color }}>{CATEGORY_META[c.category].label}</em>
            </button>)}
          </div>
        </section>

        <section className="panel atom-panel">
          <span className="eyebrow">LIGAND ATOM</span>
          <div className="atom-row">
            <span className="atom-badge">{selectedAtom.slice(0, 1)}</span>
            <div><b>{selectedAtom}</b><small>点击模型中的原子或分析结果定位</small></div>
            <span className="atom-value">{visibleByAtom.has(selectedAtom) ? `${visibleContacts.filter(c => c.atomId === selectedAtom).length} 个接触` : selectedAtom === 'N1' ? '供体' : selectedAtom.startsWith('O') ? '受体' : '骨架'}</span>
          </div>
        </section>
      </aside>
    </main>
  </div>;
}
