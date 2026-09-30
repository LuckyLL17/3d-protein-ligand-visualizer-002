import { useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { ContactShadows, Line, OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

type DisplayMode = 'cartoon' | 'surface' | 'atoms';
type Residue = { id: number; code: string; name: string; x: number; y: number; z: number; color: string; role: 'active' | 'support' | 'loop' };
type Atom = { id: string; element: 'C' | 'N' | 'O' | 'S'; x: number; y: number; z: number; color: string; radius: number };
type InteractionCategory = 'hbond' | 'polar' | 'hydrophobic' | 'vdw';
type Contact = { residueId: number; residueCode: string; residueName: string; atomId: string; atomElement: Atom['element']; distance: number; category: InteractionCategory };
type NearestContact = { atomId: string; distance: number };
type AnalysisResult =
  | { status: 'ok'; contacts: Contact[]; nearest: Record<number, NearestContact>; skipped: string[] }
  | { status: 'missing'; missingResidues: string[]; missingAtoms: string[] };
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
const makeBond = (a: THREE.Vector3, b: THREE.Vector3) => [a, b];
// 配体在场景中的平移偏移，3D 渲染与相互作用计算共用，保证两处一致。
const LIGAND_OFFSET: [number, number, number] = [.1, .45, .15];
const HBOND_MAX_DISTANCE = 3.5;
const CATEGORY_ORDER: InteractionCategory[] = ['hbond', 'polar', 'hydrophobic', 'vdw'];
const CATEGORY_META: Record<InteractionCategory, { label: string; color: string }> = {
  hbond: { label: '氢键', color: '#67d7f5' },
  polar: { label: '极性接触', color: '#8fe6a4' },
  hydrophobic: { label: '疏水接触', color: '#f0c76e' },
  vdw: { label: '范德华接触', color: '#c0a9ef' }
};
const HBOND_RESIDUES = new Set(['ASP', 'GLU', 'HIS', 'SER', 'THR', 'TYR', 'ASN', 'GLN', 'LYS', 'ARG', 'CYS', 'TRP']);
const POLAR_RESIDUES = HBOND_RESIDUES;
const HYDROPHOBIC_RESIDUES = new Set(['PHE', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'TRP', 'GLY']);
const residueType = (code: string) => code.slice(0, 3).toUpperCase();
const hasValidCoords = (point: { x: number; y: number; z: number }) => [point.x, point.y, point.z].every(value => typeof value === 'number' && Number.isFinite(value));
const ligandScenePosition = (atom: Atom, rotation: number): [number, number, number] => {
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);
  return [atom.x * cos + atom.z * sin + LIGAND_OFFSET[0], atom.y + LIGAND_OFFSET[1], -atom.x * sin + atom.z * cos + LIGAND_OFFSET[2]];
};
const classifyContact = (residueCode: string, element: Atom['element'], distance: number): InteractionCategory => {
  const type = residueType(residueCode);
  if ((element === 'N' || element === 'O') && HBOND_RESIDUES.has(type) && distance <= HBOND_MAX_DISTANCE) return 'hbond';
  if ((element === 'N' || element === 'O' || element === 'S') && POLAR_RESIDUES.has(type)) return 'polar';
  if (element === 'C' && HYDROPHOBIC_RESIDUES.has(type)) return 'hydrophobic';
  return 'vdw';
};
function computeAnalysis(rotation: number, threshold: number): AnalysisResult {
  const validResidues = residues.filter(hasValidCoords);
  const validAtoms = ligandAtoms.filter(hasValidCoords);
  const missingResidues = residues.filter(item => !hasValidCoords(item)).map(item => item.code);
  const missingAtoms = ligandAtoms.filter(item => !hasValidCoords(item)).map(item => item.id);
  if (validResidues.length === 0 || validAtoms.length === 0) return { status: 'missing', missingResidues, missingAtoms };
  const contacts: Contact[] = [];
  const nearest: Record<number, NearestContact> = {};
  validResidues.forEach(residue => validAtoms.forEach(atom => {
    const [ax, ay, az] = ligandScenePosition(atom, rotation);
    const distance = Math.hypot(residue.x - ax, residue.y - ay, residue.z - az);
    const current = nearest[residue.id];
    if (!current || distance < current.distance) nearest[residue.id] = { atomId: atom.id, distance };
    if (distance <= threshold) contacts.push({ residueId: residue.id, residueCode: residue.code, residueName: residue.name, atomId: atom.id, atomElement: atom.element, distance, category: classifyContact(residue.code, atom.element, distance) });
  }));
  contacts.sort((a, b) => a.distance - b.distance);
  return { status: 'ok', contacts, nearest, skipped: [...missingResidues, ...missingAtoms] };
}
function Ligand({ rotation, selectedAtom, onSelect }: { rotation: number; selectedAtom: string; onSelect: (id: string) => void }) {
  const positions = useMemo(() => Object.fromEntries(ligandAtoms.map(atom => [atom.id, new THREE.Vector3(atom.x, atom.y, atom.z)])), []);
  return <group position={LIGAND_OFFSET} rotation={[0, rotation, 0]}>{bonds.map(([from, to]) => <Line key={`${from}-${to}`} points={makeBond(positions[from], positions[to])} color="#cbd9dc" lineWidth={3} />)}{ligandAtoms.map(atom => <mesh key={atom.id} position={[atom.x, atom.y, atom.z]} onClick={(event) => { event.stopPropagation(); onSelect(atom.id); }}><sphereGeometry args={[atom.radius * (selectedAtom === atom.id ? 1.35 : 1), 16, 12]} /><meshStandardMaterial color={atom.color} emissive={selectedAtom === atom.id ? atom.color : '#000000'} emissiveIntensity={selectedAtom === atom.id ? .7 : 0} /></mesh>)}</group>;
}
function ProteinModel({ displayMode, selectedResidue, setSelectedResidue, ligandRotation, selectedAtom, setSelectedAtom, contacts, showInteractions }: { displayMode: DisplayMode; selectedResidue: number; setSelectedResidue: (id: number) => void; ligandRotation: number; selectedAtom: string; setSelectedAtom: (id: string) => void; contacts: Contact[]; showInteractions: boolean }) {
  const showAtoms = displayMode === 'atoms';
  const surface = displayMode === 'surface';
  const atomPositions = useMemo(() => Object.fromEntries(ligandAtoms.map(atom => [atom.id, ligandScenePosition(atom, ligandRotation)])), [ligandRotation]);
  const residuePositions = useMemo(() => Object.fromEntries(residues.map(residue => [residue.id, [residue.x, residue.y, residue.z] as [number, number, number]])), []);
  return <Canvas shadows camera={{ position: [8, 5.5, 9], fov: 40 }}><color attach="background" args={['#09151d']} /><ambientLight intensity={1.2} /><directionalLight position={[5, 8, 5]} intensity={2.4} castShadow /><pointLight position={[-4, 3, 3]} color="#7fe0ff" intensity={18} distance={11} />
    <group rotation={[.08, -.25, 0]}>{!showAtoms && <><Line points={helixPoints} color="#6b9fe9" lineWidth={surface ? 12 : 5} /> <Line points={sheetPoints} color="#d58acb" lineWidth={surface ? 13 : 6} /></>}{showAtoms && residues.map(residue => <mesh key={residue.id} position={[residue.x, residue.y, residue.z]}><sphereGeometry args={[.22, 12, 10]} /><meshStandardMaterial color={residue.color} /></mesh>)}{residues.map(residue => <group key={`res-${residue.id}`} position={[residue.x, residue.y, residue.z]}><mesh onClick={(event) => { event.stopPropagation(); setSelectedResidue(residue.id); }}><sphereGeometry args={[selectedResidue === residue.id ? .28 : .18, 16, 12]} /><meshStandardMaterial color={residue.color} emissive={selectedResidue === residue.id ? residue.color : '#000'} emissiveIntensity={selectedResidue === residue.id ? .8 : 0} transparent opacity={surface ? .9 : 1} /></mesh><Text position={[0, .43, 0]} fontSize={.14} color={selectedResidue === residue.id ? '#ffe9a2' : '#8baeb9'} anchorX="center">{residue.code}</Text></group>)}<Ligand rotation={ligandRotation} selectedAtom={selectedAtom} onSelect={setSelectedAtom} />{showInteractions && contacts.map(contact => { const residuePosition = residuePositions[contact.residueId]; const atomPosition = atomPositions[contact.atomId]; if (!residuePosition || !atomPosition) return null; const active = contact.residueId === selectedResidue || contact.atomId === selectedAtom; const color = CATEGORY_META[contact.category].color; return <Line key={`${contact.residueId}-${contact.atomId}`} points={[residuePosition, atomPosition]} color={active ? color : `#${new THREE.Color(color).multiplyScalar(.45).getHexString()}`} lineWidth={active ? 2.6 : 1.1} dashed dashSize={.08} gapSize={.05} />; })}</group><ContactShadows position={[0, -1.5, 0]} opacity={.32} scale={10} blur={2.5} /><OrbitControls makeDefault minDistance={5} maxDistance={16} /></Canvas>;
}
function Metric({ label, value, note }: { label: string; value: string; note: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>; }
function InteractionPanel({ analysis, computing, threshold, onThresholdChange, categoryCounts, hiddenCategories, onToggleCategory, visibleContacts, selectedResidue, selectedAtom, onLocate }: { analysis: AnalysisResult | null; computing: boolean; threshold: number; onThresholdChange: (value: number) => void; categoryCounts: Record<InteractionCategory, number>; hiddenCategories: ReadonlySet<InteractionCategory>; onToggleCategory: (category: InteractionCategory) => void; visibleContacts: Contact[]; selectedResidue: number; selectedAtom: string; onLocate: (contact: Contact) => void }) {
  const missing = analysis?.status === 'missing' ? analysis : null;
  const state = !analysis ? 'loading' : analysis.status === 'missing' ? 'missing' : analysis.contacts.length === 0 ? 'empty' : visibleContacts.length === 0 ? 'hidden' : 'ready';
  return <section className="panel analysis-panel"><div className="panel-title"><div><span className="eyebrow">INTERACTION ANALYSIS</span><h3>相互作用分析</h3></div>{computing ? <span className="analysis-status">计算中…</span> : <span className="count">{visibleContacts.length}</span>}</div><div className="threshold-row"><div className="threshold-label"><span>距离阈值</span><b>{threshold.toFixed(2)} Å</b></div><input type="range" min={.2} max={3} step={.05} value={threshold} onChange={event => onThresholdChange(Number(event.target.value))} /></div><div className="category-row">{CATEGORY_ORDER.map(category => <button key={category} className={hiddenCategories.has(category) ? 'category-chip off' : 'category-chip'} onClick={() => onToggleCategory(category)} title={hiddenCategories.has(category) ? '点击显示该类别' : '点击隐藏该类别'}><i style={{ background: CATEGORY_META[category].color }} />{CATEGORY_META[category].label}<b>{categoryCounts[category]}</b></button>)}</div>{analysis?.status === 'ok' && analysis.skipped.length > 0 && <div className="analysis-warning">⚠ {analysis.skipped.length} 条记录坐标缺失（{analysis.skipped.join('、')}），已在分析中跳过</div>}{state === 'loading' && <div className="analysis-skeleton"><span /><span /><span /></div>}{state === 'missing' && missing && <div className="analysis-state"><span className="state-icon">!</span><b>坐标数据缺失</b><p>{missing.missingResidues.length} 个残基、{missing.missingAtoms.length} 个配体原子缺少有效三维坐标，无法计算相互作用，请检查输入数据。</p></div>}{state === 'empty' && <div className="analysis-state"><span className="state-icon">∅</span><b>阈值内没有任何接触</b><p>当前阈值 {threshold.toFixed(2)} Å 内未检测到残基–配体接触，可尝试增大距离阈值。</p></div>}{state === 'hidden' && <div className="analysis-state"><span className="state-icon">–</span><b>所有类别均已隐藏</b><p>点击上方类别标签即可重新显示对应接触。</p></div>}{state === 'ready' && <div className={computing ? 'contact-groups busy' : 'contact-groups'}>{CATEGORY_ORDER.map(category => { const list = visibleContacts.filter(contact => contact.category === category); if (list.length === 0) return null; return <div key={category} className="contact-group"><div className="contact-group-head"><i style={{ background: CATEGORY_META[category].color }} />{CATEGORY_META[category].label}<span>{list.length}</span></div>{list.map(contact => <button key={`${contact.residueId}-${contact.atomId}`} className={contact.residueId === selectedResidue && contact.atomId === selectedAtom ? 'contact-row active' : 'contact-row'} onClick={() => onLocate(contact)} title={`定位到 ${contact.residueName} ${contact.residueCode} 与配体原子 ${contact.atomId}`}><span className="contact-pair"><b>{contact.residueCode}</b><em>↔</em><b>{contact.atomId}</b></span><span className="contact-dist">{contact.distance.toFixed(2)} Å</span><span className="contact-go">定位 →</span></button>)}</div>; })}</div>}</section>;
}
export default function App() {
  const [mode, setMode] = useState<DisplayMode>('cartoon');
  const [selectedResidue, setSelectedResidue] = useState(102);
  const [selectedAtom, setSelectedAtom] = useState('N1');
  const [ligandRotation, setLigandRotation] = useState(0.2);
  const [showInteractions, setShowInteractions] = useState(true);
  const [threshold, setThreshold] = useState(1.5);
  const [hiddenCategories, setHiddenCategories] = useState<ReadonlySet<InteractionCategory>>(new Set());
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [computing, setComputing] = useState(true);
  useEffect(() => {
    setComputing(true);
    const timer = window.setTimeout(() => { setAnalysis(computeAnalysis(ligandRotation, threshold)); setComputing(false); }, 320);
    return () => window.clearTimeout(timer);
  }, [ligandRotation, threshold]);
  const residue = residues.find(item => item.id === selectedResidue) ?? residues[2];
  const allContacts = useMemo(() => (analysis?.status === 'ok' ? analysis.contacts : []), [analysis]);
  const visibleContacts = useMemo(() => allContacts.filter(contact => !hiddenCategories.has(contact.category)), [allContacts, hiddenCategories]);
  const categoryCounts = useMemo(() => { const counts: Record<InteractionCategory, number> = { hbond: 0, polar: 0, hydrophobic: 0, vdw: 0 }; allContacts.forEach(contact => { counts[contact.category] += 1; }); return counts; }, [allContacts]);
  const contactsByResidue = useMemo(() => { const map = new Map<number, Contact[]>(); visibleContacts.forEach(contact => { const list = map.get(contact.residueId) ?? []; list.push(contact); map.set(contact.residueId, list); }); return map; }, [visibleContacts]);
  const interactingResidues = useMemo(() => new Set(allContacts.map(contact => contact.residueId)).size, [allContacts]);
  const selectedContacts = contactsByResidue.get(selectedResidue) ?? [];
  const nearestSelected = analysis?.status === 'ok' ? analysis.nearest[selectedResidue] : undefined;
  const selectedSummary = CATEGORY_ORDER.filter(category => selectedContacts.some(contact => contact.category === category)).map(category => `${CATEGORY_META[category].label}×${selectedContacts.filter(contact => contact.category === category).length}`).join(' · ');
  const toggleCategory = (category: InteractionCategory) => setHiddenCategories(previous => { const next = new Set(previous); if (next.has(category)) next.delete(category); else next.add(category); return next; });
  const locateContact = (contact: Contact) => { setSelectedResidue(contact.residueId); setSelectedAtom(contact.atomId); };
  return <div className="app"><header className="topbar"><div className="brand"><div className="brand-icon">P</div><div><span className="eyebrow">STRUCTURAL BIOLOGY / ANALYSIS WORKSPACE</span><h1>Protein Atlas <span>·</span> 配体结合可视化</h1></div></div><div className="header-meta"><span className="dataset">PDB-LITE / 1MOL</span><button onClick={() => setShowInteractions(value => !value)}>{showInteractions ? '隐藏相互作用' : '显示相互作用'}</button><button onClick={() => setLigandRotation(value => value + .35)}>旋转配体 ↻</button></div></header><main className="layout"><section className="model-card"><div className="model-head"><div><span className="eyebrow">MOLECULAR VIEWER</span><h2>蛋白质结合口袋 · Ligand LIG-17</h2></div><div className="mode-switch">{(['cartoon','surface','atoms'] as DisplayMode[]).map(item => <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{item === 'cartoon' ? '卡通带状' : item === 'surface' ? '表面模型' : '原子视图'}</button>)}</div></div><div className="canvas-wrap"><ProteinModel displayMode={mode} selectedResidue={selectedResidue} setSelectedResidue={setSelectedResidue} ligandRotation={ligandRotation} selectedAtom={selectedAtom} setSelectedAtom={setSelectedAtom} contacts={visibleContacts} showInteractions={showInteractions} /><div className="viewer-legend"><span><i className="dot protein" />蛋白骨架</span><span><i className="dot ligand" />配体原子</span><span><i className="dot interaction" />相互作用</span></div><div className="viewer-hint">左键旋转 · 滚轮缩放 · 点击残基或原子</div></div></section><aside className="sidebar"><section className="panel intro-panel"><span className="eyebrow">ACTIVE SITE</span><h3>结合口袋分析</h3><p>当前模型显示 7 个关键残基和一个小分子配体。通过选择残基查看空间关系与作用类型。</p><div className="metrics"><Metric label="口袋体积" value="486 Å³" note="预测区域" /><Metric label="结合残基" value={analysis ? String(interactingResidues) : '…'} note="当前阈值内" /><Metric label="氢键" value={analysis ? String(categoryCounts.hbond) : '…'} note="几何筛选 · 阈值内" /></div></section><InteractionPanel analysis={analysis} computing={computing} threshold={threshold} onThresholdChange={setThreshold} categoryCounts={categoryCounts} hiddenCategories={hiddenCategories} onToggleCategory={toggleCategory} visibleContacts={visibleContacts} selectedResidue={selectedResidue} selectedAtom={selectedAtom} onLocate={locateContact} /><section className="panel"><div className="panel-title"><div><span className="eyebrow">RESIDUES</span><h3>关键残基</h3></div><span className="count">{residues.length}</span></div><div className="residue-list">{residues.map(item => { const itemContacts = contactsByResidue.get(item.id) ?? []; const categories = CATEGORY_ORDER.filter(category => itemContacts.some(contact => contact.category === category)); return <button key={item.id} className={selectedResidue === item.id ? 'residue selected' : 'residue'} onClick={() => setSelectedResidue(item.id)}><span className="residue-color" style={{ background: item.color }} /><span><b>{item.code}</b><small>{item.name}</small></span>{itemContacts.length > 0 && <span className="residue-contacts">{categories.map(category => <i key={category} style={{ background: CATEGORY_META[category].color }} />)}<b>{itemContacts.length}</b></span>}<em>{item.role === 'active' ? 'ACTIVE' : item.role === 'support' ? 'SUPPORT' : 'LOOP'}</em></button>; })}</div></section><section className="panel detail-panel"><span className="eyebrow">SELECTED RESIDUE</span><div className="detail-title"><h3>{residue.code}</h3><span>{residue.name}</span></div><div className="detail-grid"><div><span>空间坐标</span><b>{residue.x.toFixed(2)} / {residue.y.toFixed(2)} / {residue.z.toFixed(2)} Å</b></div><div><span>残基角色</span><b>{residue.role === 'active' ? '结合活性位点' : residue.role === 'support' ? '口袋支撑' : '柔性环区'}</b></div><div><span>配体距离</span><b>{nearestSelected ? `${nearestSelected.distance.toFixed(2)} Å · ${nearestSelected.atomId}` : analysis ? '坐标缺失' : '计算中…'}</b></div><div><span>相互作用</span><b>{!analysis ? '计算中…' : selectedContacts.length === 0 ? '阈值内无接触' : selectedSummary}</b></div></div>{selectedContacts.length > 0 && <div className="detail-contacts"><span>接触原子 · 点击定位</span><div>{selectedContacts.map(contact => <button key={contact.atomId} className={selectedAtom === contact.atomId ? 'contact-atom active' : 'contact-atom'} onClick={() => setSelectedAtom(contact.atomId)}><i style={{ background: CATEGORY_META[contact.category].color }} />{contact.atomId} · {contact.distance.toFixed(2)} Å</button>)}</div></div>}</section><section className="panel atom-panel"><span className="eyebrow">LIGAND ATOM</span><div className="atom-row"><span className="atom-badge">{selectedAtom.slice(0,1)}</span><div><b>{selectedAtom}</b><small>点击模型中的原子切换</small></div><span className="atom-value">{selectedAtom === 'N1' ? '供体' : selectedAtom.startsWith('O') ? '受体' : '骨架'}</span></div></section></aside></main></div>;
}
