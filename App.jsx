/* Crew Health frontend (single JSX file, compiled in the browser by Babel: no build step).
   React and ReactDOM come from CDN globals; data comes from the JSON API in server.js. */
const { useState, useEffect } = React;

/* ============================== helpers ============================== */
const api = (p, m = 'GET', b) => fetch('/api/' + p, { method: m, headers: { 'Content-Type': 'application/json' }, body: b ? JSON.stringify(b) : undefined })
  .then(async r => { const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.error || r.statusText); return j; });
const todayISO = () => new Date().toLocaleDateString('en-CA');
const addDays = (d, n) => { const t = new Date(d + 'T00:00:00Z'); t.setUTCDate(t.getUTCDate() + n); return t.toISOString().slice(0, 10); };
const dayDiff = (a, b) => Math.round((new Date(b + 'T00:00:00Z') - new Date(a + 'T00:00:00Z')) / 864e5);
const fd = (d, o) => new Date(d + 'T00:00:00').toLocaleDateString('en-GB', o || { weekday: 'short', day: 'numeric', month: 'short' });
const num = (x, d = 1) => x == null || !isFinite(x) ? '-' : String(+(+x).toFixed(Math.abs(x) >= 100 ? 0 : d));
const r2 = x => +x.toFixed(2);
const REF = { rda: 'RDA', limit: 'limit', avg: 'average', pre: 'baseline weight' };
const flagLvl = (f, v) => {                       // same rule as the server: 2 = clearly off, 1 = below benchmark
  if (v == null) return 0; const r = f.ref;
  if (f.t === 'pre') return v < .95 * r ? 2 : v < .98 * r ? 1 : 0;
  if (f.t === 'limit') return v > r ? 2 : 0;
  if (f.bad === 'high') return v > r * 1.15 ? 2 : v > r ? 1 : 0;
  return v < (f.t === 'rda' ? .7 : .8) * r ? 2 : v < r ? 1 : 0;
};
const vsTxt = (v, ref, w) => `${num(Math.abs(v - ref) / ref * 100, 0)}% ${v < ref ? 'below' : 'above'} ${w}`;
const lvlOf = n => n <= 2 ? 'ok' : n <= 5 ? 'warn' : 'bad';
const ETYPES = ['None', 'Cardio / aerobic', 'Strength / resistance', 'Mixed (cardio and strength)'];
const NO_CHART = new Set(['resist_min', 'aerobic_min', 'weight_kg', 'radiation_msv', 'noise_dba', 'mood', 'social_min', 'co2_ppm', 'spo2_pct']);   // factors still logged, but not drawn on the Charts page
const EX_TIME_KEYS = new Set(['exercise_min', 'resist_min', 'aerobic_min']);   // entered in the Exercise block, not in the generic list
const exNum = v => v === '' || v == null || isNaN(+v) ? null : +v;
/** Exercise type -> which parts were done. Total time = cardio time + strength time. */
const withExTotals = v => {
  const t = v.exercise_type === '' || v.exercise_type == null ? '' : String(v.exercise_type);
  if (t === '') return { ...v, exercise_min: '' };
  if (t === '0') return { ...v, exercise_min: 0 };
  const parts = [];
  if (t === '1' || t === '3') parts.push(exNum(v.aerobic_min));
  if (t === '2' || t === '3') parts.push(exNum(v.resist_min));
  const mins = parts.filter(m => m != null);
  return { ...v, exercise_min: mins.length ? r2(mins.reduce((s, m) => s + m, 0)) : '' };
};
/** Days saved with a single total time: show that time under the matching type (cardio or strength). */
const initEx = v0 => {
  const v = { ...v0 }, has = k => v[k] != null && v[k] !== '' && !isNaN(+v[k]);
  const t = has('exercise_type') ? +v.exercise_type : null;
  if (t > 0 && !has('aerobic_min') && !has('resist_min') && has('exercise_min')) {
    if (t === 1) v.aerobic_min = v.exercise_min;
    else if (t === 2) v.resist_min = v.exercise_min;
  }
  return v;
};
const groupBy = fs => { const g = []; fs.forEach(f => { let x = g.find(a => a[0] === f.g); if (!x) g.push(x = [f.g, []]); x[1].push(f); }); return g; };
const useDays = init => { const [raw, setRaw] = useState(String(init)); return [raw, setRaw, Math.min(365, Math.max(1, Math.floor(+raw) || init))]; };
const DaysInput = ({ raw, setRaw }) => <label className="inl">Number of days
  <input type="number" min="1" max="365" step="1" value={raw} onChange={e => setRaw(e.target.value)} /></label>;

/* ============================== styles (white + medical green) ============================== */
const DARK = '--bg:#0a1620;--card:#112130;--ink:#e6f1f5;--mut:#8aa4b3;--line:#1d3446;--acc:#14b8a6;--acc2:#3b82f6;--deep:#0f766e;--tint:#143342;--ok:#4ade80;--warn:#fbbf24;--bad:#fb7185;--sh:0 14px 34px -14px rgba(0,0,0,.6)';
const CSS = `
:root{box-sizing:border-box;--bg:#e8f5f6;--card:#fff;--ink:#0f2b3d;--mut:#6a8493;--line:#e0edef;--acc:#14b8a6;--acc2:#2f7bff;--deep:#0e8f84;--tint:#e8f7f5;--ok:#12a56b;--warn:#ee8f1f;--bad:#e5484d;--sh:0 14px 34px -14px rgba(18,96,120,.25),0 2px 6px rgba(18,96,120,.05);--grad:linear-gradient(135deg,var(--acc),var(--acc2))}
@media (prefers-color-scheme:dark){:root{${DARK}}}
*,*:before,*:after{box-sizing:inherit}
body{margin:0;background:radial-gradient(900px 500px at 100% -10%,color-mix(in srgb,var(--acc2) 14%,transparent),transparent) fixed,var(--bg);color:var(--ink);font:15px/1.5 "Plus Jakarta Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h2{font-size:19px;margin:20px 0 12px;letter-spacing:-.01em}h3{font-size:15px;margin:22px 0 10px}p{margin:6px 0}
button,select,input{font:inherit;color:inherit}button{cursor:pointer;border:1px solid var(--line);background:var(--card);border-radius:14px;padding:8px 14px;transition:transform .15s,box-shadow .15s}
button:hover{box-shadow:0 6px 16px -8px rgba(18,96,120,.35)}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.mut{color:var(--mut)}.err{color:var(--bad)}
.app{display:grid;grid-template-columns:250px minmax(0,1fr)}
nav{position:sticky;top:14px;align-self:start;height:calc(100vh - 28px);margin:14px 0 14px 14px;display:flex;flex-direction:column;gap:6px;padding:22px 14px;background:var(--card);border-radius:28px;box-shadow:var(--sh)}
.brand{display:flex;align-items:center;gap:10px;margin:0 6px 20px;font-size:18px}
.logo{width:38px;height:38px;border-radius:13px;background:var(--grad);position:relative;flex:none;box-shadow:0 8px 18px -6px var(--acc2)}
.logo:before,.logo:after{content:"";position:absolute;background:#fff;border-radius:2px;left:50%;top:50%;transform:translate(-50%,-50%)}
.logo:before{width:16px;height:4px}.logo:after{width:4px;height:16px}
.nv{display:flex;align-items:center;gap:12px;border:0;background:none;padding:12px 14px;border-radius:16px;font-weight:600;color:var(--mut);text-align:left;box-shadow:none}
.nv:hover{background:var(--tint);box-shadow:none}.nv.on{background:var(--grad);color:#fff;box-shadow:0 10px 20px -10px var(--acc2)}
.nv svg{width:22px;height:22px;fill:none;stroke:currentColor;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round;flex:none}
.foot{margin-top:auto;font-size:12px}
.hero{position:relative;overflow:hidden;margin:14px 14px 0;background:var(--grad);color:#fff;padding:30px 32px 82px;border-radius:30px}
.hero:before,.hero:after{content:"";position:absolute;border-radius:50%;background:rgba(255,255,255,.12);pointer-events:none}
.hero:before{width:280px;height:280px;right:-60px;top:-120px}.hero:after{width:180px;height:180px;right:200px;bottom:-110px}
.hw{position:relative;display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap}
.hero h1{margin:0;font-size:30px;letter-spacing:-.02em}.hero p{margin:6px 0 0;max-width:62ch;opacity:.92}
.hp{display:flex;align-items:center;gap:12px;padding:10px 16px 10px 10px;border-radius:99px;background:rgba(255,255,255,.18);backdrop-filter:blur(8px)}
.hp small{display:block;font-size:12px;opacity:.85;line-height:1.2}.hp b{line-height:1.2}
.av{width:40px;height:40px;border-radius:50%;background:#fff;color:var(--deep);display:grid;place-items:center;font-weight:800}
.pg{padding:0 28px 40px;margin-top:-50px;position:relative}
.card{background:var(--card);border:1px solid var(--line);border-radius:24px;padding:18px;box-shadow:var(--sh)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.ch{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-bottom:6px}.ch .mut{font-size:12px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-bottom:16px}
.st{display:flex;align-items:center;gap:14px}.st span{font-size:13px;color:var(--mut);display:block;line-height:1.3}.st b{display:block;font-size:24px;letter-spacing:-.01em;line-height:1.2}.st small{font-size:12px;color:var(--mut)}
.ring{position:relative;flex:none;display:grid;place-items:center}.ring svg{position:absolute;inset:0}.ring span{position:relative;font-size:13px;font-weight:800;color:var(--ink)}
.duo{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-bottom:16px}
.wk{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin:10px 0 14px}
.wd{display:grid;justify-items:center;gap:2px;padding:10px 2px 8px;border-radius:18px;border:0;background:var(--tint)}
.wd small{font-size:11px;color:var(--mut)}.wd b{font-size:18px;line-height:1.2}.wd i{width:8px;height:8px;border-radius:50%;background:var(--c,var(--line))}
.wd.now{background:var(--grad);color:#fff}.wd.now small{color:#fff;opacity:.85}.wd.now i{box-shadow:0 0 0 2px #fff}
.area{width:100%;display:block;margin:6px 0 2px}.alab{display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}.alab b{color:var(--ink)}
.logc{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}.logc b{margin-right:auto}
input[type=number],input[type=text],input[type=date],select{padding:9px 12px;border-radius:12px;border:1px solid var(--line);background:var(--card)}
.pri{background:var(--grad);border-color:transparent;color:#fff;font-weight:700;box-shadow:0 10px 20px -10px var(--acc2)}.pri:disabled{opacity:.6}
.leg{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--mut)}
.leg span:before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--c);margin-right:6px}.leg .sy:before{display:none}
.leg u,.db u{text-decoration:none;color:var(--bad);font-weight:700}
.ok{--c:var(--ok)}.warn{--c:var(--warn)}.bad{--c:var(--bad)}.none{--c:var(--line)}
.days{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:10px}
.db{position:relative;display:grid;justify-items:center;gap:1px;padding:12px 4px 10px;border-radius:20px;border:0;background:color-mix(in srgb,var(--c) 9%,var(--card));box-shadow:inset 0 -4px 0 var(--c);transition:transform .15s}
.db:hover{transform:translateY(-3px)}.db small{font-size:12px;color:var(--mut)}.db b{font-size:24px;line-height:1.15}
.db span{font-size:11px;margin-top:3px;padding:1px 9px;border-radius:99px;background:color-mix(in srgb,var(--c) 16%,transparent);color:var(--c);font-weight:700}
.db u{position:absolute;top:8px;right:10px;font-size:13px}
.seg{display:inline-flex;gap:4px;padding:5px;background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--sh)}
.seg button{border:0;background:none;padding:9px 18px;font-weight:600;color:var(--mut);border-radius:14px;box-shadow:none}.seg .on{background:var(--grad);color:#fff}
.bar2{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}.inl{display:flex;gap:8px;align-items:center;font-weight:600}.inl input{width:92px}
.tag{display:inline-block;padding:1px 10px;border-radius:99px;font-size:12px;font-weight:700;border:1px solid var(--line);margin:0 4px 2px 0}
.tag.RED,.tag.bad{color:var(--bad);border-color:transparent;background:color-mix(in srgb,var(--bad) 12%,transparent)}.tag.YELLOW,.tag.warn{color:var(--warn);border-color:transparent;background:color-mix(in srgb,var(--warn) 14%,transparent)}.tag.GREEN,.tag.ok{color:var(--ok);border-color:transparent;background:color-mix(in srgb,var(--ok) 13%,transparent)}.tag.INFO,.tag.MISSING{color:var(--mut)}
.al{list-style:none;margin:0;padding:0}.al li{padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}.al li:last-child{border:0}
.dayc{margin-bottom:12px}.dayc.miss{border-style:dashed}.dayc header{display:flex;gap:10px;align-items:center;margin-bottom:6px}.dayc header b{margin-right:auto}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.chip{border-radius:99px;padding:6px 14px;background:var(--tint);border-color:transparent}.chip:hover{background:color-mix(in srgb,var(--acc) 18%,var(--tint))}.chip.red{color:var(--bad)}
.sq{width:100%;padding:14px 18px;border-radius:18px;background:var(--tint);border:1px solid transparent}
.sy{display:inline-flex;gap:6px;align-items:center;padding:4px 6px 4px 12px;border-radius:99px;background:var(--tint)}.sy.red{color:var(--bad)}.sy select{padding:2px 4px}.sy button{border:0;background:none;padding:2px 6px;box-shadow:none}
.set{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}.set label{display:grid;gap:3px;font-size:13px;font-weight:600}
.verdict{border-left:5px solid var(--acc)}.verdict.none{border-left-color:var(--warn)}.verdict h2{margin:0 0 6px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;margin-top:12px}
.tw{overflow-x:auto}.tb{width:100%;border-collapse:separate;border-spacing:0;font-size:13px}.tb th,.tb td{padding:10px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
.tb thead th{background:var(--tint);color:var(--mut);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.04em}.tb thead th:first-child{border-radius:12px 0 0 12px}.tb thead th:last-child{border-radius:0 12px 12px 0}
.tb tbody tr{cursor:pointer}.tb tbody tr:hover{background:color-mix(in srgb,var(--tint) 55%,transparent)}.tb tr.sel{background:var(--tint)}
.sbar{display:inline-block;vertical-align:middle;width:90px;height:8px;margin-right:8px;background:var(--tint);border-radius:6px;overflow:hidden}.sbar i{display:block;height:100%;border-radius:6px;background:var(--grad)}
.cmp{display:grid;gap:8px;margin-top:8px}.cr{display:grid;grid-template-columns:150px 1fr 96px;gap:8px;align-items:center;font-size:13px}
.bar{height:10px;background:var(--tint);border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;border-radius:6px;background:var(--acc)}.bar i.ref{background:var(--bad)}.bar i.w{background:var(--warn)}
.kv{display:grid;gap:4px;font-size:13px;margin-top:8px}.kv div{padding:6px 0;border-bottom:1px solid var(--line)}
svg{width:100%;height:auto;display:block}.ax{stroke:var(--line)}.tx{fill:var(--mut);font-size:10px}
.ln{fill:none;stroke:var(--acc);stroke-width:1.8}.rf{stroke:var(--bad);stroke-dasharray:5 4;stroke-width:1.4}.nm{stroke:var(--ok);stroke-width:1.4}.bd{fill:var(--ok);opacity:.13}
.dn{fill:var(--acc);fill-opacity:.2;stroke:var(--acc)}.pf{stroke:#3b82c4;stroke-dasharray:2 3;stroke-width:1.4}.pt{fill:var(--acc)}.pt.w{fill:var(--warn)}.pt.b{fill:var(--bad)}
.cb{fill:var(--acc);opacity:.18}.cl{fill:none;stroke:var(--acc);stroke-width:2.2}.lowz{fill:var(--warn);opacity:.14}
.gauge{position:relative;height:14px;background:var(--tint);border-radius:8px;overflow:hidden;margin:10px 0}.gauge i{display:block;height:100%;background:var(--grad);border-radius:8px}.gauge i.w{background:var(--warn)}.gauge i.b{background:var(--bad)}
.gauge s{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);opacity:.35}
.modal{position:fixed;inset:0;background:rgba(8,30,40,.6);backdrop-filter:blur(4px);display:flex;justify-content:center;align-items:flex-start;padding:16px 12px;overflow:auto;z-index:20}
.sheet{background:var(--bg);border-radius:28px;max-width:980px;width:100%;padding:22px;position:relative}.sheet h2{margin:0 80px 6px 0}.x{position:absolute;top:14px;right:14px}
.fg{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.fi{display:grid;gap:3px;font-size:13px;font-weight:600}.fi i{font-style:normal;color:var(--mut);font-weight:400}
.fi em{font-style:normal;font-size:11.5px;font-weight:400;color:var(--mut)}.fi em.ok{color:var(--ok)}.fi em.warn{color:var(--warn)}.fi em.bad{color:var(--bad)}
.act{display:flex;gap:10px;margin-top:16px}.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.howto summary{cursor:pointer}.howto p{margin:8px 0}.sm{font-size:12px}
.ogrid{grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.oc{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;align-items:center;width:100%;text-align:left;padding:16px;border-radius:24px;background:var(--card);border:1px solid var(--line);box-shadow:var(--sh)}.oc:hover{transform:translateY(-2px)}.oc.sel{outline:2px solid var(--acc);outline-offset:2px}
.oc p{grid-column:1/-1;margin:6px 0 0}
.oc b{display:block}.oc .big{font-size:24px;font-weight:800;letter-spacing:-.02em;line-height:1.15}.oc .big small{font-size:12px;color:var(--mut);font-weight:500;margin-left:3px}
.dl{font-size:12px;font-weight:700;color:var(--mut)}.dl.up{color:var(--ok)}.dl.dn2{color:var(--bad)}
.pbar{height:8px;background:var(--tint);border-radius:6px;overflow:hidden;margin:6px 0}.pbar i{display:block;height:100%;border-radius:6px;background:var(--c,var(--acc))}.spark{height:34px;margin:4px 0}
@media(max-width:760px){.app{grid-template-columns:1fr}
nav{position:fixed;top:auto;bottom:0;left:0;right:0;height:auto;margin:0;flex-direction:row;padding:6px 10px calc(6px + env(safe-area-inset-bottom,0px));border-radius:22px 22px 0 0;z-index:10}
.brand,.foot{display:none}.nv{flex:1;flex-direction:column;align-items:center;justify-content:center;gap:2px;font-size:12px;padding:6px 4px}
.hero{margin:8px 8px 0;padding:22px 18px 62px;border-radius:24px}.hero h1{font-size:24px}.pg{padding:0 12px 96px}.modal{padding:0}.sheet{border-radius:0;min-height:100%}.cr{grid-template-columns:110px 1fr 80px}}`;
document.head.appendChild(Object.assign(document.createElement('style'), { textContent: CSS }));
const SPACE_CSS = `
.ln.sp{stroke:#3b82f6}
.dn.sp{fill:#3b82f6;fill-opacity:.3;stroke:#3b82f6;stroke-width:1.5}
.spnm{stroke:#3b82f6;stroke-dasharray:5 4;stroke-width:1.4}
.cb.sp{fill:#3b82f6;opacity:.25}
.bd.sp{fill:#3b82f6;opacity:.16}
.cl.sp{stroke:#3b82f6}
.urs{display:flex;gap:8px;flex-wrap:wrap}
.ur{width:46px;height:46px;border-radius:12px;font-weight:700;border:2px solid transparent;padding:0}
.ur.on{border-color:var(--ink);outline:2px solid var(--acc);outline-offset:1px}
`;

document.head.appendChild(Object.assign(document.createElement('style'), { textContent: SPACE_CSS }));
const Ring = ({ pct, size = 64, w = 8, c = 'var(--acc)', children }) => {
  const r = (size - w) / 2, L = 2 * Math.PI * r, p = Math.min(100, Math.max(0, pct || 0)), m = size / 2;
  return <div className="ring" style={{ width: size, height: size }}>
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: size, height: size }} aria-hidden="true">
      <circle cx={m} cy={m} r={r} fill="none" strokeWidth={w} style={{ stroke: 'var(--tint)' }} />
      <circle cx={m} cy={m} r={r} fill="none" strokeWidth={w} strokeLinecap="round" strokeDasharray={`${L * p / 100} ${L}`} transform={`rotate(-90 ${m} ${m})`} style={{ stroke: c }} />
    </svg>
    <span>{children}</span>
  </div>;
};
const Area = ({ vals, line, c = 'var(--acc)', h = 76 }) => {
  if (vals.length < 2) return <p className="mut sm">Not enough data yet.</p>;
  const W = 300, lo = Math.min(...vals, line ?? Infinity), hi = Math.max(...vals, line ?? -Infinity), sp = hi - lo || 1,
        X = i => i / (vals.length - 1) * W, Y = v => 6 + (1 - (v - lo) / sp) * (h - 12),
        d = vals.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join('');
  return <svg className="area" viewBox={`0 0 ${W} ${h}`} preserveAspectRatio="none" style={{ height: h }} aria-hidden="true">
    <path d={d + `L${W} ${h}L0 ${h}Z`} style={{ fill: c, opacity: .16 }} />
    {line != null && <line className="rf" x1="0" x2={W} y1={Y(line)} y2={Y(line)} vectorEffect="non-scaling-stroke" />}
    <path d={d} vectorEffect="non-scaling-stroke" strokeLinejoin="round" style={{ fill: 'none', stroke: c, strokeWidth: 2.4 }} />
  </svg>;
};
/* ============================== charts (plain SVG) ============================== */
/** Daily values over time, with the benchmark (red dashed), your normal (green, shaded +-1 SD) and your own probability curve on the right. */
function Chart({ f, pts, start, n, space }) {
  const W = 460, H = 156, L = 40, B = 20, T = 8, pw = 326, dx = L + pw + 12, dw = W - dx - 6;

  const spOn = !!(space && space.space_start);

  const earthPts = spOn ? pts.filter(p => p.date < space.space_start) : pts;
  const spacePts = spOn ? pts.filter(p => p.date >= space.space_start) : [];

  const greenNormal = spOn
    ? (f.earth_normal != null ? f.earth_normal : f.normal)
    : f.normal;

  const greenSd = spOn
    ? (f.earth && f.earth.sd != null ? f.earth.sd : f.sd)
    : f.sd;

  const spReady = spOn && space.space_ready_30,
        bNormal = spReady && f.space && f.space.sd != null ? f.space.normal : null,
        bSd = bNormal != null ? f.space.sd : null,
        sAvg = spOn && f.space_avg != null ? f.space_avg : null;

  const cand = [
    ...pts.map(p => p.v),
    f.ref,
    ...(greenNormal != null && greenSd != null ? [greenNormal - greenSd, greenNormal + greenSd] : []),
    ...(sAvg != null ? [sAvg] : []),
    ...(bNormal != null ? [bNormal - bSd, bNormal + bSd] : [])
  ];

  let lo = Math.min(...cand), hi = Math.max(...cand);
  const pad = (hi - lo || 1) * .1;
  lo = Math.max(0, lo - pad);
  hi += pad;

  const y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B),
        x = d => L + dayDiff(start, d) / Math.max(1, n - 1) * pw;

  const path = arr => arr.map((p, i) => (i ? 'L' : 'M') + x(p.date).toFixed(1) + ' ' + y(p.v).toFixed(1)).join('');

  const kGreen = spOn
    ? (f.earth && f.earth.kde ? f.earth.kde : null)
    : f.kde;

  const kBlue = spReady && f.space && f.space.kde ? f.space.kde : null;

  const my = Math.max(
    kGreen ? Math.max(...kGreen.y) : 0,
    kBlue ? Math.max(...kBlue.y) : 0,
    1e-9
  );

  const densityPath = k => {
    if (!k) return '';

    const kp = k.x.map((v, i) => [v, k.y[i]]).filter(([v]) => v >= lo && v <= hi);
    if (!kp.length) return '';

    return kp.map(([v, d], i) =>
      (i ? 'L' : 'M') + (dx + d / my * dw).toFixed(1) + ' ' + y(v).toFixed(1)
    ).join('') +
    `L${dx} ${y(kp[kp.length - 1][0]).toFixed(1)}L${dx} ${y(kp[0][0]).toFixed(1)}Z`;
  };

  const eLo = greenNormal != null && greenSd != null ? greenNormal - greenSd : null,
        sLo = bNormal != null ? bNormal - bSd : null;

  const ptCls = (p, inSpace) => {
    const useS = inSpace && sLo != null,
          need = (eLo != null ? 1 : 0) + (useS ? 1 : 0),
          n = (eLo != null && p.v < eLo ? 1 : 0) + (useS && p.v < sLo ? 1 : 0);

    return !n ? '' : n >= need ? 'b' : 'w';
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${f.l} over time`}>
      {[lo, (lo + hi) / 2, hi].map((t, i) => (
        <g key={i}>
          <line className="ax" x1={L} x2={L + pw} y1={y(t)} y2={y(t)} />
          <text className="tx" x={L - 5} y={y(t) + 3} textAnchor="end">{num(t)}</text>
        </g>
      ))}

      {greenNormal != null && greenSd != null && (
        <rect
          className="bd"
          x={L}
          width={pw}
          y={y(greenNormal + greenSd)}
          height={Math.max(0, y(greenNormal - greenSd) - y(greenNormal + greenSd))}
        />
      )}

      {bNormal != null && (
        <rect
          className="bd sp"
          x={L}
          width={pw}
          y={y(bNormal + bSd)}
          height={Math.max(0, y(bNormal - bSd) - y(bNormal + bSd))}
        />
      )}

      {greenNormal != null && (
        <line className="nm" x1={L} x2={L + pw} y1={y(greenNormal)} y2={y(greenNormal)} />
      )}

      {sAvg != null && (
        <line className="spnm" x1={L} x2={L + pw} y1={y(sAvg)} y2={y(sAvg)} />
      )}

      <line className="rf" x1={L} x2={L + pw} y1={y(f.ref)} y2={y(f.ref)} />

      {earthPts.length > 1 && <path className="ln" d={path(earthPts)} />}
      {spacePts.length > 1 && <path className="ln sp" d={path(spacePts)} />}

      {earthPts.map(p => {
        return (
          <circle
            key={p.date}
            className={'pt ' + ptCls(p, false)}
            cx={x(p.date)}
            cy={y(p.v)}
            r="2.8"
          >
            <title>{fd(p.date)}: {num(p.v)} {f.u}</title>
          </circle>
        );
      })}

      {spacePts.map(p => {
        return (
          <circle
            key={p.date}
            className={'pt ' + ptCls(p, true)}
            cx={x(p.date)}
            cy={y(p.v)}
            r="2.8"
          >
            <title>{fd(p.date)}: {num(p.v)} {f.u}</title>
          </circle>
        );
      })}

      {kGreen && <path className="dn" d={densityPath(kGreen)} />}
      {kBlue && <path className="dn sp" d={densityPath(kBlue)} />}

      <text className="tx" x={L} y={H - 4}>{fd(start, { day: 'numeric', month: 'short' })}</text>
      <text className="tx" x={L + pw} y={H - 4} textAnchor="end">{fd(addDays(start, n - 1), { day: 'numeric', month: 'short' })}</text>
      <text className="tx" x={dx + dw / 2} y={H - 4} textAnchor="middle">your curve</text>
    </svg>
  );
}

/* ============================== symptom keyword search (matches against the symptom table in the DB) ============================== */
function SymSearch({ kb, onPick }) {
  const [q, setQ] = useState(''), [hits, setHits] = useState([]);
  useEffect(() => { if (!q.trim()) { setHits([]); return; } const t = setTimeout(() => api('match?q=' + encodeURIComponent(q)).then(setHits).catch(() => { }), 180); return () => clearTimeout(t); }, [q]);
  const strong = hits.filter(h => !h.weak), weak = hits.filter(h => h.weak), list = q.trim() ? strong : kb.slice(0, 14);
  return <div>
    <input className="sq" type="text" value={q} onChange={e => setQ(e.target.value)} placeholder="Describe how you feel, e.g. tired, leg cramps, headache" aria-label="Describe your symptoms" autoComplete="off" />
    <div className="chips">{list.map(s => <button key={s.key} className={'chip' + (s.red ? ' red' : '')} onClick={() => { onPick(s.key); setQ(''); }}>{s.label}{s.red ? ' !' : ''}</button>)}</div>
    {q.trim() && !strong.length && weak.length > 0 && <div><p className="mut">No exact match. Did you mean:</p><div className="chips">{weak.slice(0, 5).map(s => <button key={s.key} className={'chip' + (s.red ? ' red' : '')} style={{ borderStyle: 'dashed' }} onClick={() => { onPick(s.key); setQ(''); }}>{s.label}</button>)}</div></div>}
    {q.trim() && !hits.length && <p className="mut">No symptom in the database matches that wording. Try a simpler word.</p>}
  </div>;
}

/* ============================== Days ============================== */
function DaysPage({ meta, days, open }) {
  const td = todayISO(), [d, setD] = useState(td), last = days.slice(-7), since = addDays(td, -29), g = {};
  const avg = k => { const v = last.map(x => x.values[k]).filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const ref = k => (meta.factors.find(f => f.k === k) || {}).ref;
  const low = x => meta.factors.filter(f => flagLvl(f, x.values[f.k]) === 2).length;
  const ser = k => days.slice(-14).map(x => x.values[k]).filter(v => v != null);
  [...days].reverse().forEach(x => { const k = fd(x.date, { month: 'long', year: 'numeric' }); (g[k] = g[k] || []).push(x); });
  const sl = avg('sleep_h'), wt = avg('water_l'), s30 = days.filter(x => x.date >= since && Object.keys(x.symptoms).length).length;
  const ST = [
    { t: 'Sleep, 7-day average', v: num(sl) + ' h', sub: `target ${num(ref('sleep_h'))} h`, pct: sl ? sl / ref('sleep_h') * 100 : 0, c: 'var(--acc2)' },
    { t: 'Water, 7-day average', v: num(wt) + ' L', sub: `target ${num(ref('water_l'))} L`, pct: wt ? wt / ref('water_l') * 100 : 0, c: '#38bdf8' },
    { t: 'Days with symptoms', v: s30, sub: 'in the last 30 days', pct: s30 / 30 * 100, c: 'var(--warn)' }
  ];
  const wk = Array.from({ length: 7 }, (_, i) => { const date = addDays(td, i - 6), x = days.find(y => y.date === date); return { date, c: x ? lvlOf(low(x)) : 'none' }; });
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  return <>
    <div className="stats">{ST.map(s => <div className="card st" key={s.t}><Ring pct={s.pct} c={s.c}>{Math.round(s.pct)}%</Ring><div><span>{s.t}</span><b>{s.v}</b><small>{s.sub}</small></div></div>)}</div>
    <div className="duo">
      <div className="card"><div className="ch"><b>This week</b><span className="mut">green: on track · orange: a few low · red: many low</span></div>
        <div className="wk">{wk.map(w => <button key={w.date} className={'wd ' + w.c + (w.date === td ? ' now' : '')} onClick={() => open(w.date)} aria-label={fd(w.date)}><small>{fd(w.date, { weekday: 'short' })}</small><b>{+w.date.slice(8)}</b><i /></button>)}</div>
        <div className="logc" style={{ margin: 0 }}><b>Log a day</b><input type="date" value={d} max={td} onChange={e => setD(e.target.value)} aria-label="Date to log" /><button className="pri" onClick={() => d && open(d)}>Open this day</button></div>
      </div>
      <div className="card"><div className="ch"><b>Sleep and water, last 14 logs</b><span className="mut">dashed line: benchmark</span></div>
        <div className="alab"><span>Sleep</span><b>{num(mean(ser('sleep_h')))} h avg</b></div><Area vals={ser('sleep_h')} line={ref('sleep_h')} c="var(--acc2)" />
        <div className="alab"><span>Water</span><b>{num(mean(ser('water_l')))} L avg</b></div><Area vals={ser('water_l')} line={ref('water_l')} c="#38bdf8" />
      </div>
    </div>
    {!days.length && <p className="mut">No data yet. Log your first day above, or load 90 days of demo data from the Profile page.</p>}
    {days.length > 0 && <div className="leg"><span className="ok">0 to 2 nutrients or factors clearly off</span><span className="warn">3 to 5</span><span className="bad">6 or more</span><span className="sy"><u>!</u> symptoms logged</span></div>}
    {Object.entries(g).map(([m, a]) => <section key={m}><h3>{m}</h3><div className="days">{a.map(x => {
      const lw = low(x), sy = Object.keys(x.symptoms).length;
      return <button key={x.date} className={'db ' + lvlOf(lw)} onClick={() => open(x.date)} aria-label={`${fd(x.date)}: ${lw} clearly off${sy ? ', symptoms logged' : ''}`}>
        <small>{x.date === td ? 'Today' : fd(x.date, { weekday: 'short' })}</small><b>{+x.date.slice(8)}</b><span>{lw} low</span>{sy > 0 && <u title="Symptoms logged">!</u>}</button>;
    })}</div></section>)}
  </>;
}

function DaySheet({ date, meta, days, close, saved }) {
  const ex = days.find(x => x.date === date);
  const [vals, setVals] = useState(() => initEx(ex ? ex.values : {})), [syms, setSyms] = useState(() => ({ ...(ex ? ex.symptoms : {}) }));
  const [note, setNote] = useState(ex ? ex.note : ''), [msg, setMsg] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => { const h = e => e.key === 'Escape' && close(); addEventListener('keydown', h); return () => removeEventListener('keydown', h); }, []);
  const kb = Object.fromEntries(meta.symptoms.map(s => [s.key, s]));
  const save = async () => { setBusy(true); setMsg(''); try { await api('days/' + date, 'PUT', { values: vals, symptoms: syms, note }); await saved(); close(); } catch (e) { setMsg(e.message); setBusy(false); } };
  const del = async () => { if (!confirm('Delete this day?')) return; await api('days/' + date, 'DELETE'); await saved(); close(); };
  const en = meta.exercise_normal, xt = vals.exercise_type === '' || vals.exercise_type == null ? '' : String(vals.exercise_type);   // '' not logged, 0 none, 1 cardio, 2 strength, 3 mixed
  const showA = xt === '1' || xt === '3', showR = xt === '2' || xt === '3', mixed = xt === '3';
  const setEx = patch => setVals(c => withExTotals({ ...c, ...patch }));
  const setType = t => setEx(t === '' ? { exercise_type: '', aerobic_min: '', resist_min: '' }
    : t === '0' ? { exercise_type: 0, aerobic_min: 0, resist_min: 0 }
    : { exercise_type: +t, ...(t === '1' ? { resist_min: '' } : {}), ...(t === '2' ? { aerobic_min: '' } : {}) });
  const exTotal = exNum(vals.exercise_min);
  const timeIn = (key, label) => { const f = meta.factors.find(x => x.k === key); return <label className="fi"><span>{label} <i>min</i></span><input type="number" inputMode="decimal" min="0" step="any" value={vals[key] ?? ''} placeholder={f ? num(f.ref) : ''} onChange={e => setEx({ [key]: e.target.value })} /></label>; };
  const exBlock = <section><h3>Exercise</h3>
    {en && en.ready && <p className="mut">Your normal: about {en.minutes ? num(en.minutes.avg, 0) : 0} minutes, mostly {en.type ? en.type.label : 'no particular type'}; you exercise about {num(en.per_week)} days a week.</p>}
    <div className="fg">
      <label className="fi"><span>Type</span><select value={xt} onChange={e => setType(e.target.value)}><option value="">Not logged</option><option value="0">No exercise</option>{ETYPES.slice(1).map((l, i) => <option key={i} value={i + 1}>{l}</option>)}</select></label>
      {showA && timeIn('aerobic_min', mixed ? 'Time in cardio / aerobic exercise' : 'Time')}
      {showR && timeIn('resist_min', mixed ? 'Time in strength / resistance exercise' : 'Time')}
    </div>
    {(showA || showR) && exTotal != null && <p className="mut">Total exercise time: {num(exTotal)} min</p>}
  </section>;
  const uc = vals.urine_color === '' || vals.urine_color == null ? null : +vals.urine_color, uo = uc ? meta.urine[uc - 1] : null;
  const wf = meta.factors.find(f => f.k === 'weight_kg'), kg = +vals.weight_kg > 0 ? +vals.weight_kg : (wf ? wf.ref : null);   // today's weight, else profile weight
  const needL = uo && kg ? Math.round(kg * uo.avg / 100 * 10) / 10 : null;
  const hydBlock = <section><h3>Hydration</h3>
    <div className="fg"><label className="fi"><span>{wf.l} <i>{wf.u}</i></span>
      <input type="number" inputMode="decimal" min="0" step="any" value={vals.weight_kg ?? ''} placeholder={num(wf.ref)} onChange={e => setVals({ ...vals, weight_kg: e.target.value })} /></label></div>
    <p className="mut">Select the colour that best matches your urine (Armstrong 8-colour chart).</p>
    <div className="urs">{meta.urine.map(u => <button key={u.n} className={'ur' + (uc === u.n ? ' on' : '')} style={{ background: u.h, color: u.n >= 6 ? '#fff' : '#3a2a00' }} aria-pressed={uc === u.n} title={`${u.n}: ${u.l}`}
      onClick={() => setVals(c => ({ ...c, urine_color: uc === u.n ? '' : u.n }))}>{u.n}</button>)}</div>
    {uo && <p><b>{uo.s}</b> ({uo.l.toLowerCase()}). Estimated body mass lost to dehydration: {uo.loss}. {uo.avg ? (needL != null ? `Drink about ${num(needL)} L water over the next few hours${uo.med ? ' and seek medical attention' : ''}.` : 'Enter your body weight above to see how much water you need.') : 'No extra water needed.'}</p>}
  </section>;
  return <div className="modal" onMouseDown={e => e.target === e.currentTarget && close()}><div className="sheet" role="dialog" aria-label="Day log">
    <button className="x" onClick={close}>Close</button>
    <h2>{fd(date, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}</h2>
    <p className="mut">Enter what you ate, drank and did on this day. Leave a box empty if unknown. The grey placeholder is your benchmark.</p>
    {exBlock}{hydBlock}{groupBy(meta.factors.filter(f => !EX_TIME_KEYS.has(f.k) && f.k !== 'weight_kg')).map(([g, fs]) => <section key={g}><h3>{g}</h3><div className="fg">{fs.map(f => {
      const v = vals[f.k];
      return <label className="fi" key={f.k}><span>{f.l} <i>{f.u}</i></span>
        <input type="number" inputMode="decimal" min="0" step="any" value={v ?? ''} placeholder={num(f.ref)} onChange={e => setVals({ ...vals, [f.k]: e.target.value })} /></label>;
    })}</div></section>)}
    <h3>Symptoms</h3>
    <div className="chips">{Object.entries(syms).map(([k, s]) => <span className={'sy' + (kb[k] && kb[k].red ? ' red' : '')} key={k}>{kb[k] ? kb[k].label : k}
      <select value={s} aria-label="Severity" onChange={e => setSyms({ ...syms, [k]: +e.target.value })}>{[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}/5</option>)}</select>
      <button aria-label="Remove symptom" onClick={() => { const c = { ...syms }; delete c[k]; setSyms(c); }}>×</button></span>)}</div>
    {!Object.keys(syms).length && <p className="mut">No symptoms recorded for this day.</p>}
    <SymSearch kb={meta.symptoms} onPick={k => setSyms({ ...syms, [k]: syms[k] || 2 })} />
    <label className="fi"><span>Note</span><input type="text" value={note} onChange={e => setNote(e.target.value)} /></label>
    {msg && <p className="err">{msg}</p>}
    <div className="act"><button className="pri" disabled={busy} onClick={save}>Save day</button>{ex && <button onClick={del}>Delete day</button>}</div>
  </div></div>;
}

/* ============================== Charts + Alerts ============================== */
function Graphs({ meta, days }) {
  const [raw, setRaw, n] = useDays(30),
        end = todayISO(),
        start = addDays(end, -(n - 1)),
        inR = days.filter(d => d.date >= start && d.date <= end);

  const space = {
    space_start: meta.space_start,
    space_age: meta.space_age,
    space_ready_30: meta.space_ready_30,
    space_ready_60: meta.space_ready_60
  };

  return (
    <>
      <div className="bar2">
        <DaysInput raw={raw} setRaw={setRaw} />

        <span className="mut">
          Green band: Earth normal ±1 SD. Blue dotted line: space average, always shown. Blue band and blue curve: space ±1 SD, shown after 30 space days. Dots: red = below the lower band (below both bands in space), orange = below only one of the two bands, green = otherwise. Red dashed: benchmark.
        </span>
      </div>

      {groupBy(meta.factors.filter(f => !NO_CHART.has(f.k))).map(([g, fs]) => (
        <section key={g}>
          <h3>{g}</h3>

          <div className="grid">
            {fs.map(f => {
              const pts = inR
                .filter(d => d.values[f.k] != null)
                .map(d => ({ date: d.date, v: d.values[f.k] }));

              const last = pts[pts.length - 1];

              return (
                <div className="card" key={f.k}>
                  <div className="ch">
                    <b>{f.l}</b>

                    <span className="mut">
                      {REF[f.t]} {num(f.ref)} {f.u}
                      {f.normal != null && ` · active ${num(f.normal)}`}
                      {space.space_start && f.space_avg != null && ` · space ${num(f.space_avg)}`}
                    </span>
                  </div>

                  {pts.length
                    ? <Chart f={f} pts={pts} start={start} n={n} space={space} />
                    : <p className="mut">No data in this range.</p>}

                  {last && (
                    <p className="mut" style={{ fontSize: 12 }}>
                      Latest {num(last.v)} {f.u}: {vsTxt(last.v, f.ref, REF[f.t])}
                      {f.normal != null && ', ' + vsTxt(last.v, f.normal, 'your normal')}.
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </>
  );
}

function AlertsView({ days, open }) {
  const [raw, setRaw, n] = useDays(10), [list, setList] = useState(null), [err, setErr] = useState('');
  useEffect(() => { const t = setTimeout(() => api('alerts?days=' + n).then(setList).catch(e => setErr(e.message)), 150); return () => clearTimeout(t); }, [n, days]);
  return <>
    <div className="bar2"><DaysInput raw={raw} setRaw={setRaw} /><span className="mut">Nutrients are checked against the RDA; sleep, exercise and water against a typical average. Each line also shows how far you are from your own normal. Urine colour gives your hydration level, the body mass it suggests you have lost, and about how much water to drink (from your body weight).</span></div>
    {err && <p className="err">{err}</p>}
    {list && list.map(d => <div key={d.date} className={'card dayc' + (d.status === 'MISSING' ? ' miss' : '')}>
      <header><b>{fd(d.date)}</b><span className={'tag ' + d.status}>{d.status === 'GREEN' ? 'All good' : d.status === 'MISSING' ? 'Reminder' : d.status}</span><button onClick={() => open(d.date)}>{d.status === 'MISSING' ? 'Log this day' : 'Open'}</button></header>
      {d.status === 'GREEN' && !d.items.length ? <p className="mut">Nothing outside your benchmarks or your normal.</p> :
        <ul className="al">{d.items.map((it, i) => <li key={i}><span className={'tag ' + it.level}>{it.level}</span>{it.text}</li>)}</ul>}</div>)}
  </>;
}

function ChartsPage({ meta, days, open }) {
  const [tab, setTab] = useState('graphs');
  return <>
    <div className="seg" role="tablist"><button className={tab === 'graphs' ? 'on' : ''} onClick={() => setTab('graphs')}>Factor graphs</button><button className={tab === 'alerts' ? 'on' : ''} onClick={() => setTab('alerts')}>Alerts</button></div>
    {tab === 'graphs' ? <Graphs meta={meta} days={days} /> : <AlertsView days={days} open={open} />}
  </>;
}

/* ============================== Symptoms (the data-science page) ============================== */
function Analysis({ r }) {
  const [all, setAll] = useState(false), F = r.factors, scored = F.filter(f => f.score != null), md = r.model, top = r.top && F.find(f => f.k === r.top);
  const sp = r.n_days ? r.n_symptom_days / r.n_days * 100 : 0, mw = Math.max(...(r.causes || []).map(c => c.w), 0.01);
  return <>
    {r.notes.length > 0 && <div style={{ marginTop: 12 }}>{r.notes.map((n, i) => <p key={i} className={r.symptom.red ? 'err' : 'mut'}>{n}</p>)}</div>}
    <div className="stats" style={{ marginTop: 12 }}>
      <div className="card st"><Ring pct={sp} c="var(--acc)">{Math.round(sp)}%</Ring><div><span>Days with this symptom</span><b>{r.n_symptom_days} of {r.n_days}</b><small>days analysed</small></div></div>
      {md && <div className="card st"><Ring pct={md.risk_now * 100} c={md.risk_now >= .5 ? 'var(--bad)' : md.risk_now >= .25 ? 'var(--warn)' : 'var(--ok)'}>{Math.round(md.risk_now * 100)}%</Ring><div><span>Estimated risk right now</span><b>{Math.round(md.risk_lo * 100)} to {Math.round(md.risk_hi * 100)}%</b><small>usual rate {Math.round(md.base_rate * 100)}%</small></div></div>}
      <div className="card st"><Ring pct={top ? top.score * 100 : 0} c="var(--acc2)">{top ? Math.round(top.score * 100) : '-'}</Ring><div><span>Strongest link</span><b>{top ? top.l : 'Not enough data'}</b><small>{top ? 'evidence score' : 'keep logging days'}</small></div></div>
    </div>

    <div className="card" style={{ marginTop: 12 }}><h3 style={{ marginTop: 0 }}>Nutrient deficiencies linked to this symptom</h3>
      {r.causes && r.causes.length
        ? <div className="tw"><table className="tb"><thead><tr><th>Symptom</th><th>Nutrient deficiency</th><th>Link strength</th></tr></thead>
          <tbody>{r.causes.map(c => <tr key={c.k}><td>{r.symptom.label}</td><td><b>{c.l}</b></td><td><span className="sbar"><i style={{ width: c.w / mw * 100 + '%' }} /></span></td></tr>)}</tbody></table></div>
        : <p className="mut">No nutrient deficiency is linked to this symptom in the database.</p>}
    </div>

    {scored.length > 0 && <div className="card" style={{ marginTop: 12 }}><h3 style={{ marginTop: 0 }}>All factors ranked by evidence</h3>
      <div className="tw"><table className="tb"><thead><tr><th>Factor</th><th>Right now</th><th>Score</th></tr></thead>
        <tbody>{(all ? scored : scored.slice(0, 10)).map(f => <tr key={f.k}>
          <td><b>{f.l}</b></td>
          <td>{f.now ? `${num(f.now.x)} ${f.u} (${num(Math.abs(f.now.vs_ref), 0)}% ${f.now.vs_ref < 0 ? 'below' : 'above'} ${REF[f.t]})` : '-'}</td>
          <td><span className="sbar"><i style={{ width: Math.min(100, f.score * 100) + '%' }} /></span><b>{num(f.score, 2)}</b></td></tr>)}</tbody></table></div>
      {scored.length > 10 && <button style={{ marginTop: 10 }} onClick={() => setAll(!all)}>{all ? 'Show top 10' : `Show all ${scored.length} factors`}</button>}
    </div>}
  </>;
}

function SymptomsPage({ meta, days }) {
  const [sel, setSel] = useState(null), [win, setWin] = useState(3), [r, setR] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState('');
  useEffect(() => {
    if (!sel) return; let live = true; setBusy(true); setErr('');
    api(`analyze?symptom=${sel}&window=${win}`).then(x => { if (live) setR(x); })
      .catch(e => live && setErr(e.message)).finally(() => live && setBusy(false));
    return () => { live = false; };
  }, [sel, win, days]);          // `days` (not just the count): editing a day must refresh the analysis
  const label = (meta.symptoms.find(s => s.key === sel) || {}).label;
  return <>
    <div className="card"><SymSearch kb={meta.symptoms} onPick={setSel} />
      {sel && <><p>Analysing: <b>{label}</b></p>
        <div className="set">
          <label>Intake window<select value={win} onChange={e => setWin(+e.target.value)}>{[1, 2, 3, 5, 7, 14, 21].map(v => <option key={v} value={v}>average of the last {v} day{v > 1 ? 's' : ''}{v > 7 ? ` (${v / 7} weeks)` : ''}</option>)}</select></label></div></>}
    </div>
    {busy && <p className="mut">Running the Bayesian models…</p>}{err && <p className="err">{err}</p>}
    {!sel && <p className="mut">Pick a symptom or describe it above.</p>}
    {sel && r && !busy && <Analysis r={r} />}
  </>;
}

/* ============================== Health (organ and mental health bars) ============================== */
const barLvl = h => h >= 75 ? 'ok' : h >= 55 ? 'warn' : 'bad';      // h = points earned as a % of the points possible (1 per day)
function OrgansPage({ meta, days }) {
  const [sel, setSel] = useState('brain'), [date, setDate] = useState(''), [d, setD] = useState(null), [err, setErr] = useState('');
  useEffect(() => { let live = true; const qs = date ? `date=${date}` : '';
    api('organs' + (qs ? '?' + qs : '')).then(x => { if (live) { setD(x); setErr(''); } }).catch(e => live && setErr(e.message)); return () => { live = false; }; }, [date, days]);
  if (err) return <p className="err">{err}</p>;
  if (!d) return <p className="mut">Loading…</p>;
  if (!d.organs.length) return <p className="mut">Log a few days first: the health bars start as soon as there is data.</p>;
  const o = d.organs.find(x => x.k === sel) || d.organs[0], t = o.today, fm = Object.fromEntries(meta.factors.map(f => [f.k, f])), pts = v => (v < 0 ? '−' : '') + num(Math.abs(v), 2);
  return <>
    <details className="card howto"><summary><b>How the health bars work</b> (tap to read)</summary>
      <p><b>1. One point a day.</b> Each organ's bar starts full, equal to the number of days between your mission start and planned return (a 30-day stay starts at 30/30). Each day a full 1 point loses nothing; a day worth 0.7 takes 0.3 off the bar. A day with no log, or with too little logged to score it, earns 0 and takes a whole point off.</p>
      <p><b>2. The day's points.</b> <b>Points = Σ (importance × daily variable) − noise − CO₂ − radiation.</b> For example 0.17·x + 0.32·y − 0.17 − 0.02 − 0.9, where x and y are the day's variables for the nutrients that organ needs and the last three numbers are that day's environment penalties. An organ's importance multipliers add up to 1, so the positive part is at most 1. Exercise, sleep and hydration are drivers too (exercise is scaled by how well its type suits the organ).</p>
      <p><b>3. Daily variable.</b> Variable = today's intake ÷ the average of the RDA and your normal, held between 0 and 1 (extra earns nothing). Your normal is the <b>earth average</b>. In space mode the earth average is used for the first 60 days in space, then the <b>space average</b> takes over. In earth mode it is always the earth average. Sodium is a limit: full credit up to it, none at double. With fewer than 5 days of data there is no normal yet, so the RDA alone is used.</p>
      <p><b>4. Negative constants.</b> In space mode, cabin noise, CO₂ and radiation each get a severity from 0 to 1 from their level, multiplied by the organ's cap for that factor; an organ's caps add up to 1, so a day runs from +1 down to −1. Noise: 0–30 dBA costs nothing, 55 dBA and above is annoyance and stress, 85+ risks hearing damage. CO₂: from 1,000 ppm cognitive effects, 3,950 ppm is NASA's one-hour limit, 8,000 ppm the submarine ceiling. Radiation is in mSv per day, with the steps set at the rates that would use up NASA's 1-year and 30-day limits. Mental health takes noise only.</p>
      <p className="mut">The weights, caps and level tables are hand-set conventions from the cited limits, not fitted values: verify them before relying on them. A reading that was not logged costs nothing.</p></details>
    <div className="grid ogrid">{d.organs.map(x => <button key={x.k} className={'oc ' + (x.k === o.k ? 'sel ' : '') + (x.pct == null ? '' : barLvl(x.pct))} onClick={() => setSel(x.k)} aria-pressed={x.k === o.k}>
      <Ring pct={x.pct || 0} size={72} w={9} c="var(--c, var(--acc))">{x.pct == null ? '-' : Math.round(x.pct) + '%'}</Ring>
      <div><b>{x.l}</b><div className="big">{num(x.total, 1)}<small>/ {x.max} days</small></div></div>
      <p className="mut sm">{(() => { const h = [...x.env.filter(e => e.lost > .004).map(e => ({ l: e.l, v: e.lost })), ...x.drag.map(a => ({ l: a.l, v: a.lost }))].sort((a, b) => b.v - a.v).slice(0, 2);
        return h.length ? 'Held back by ' + h.map(a => `${a.l} (−${num(a.v, 1)} pts)`).join(', ') : 'Nothing is holding it back in this period.'; })()}</p></button>)}</div>
    <div className="card" style={{ marginTop: 12 }}>
      <div className="ch"><b>{o.l}: how {t ? fd(t.date) : 'a day'} scored</b><label className="inl">Day<input type="date" value={date || d.date} max={todayISO()} onChange={e => setDate(e.target.value)} /></label></div>
      {!t ? <p className="mut">There is no log for that day: it earns 0 points.</p> : <>
        <p>{t.score == null ? 'Not enough of this organ\'s drivers were logged to score the day: it earns 0 points.' : <>Drivers <b>{num(t.score, 2)}</b>{t.penalties.length > 0 && <> − environment <b>{num(t.penalty, 2)}</b></>} = <b>{pts(t.points)} point{Math.abs(t.points) === 1 ? '' : 's'}</b> out of 1.</>} {t.coverage < 1 && <span className="mut">Drivers that were not logged are left out ({num(t.coverage * 100, 0)}% of the weight was logged).</span>}</p>
        <div className="tw"><table className="tb"><thead><tr><th>Driver</th><th>Importance</th><th>You had</th><th>Benchmark</th><th>Variable</th></tr></thead>
          <tbody>{t.drivers.map(r => { const f = fm[r.k], a = r.a, c = a == null ? '' : a >= .9 ? 'ok' : a >= .6 ? 'warn' : 'bad'; return <tr key={r.k}>
            <td><b>{r.l}</b></td><td>{num(r.w, 2)}</td><td>{r.missing ? <span className="mut">not logged</span> : r.info || `${num(r.x)} ${r.u}`}</td>
            <td>{num(r.ref)} {r.u} <span className="mut">{f ? REF[f.t] : 'target'}</span>{r.normal != null && <><br />{num(r.normal)} {r.u} <span className="mut">your normal ({t.baseline} average)</span></>}</td>
            <td>{a == null ? '-' : <span className={'tag ' + c}>{num(a, 2)}</span>}</td></tr>; })}</tbody></table></div>
        {t.penalties.length > 0 && <div className="tw" style={{ marginTop: 10 }}><table className="tb"><thead><tr><th>Environment</th><th>Reading</th><th>Severity</th></tr></thead>
          <tbody>{t.penalties.map(x => <tr key={x.k}><td><b>{x.l}</b></td><td>{x.missing ? <span className="mut">not logged</span> : `${num(x.x)} ${x.u}`}</td>
            <td><span className={'tag ' + (x.sev === 0 ? 'ok' : x.sev < .5 ? 'warn' : 'bad')}>{num(x.sev * 100, 0)}%</span></td></tr>)}</tbody></table></div>}</>}
    </div>
  </>;
}


/* ============================== Profile ============================== */
function ProfilePage({ meta, reload }) {
  const [p, setP] = useState({ ...meta.profile }), [msg, setMsg] = useState(''), set = (k, v) => setP({ ...p, [k]: v });
  const save = async () => { try { await api('profile', 'PUT', { ...p, age: +p.age, weight_kg: +p.weight_kg, height_cm: +p.height_cm }); await reload(); setMsg('Saved. Benchmarks updated.'); } catch (e) { setMsg(e.message); } };
  const demo = async () => { if (!confirm('Replace all data with 90 days of simulated demo data?')) return; await api('demo', 'POST', { replace: true }); await reload(); setMsg('Demo data loaded.'); };
  const wipe = async () => { if (!confirm('Delete ALL logged days?')) return; await api('data?confirm=yes', 'DELETE'); await reload(); setMsg('All data deleted.'); };
  return <>
    <div className="card"><div className="form">
      <label className="fi">Name<input type="text" value={p.name || ''} onChange={e => set('name', e.target.value)} /></label>
      <label className="fi">Age<input type="number" value={p.age} onChange={e => set('age', e.target.value)} /></label>
      <label className="fi">Sex<select value={p.sex} onChange={e => set('sex', e.target.value)}><option value="male">Male</option><option value="female">Female</option><option value="other">Other</option></select></label>
      <label className="fi">Weight (kg)<input type="number" value={p.weight_kg} onChange={e => set('weight_kg', e.target.value)} /></label>
      <label className="fi">Height (cm)<input type="number" value={p.height_cm} onChange={e => set('height_cm', e.target.value)} /></label>
      <label className="fi">Environment<select value={p.mode} onChange={e => set('mode', e.target.value)}><option value="earth">Earth</option><option value="space">Spaceflight</option></select></label>
      {p.mode === 'space' && <>
        <label className="fi">Mission start<input type="date" value={p.mission_start || ''} onChange={e => set('mission_start', e.target.value)} /></label>
        <label className="fi">Planned return<input type="date" value={p.mission_end || ''} onChange={e => set('mission_end', e.target.value)} /></label>
        <label className="fi">Radiation from earlier missions (mSv)<input type="number" min="0" value={p.career_dose_msv ?? 0} onChange={e => set('career_dose_msv', e.target.value)} /></label>
        <label className="fi">Career dose limit (mSv)<input type="number" min="50" value={p.career_limit_msv ?? 600} onChange={e => set('career_limit_msv', e.target.value)} /></label></>}
    </div>
      <div className="act"><button className="pri" onClick={save}>Save profile</button></div></div>
    <h3>Data</h3><div className="card">
      <div className="act"><button onClick={demo}>Load 90 days of demo data</button><button onClick={wipe}>Delete all data</button></div></div>
    {msg && <p>{msg}</p>}
    <p className="mut" style={{ marginTop: 16 }}>Educational decision-support prototype, not a medical device or medical advice. Reference values are placeholders and must be verified (for example against NASA-STD-3001) and reviewed by a clinician or flight surgeon. Symptom links are statistical associations, not diagnoses.</p>
  </>;
}

/* ============================== Image diagnosis pages (PatchCore models via server.js): Kidney + Eye ============================== */
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const fileToB64 = f => new Promise((ok, no) => {
  const r = new FileReader();
  r.onload = () => ok(String(r.result).split(',')[1] || '');
  r.onerror = () => no(new Error('Could not read ' + f.name));
  r.readAsDataURL(f);
});
const nowParts = () => { const d = new Date(); return { local_date: d.toLocaleDateString('en-CA'), local_time: d.toTimeString().slice(0, 5) }; };

function DiagnosisPage({ cfg }) {
  const now = new Date(), pick = React.useRef(null);
  const [files, setFiles] = useState([]);                       // [{ file, url }]
  const [busy, setBusy] = useState(false), [res, setRes] = useState(null), [err, setErr] = useState(''), [drag, setDrag] = useState(false);
  const [status, setStatus] = useState(null);
  const [year, setYear] = useState(now.getFullYear()), [month, setMonth] = useState(now.getMonth() + 1);   // defaults to the current month
  const [rows, setRows] = useState(null), [listErr, setListErr] = useState('');
  const mk = year + '-' + String(month).padStart(2, '0');
  const maxN = status ? status.max_images : 10, maxMB = status ? status.max_mb : 6;

  const loadRows = () => api(cfg.api + '/tests?month=' + mk).then(r => { setRows(r.tests); setListErr(''); }).catch(e => setListErr(e.message));
  useEffect(() => { setRows(null); loadRows(); }, [mk]);
  useEffect(() => { api(cfg.api + '/status').then(setStatus).catch(() => { }); }, []);

  const add = list => {
    const next = [...files], problems = [];
    for (const f of Array.from(list)) {
      if (!f.type.startsWith('image/')) { problems.push(f.name + ' is not an image'); continue; }
      if (f.size > maxMB * 1048576) { problems.push(f.name + ' is larger than ' + maxMB + ' MB'); continue; }
      if (next.length >= maxN) { problems.push('Only ' + maxN + ' images can be sent at a time'); break; }
      next.push({ file: f, url: URL.createObjectURL(f) });
    }
    setFiles(next); setErr(problems.join('. '));
  };
  const remove = i => { URL.revokeObjectURL(files[i].url); setFiles(files.filter((_, j) => j !== i)); };
  const clear = () => { files.forEach(f => URL.revokeObjectURL(f.url)); setFiles([]); };

  const run = async () => {
    setBusy(true); setErr(''); setRes(null);
    try {
      const images = await Promise.all(files.map(async f => ({ name: f.file.name, data: await fileToB64(f.file) })));
      const r = await api(cfg.api + '/diagnose', 'POST', { images, ...nowParts() });
      setRes(r); clear();
      const [y, m] = r.taken_date.split('-').map(Number);          // show the month that now contains this test
      if (y === year && m === month) loadRows(); else { setYear(y); setMonth(m); }
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  const years = Array.from({ length: 6 }, (_, i) => now.getFullYear() - i); if (!years.includes(year)) years.push(year);
  const likely = res && res.verdict === cfg.yesVerdict, pct = res ? Math.round(res.probability * 100) : 0;

  return <>
    <div className="card">
      {status && !status.model_ready && <p className="err">{cfg.missing}</p>}
      <div className={'dz' + (drag ? ' on' : '')} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}>
        <p><b>{cfg.dzTitle}</b></p>
        <p className="mut sm">Drop images here or choose them (up to {maxN} images, {maxMB} MB each; JPG, PNG, BMP, TIFF, WebP). {cfg.dzHint}</p>
        <input ref={pick} type="file" accept="image/*" multiple hidden onChange={e => { add(e.target.files); e.target.value = ''; }} />
        <button onClick={() => pick.current.click()} disabled={busy}>Choose images</button>
      </div>
      {files.length > 0 && <div className="thumbs">{files.map((f, i) => <div className="th" key={f.url}>
        <img src={f.url} alt={f.file.name} /><button aria-label={'Remove ' + f.file.name} onClick={() => remove(i)} disabled={busy}>×</button><small>{f.file.name}</small></div>)}</div>}
      {err && <p className="err">{err}</p>}
      <div className="act">
        <button className="pri" onClick={run} disabled={busy || !files.length || (status && !status.model_ready)}>{busy ? 'Analysing…' : 'Run diagnosis' + (files.length ? ' (' + files.length + ')' : '')}</button>
        {files.length > 0 && !busy && <button onClick={clear}>Clear</button>}
      </div>
      {busy && <p className="mut">Analysing the images with the trained model. This can take up to a minute.</p>}
    </div>

    {res && <div className={'card verdict kd ' + (likely ? 'bad' : 'ok')} style={{ marginTop: 12 }} role="status">
      <h2>{likely ? cfg.titleYes : cfg.titleNo}</h2>
      <div className="kprob"><Ring pct={pct} size={96} w={11} c={likely ? 'var(--bad)' : res.borderline ? 'var(--warn)' : 'var(--ok)'}>{pct}%</Ring><span>probability score</span></div>
      <div className="gauge" title="The line marks the 50% decision point"><i className={likely ? 'b' : res.borderline ? 'w' : ''} style={{ width: pct + '%' }} /><s style={{ left: '50%' }} /></div>
      <p>{res.message}</p>
      <ul className="al">{res.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      <details className="howto"><summary>Result for each image</summary>
        <div className="tw"><table className="tb kt"><thead><tr><th>Image</th><th>Probability</th><th>Result</th><th>Focus</th></tr></thead>
          <tbody>{res.images.map((x, i) => <tr key={i}><td>{x.image}</td><td>{Math.round(x[cfg.pKey] * 100)}%</td>
            <td><span className={'tag ' + (x.flagged ? 'bad' : 'ok')}>{x.flagged ? 'Abnormal' : 'Normal'}</span></td>
            <td>{x.in_focus ? 'ok' : <span className="tag warn">possibly blurry</span>}</td></tr>)}</tbody></table></div></details>
    </div>}

    <div className="card" style={{ marginTop: 12 }}>
      <div className="kh">
        <h3 style={{ margin: 0 }}>Test history</h3>
        <div className="kf">
          <select aria-label="Month" value={month} onChange={e => setMonth(+e.target.value)}>{MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</select>
          <select aria-label="Year" value={year} onChange={e => setYear(+e.target.value)}>{years.map(y => <option key={y} value={y}>{y}</option>)}</select>
        </div>
      </div>
      {listErr && <p className="err">{listErr}</p>}
      {rows == null && !listErr && <p className="mut">Loading…</p>}
      {rows && !rows.length && <p className="mut">No tests were taken in {MONTHS[month - 1]} {year}.</p>}
      {rows && rows.length > 0 && <div className="tw"><table className="tb kt"><thead><tr><th>Date</th><th>Time</th><th>Images</th><th>Flagged</th><th>Probability</th><th>Result</th></tr></thead>
        <tbody>{rows.map(r => <tr key={r.id}>
          <td>{fd(r.taken_date, { day: 'numeric', month: 'short', year: 'numeric' })}</td><td>{r.taken_time}</td><td>{r.n_images}</td><td>{r.n_flagged} of {r.n_images}</td>
          <td><b>{Math.round(r.probability * 100)}%</b></td>
          <td><span className={'tag ' + (r.verdict === cfg.yesVerdict ? 'bad' : 'ok')}>{r.verdict === cfg.yesVerdict ? cfg.tagYes : cfg.tagNo}</span></td></tr>)}</tbody></table></div>}
    </div>
    <p className="mut sm" style={{ marginTop: 14 }}>Screening aid only, not a medical diagnosis. The probability score comes from an image-based anomaly model and has not been clinically validated. {cfg.confirm}</p>
  </>;
}
const KIDNEY_CFG = {
  api: 'kidney', yesVerdict: 'stones_likely', pKey: 'probability_stone',
  titleYes: 'Kidney stones likely', titleNo: 'No kidney stones detected', tagYes: 'Stones likely', tagNo: 'No stones detected',
  dzTitle: 'Add photos of your urine sample', dzHint: 'Use the same kind of image the model was trained on.',
  missing: 'The trained model was not found on the server, so diagnosis is unavailable. Train it with train_patchcore.py and place patchcore_model.pt in the "models" folder (or set KIDNEY_MODEL).',
  confirm: 'Please confirm any result with a doctor and a laboratory urine test.'
};
const EYE_CFG = {
  api: 'eye', yesVerdict: 'papilledema_likely', pKey: 'probability_abnormal',
  titleYes: 'Swollen optic nerve likely', titleNo: 'No swollen optic nerve detected', tagYes: 'Swelling likely', tagNo: 'No swelling detected',
  dzTitle: 'Add retinal (fundus) photos of your eye', dzHint: 'Use sharp retinal photos centred on the optic disc, like the ones the model was trained on.',
  missing: 'The trained eye model was not found on the server, so diagnosis is unavailable. Train it with train_eye.py and place eye_model.pt in the "models" folder (or set EYE_MODEL).',
  confirm: 'Please confirm any result with an eye doctor (a dilated eye exam or an optic nerve scan such as OCT).'
};
const KidneyPage = () => <DiagnosisPage cfg={KIDNEY_CFG} />;
const EyePage = () => <DiagnosisPage cfg={EYE_CFG} />;
const KIDNEY_CSS = `
.kd h2{margin:0 0 6px}.kd.bad{border-left-color:var(--bad)}
.kprob{display:flex;align-items:center;gap:16px;margin:8px 0}.kprob .ring span{font-size:20px}.kprob>span{color:var(--mut)}
.dz{border:2px dashed color-mix(in srgb,var(--acc) 45%,var(--line));border-radius:22px;padding:26px 20px;text-align:center;background:var(--tint)}.dz.on{border-color:var(--acc)}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px;margin-top:12px}
.th{position:relative}.th img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px;border:1px solid var(--line);display:block}
.th button{position:absolute;top:4px;right:4px;padding:0 8px;border-radius:99px;line-height:22px}
.th small{display:block;font-size:11px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kh{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}.kf{display:flex;gap:8px}
.kt tbody tr{cursor:default}`;
document.head.appendChild(Object.assign(document.createElement('style'), { textContent: KIDNEY_CSS }));

/* ============================== app shell ============================== */
const ico = p => <svg viewBox="0 0 24 24" aria-hidden="true">{p}</svg>;
const PAGES = {
  days: [
    'Daily log',
    'Tap any day to see or edit what you ate, how you slept and which factors were off.',
    ico(<><rect x="3" y="5" width="18" height="16" rx="3" /><path d="M3 10h18M8 3v4M16 3v4" /></>),
    'Days'
  ],

  charts: [
    'Charts',
    'Every factor against its benchmark and your own normal, plus alerts and reminders.',
    ico(<path d="M5 20v-7M12 20V5M19 20v-10" />),
    'Charts'
  ],

  symptoms: [
    'Symptom check',
    'Describe how you feel to see which factor is most likely linked, using your own data.',
    ico(<path d="M3 12h4l2.5-6 4 12 2.5-6H21" />),
    'Symptoms'
  ],

  organs: [
    'Health',
    'Health bars for brain, eyes, bone, muscles, kidneys, heart and (in space mode) mental health: up to 1 point per day from nutrients, exercise, mood and social contact, minus daily penalties for cabin noise, CO2 and radiation.',
    ico(<path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10z" />),
    'Health'
  ],

  kidney: [
    'Kidney diagnosis',
    'Send photos of your urine sample to the trained model. It returns a probability score and a plain-language result, and every run is logged below.',
    ico(<path d="M9 3C5.5 3 3.5 6.5 3.5 11S6 21 10 21c2.2 0 3-1.6 3-3.2 0-1.5-1.2-1.8-1.2-3.3 0-1.4 1.2-1.9 2.2-3 1.1-1.2 1-2.8-.2-4.2C12.6 4.1 11 3 9 3z" />),
    'Kidney'
  ],

  eye: [
    'Eye diagnosis',
    'Send retinal (fundus) photos to the trained model. It checks for a swollen optic nerve (papilledema), which can point to raised pressure inside the skull, and every run is logged below.',
    ico(<><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>),
    'Eye'
  ],

  profile: [
    'Profile',
    'Your details set the RDA benchmarks. Load demo data or clear everything here.',
    ico(<><circle cx="12" cy="8" r="4" /><path d="M4 21c1-4 4-6 8-6s7 2 8 6" /></>),
    'Profile'
  ]
};

// Pages that only exist in spaceflight mode (hidden from the menu on Earth, and a direct link falls back to Days).
const SPACE_ONLY = ['organs', 'kidney', 'eye'];

const pageFromHash = () => {
  const h = location.hash.slice(1);

  // Old Mission links now open Health.
  if (h === 'mission') return 'organs';

  return PAGES[h] ? h : 'days';
};

function App() {
  const [meta, setMeta] = useState(null),
        [days, setDays] = useState([]),
        [err, setErr] = useState(''),
        [sheet, setSheet] = useState(null),
        [page, setPage] = useState(pageFromHash);

  const reload = () => Promise.all([api('meta'), api('days')])
    .then(([m, d]) => {
      setMeta(m);
      setDays(d);
      setErr('');
    })
    .catch(e => setErr(e.message));

  useEffect(() => {
    reload();

    const h = () => {
      setPage(pageFromHash());
      scrollTo(0, 0);
    };

    addEventListener('hashchange', h);

    return () => removeEventListener('hashchange', h);
  }, []);

  if (!meta) {
    return <p style={{ padding: 24 }}>{err ? 'Cannot reach the server: ' + err : 'Loading…'}</p>;
  }

  const open = d => setSheet(d);
  const space = meta.profile.mode === 'space';
  const pg = PAGES[page] && !(SPACE_ONLY.includes(page) && !space) ? page : 'days';

  return (
    <div className="app">
      <nav aria-label="Main">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <b>Crew Health</b>
        </div>

        {Object.entries(PAGES).filter(([k]) => !SPACE_ONLY.includes(k) || space).map(([k, v]) => (
          <button
            key={k}
            className={'nv' + (pg === k ? ' on' : '')}
            aria-current={pg === k ? 'page' : 'false'}
            onClick={() => { location.hash = k; }}
          >
            {v[2]} <span>{v[3]}</span>
          </button>
        ))}

        <p className="mut foot">
          {meta.profile.name} · {space ? 'spaceflight' : 'Earth'} benchmarks
        </p>
      </nav>

      <div>
        <header className="hero">
          <div className="hw">
            <div>
              <h1>{PAGES[pg][0]}</h1>
              <p>{PAGES[pg][1]}</p>
            </div>
            <div className="hp">
              <span className="av">{(meta.profile.name || 'C').trim().charAt(0).toUpperCase()}</span>
              <div><b>{meta.profile.name}</b><small>{space ? 'Spaceflight' : 'Earth'} benchmarks</small></div>
            </div>
          </div>
        </header>

        <main className="pg">
          {err && <p className="err">{err}</p>}

          {pg === 'days' && <DaysPage meta={meta} days={days} open={open} />}
          {pg === 'charts' && <ChartsPage meta={meta} days={days} open={open} />}
          {pg === 'symptoms' && <SymptomsPage meta={meta} days={days} />}
          {pg === 'organs' && <OrgansPage meta={meta} days={days} />}
          {pg === 'kidney' && <KidneyPage />}
          {pg === 'eye' && <EyePage />}
          {pg === 'profile' && <ProfilePage meta={meta} reload={reload} />}
        </main>
      </div>

      {sheet && (
        <DaySheet
          key={sheet}
          date={sheet}
          meta={meta}
          days={days}
          close={() => setSheet(null)}
          saved={reload}
        />
      )}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
