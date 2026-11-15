'use strict';
/* npm test: starts the server on a random port with a throw-away database and checks security, validation, symptom matching,
   the Bayesian statistics (they must find the effects planted in the demo data), the spaceflight layer, and the organ health bars. */
const os = require('os'), path = require('path'), fs = require('fs'), http = require('http');
process.env.DB_FILE = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'crew-health-')), 'test.db');
const S = require('./server.js');
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => { cond ? pass++ : fail++; console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra ? '   ' + extra : '')); };
const add = (iso, n) => { const d = new Date(iso + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); };

(async () => {
  await new Promise(r => S.server.listen(0, '127.0.0.1', r));
  const port = S.server.address().port;
  const raw = (method, p, { body, headers = {} } = {}) => new Promise((res, rej) => {
    const data = body === undefined ? null : typeof body === 'string' ? body : JSON.stringify(body), h = { ...headers };
    if (data !== null && !('Content-Type' in h)) h['Content-Type'] = 'application/json';
    const rq = http.request({ host: '127.0.0.1', port, method, path: p, headers: h }, r => { let b = ''; r.on('data', c => b += c); r.on('end', () => { let j = null; try { j = JSON.parse(b); } catch (e) { } res({ s: r.statusCode, j }); }); });
    rq.on('error', rej); if (data !== null) rq.write(data); rq.end();
  });
  const top = async (sym, qs = '') => { const r = await raw('GET', `/api/analyze?symptom=${sym}${qs}`); return { r: r.j, s: r.s, f: (r.j.factors || []).filter(f => f.score != null) }; };

  console.log('--- security');
  ok('Host header from another domain is refused (DNS rebinding)', (await raw('GET', '/api/meta', { headers: { Host: 'attacker.example' } })).s === 403);
  ok('cross-site Origin is refused', (await raw('POST', '/api/demo', { body: {}, headers: { Origin: 'https://evil.example' } })).s === 403);
  ok('non-JSON POST (a "simple" cross-site request) is refused', (await raw('POST', '/api/demo', { body: '', headers: { 'Content-Type': 'text/plain' } })).s === 415);
  ok('demo on an empty database works', (await raw('POST', '/api/demo', { body: {} })).s === 200);
  ok('demo would replace real data: needs {"replace": true}', (await raw('POST', '/api/demo', { body: {} })).s === 409);
  ok('wiping all data needs ?confirm=yes', (await raw('DELETE', '/api/data')).s === 400);

  console.log('--- validation');
  for (const d of ['2026-13-45', '2026-02-30', 'not-a-date']) ok(`impossible date ${d} rejected`, [400, 404].includes((await raw('PUT', `/api/days/${d}`, { body: { values: { sleep_h: 7 } } })).s));
  ok('future date rejected', (await raw('PUT', '/api/days/2099-01-01', { body: {} })).s === 400);
  for (const [k, v] of [['magnesium_mg', 1e308], ['sleep_h', 200], ['resting_hr', 9999], ['mood', 0]]) ok(`implausible ${k}=${v} rejected`, (await raw('PUT', `/api/days/${add((await raw('GET', '/api/meta')).j.today, -200)}`, { body: { values: { [k]: v } } })).s === 400);
  ok('analyze without a symptom is a 400, not a crash', (await raw('GET', '/api/analyze')).s === 400);
  ok('analyze with an unknown symptom is a 404', (await raw('GET', '/api/analyze?symptom=zzz')).s === 404);
  ok('analysis still works after all the bad input (no poisoned data)', (await raw('GET', '/api/analyze?symptom=headache')).s === 200);

  console.log('--- symptom keyword matching');
  const m1 = S.match('my chest hurts'); ok('"my chest hurts" -> Chest pain', m1[0] && /Chest pain/.test(m1[0].label) && !m1[0].weak && !m1.some(x => !x.weak && /Headache/.test(x.label)));
  const m2 = S.match('i feel dizzy and tired').filter(x => !x.weak).map(x => x.label).join('|'); ok('"i feel dizzy and tired" -> Dizziness and Fatigue', /Dizz/.test(m2) && /Fatigue/.test(m2), m2);
  ok('"legs cramping at night" -> Muscle cramps', /cramps/i.test(S.match('legs cramping at night')[0]?.label));
  ok('typo "headche" -> Headache', /Headache/.test(S.match('headche')[0]?.label));
  ok('"cant sleep" -> poor sleep', /sleep/i.test(S.match('cant sleep')[0]?.label));
  ok('"pain" alone is only a weak suggestion', S.match('pain').every(x => x.weak));
  ok('gibberish matches nothing', S.match('xyzzy plugh').length === 0);
  ok('space wording: "stuffy nose and puffy face" -> fluid shift', /Nasal|puffiness/i.test(S.match('stuffy nose and puffy face')[0]?.label));

  console.log('--- Earth statistics (demo data has planted effects)');
  const meta = (await raw('GET', '/api/meta')).j;
  ok('Earth mode hides space factors', meta.factors.every(f => !f.sp) && meta.factors.length >= 27, meta.factors.length + ' factors');
  const names = f => f.slice(0, 3).map(x => x.l);
  const h = await top('headache'), fa = await top('fatigue'), cr = await top('muscle_cramps'), bo = await top('bone_joint_pain');
  ok('headache: Magnesium ranks first', h.f[0].l === 'Magnesium', names(h.f).join(', '));
  ok('fatigue: Iron in the top two', names(fa.f).slice(0, 2).includes('Iron'), names(fa.f).join(', '));
  ok('muscle cramps: Potassium ranks first', cr.f[0].l === 'Potassium', names(cr.f).join(', '));
  ok('bone pain: Vitamin D in the top two', names(bo.f).slice(0, 2).includes('Vitamin D'), names(bo.f).join(', '));
  ok('benchmark ("clearly off the RDA") odds ratio is reported', h.f.some(f => f.tails && f.tails.ref), '');
  ok('lag defaults to 1 and the overlap design effect is reported', h.r.params.lag === 1 && h.r.params.design_effect >= 1, 'design effect ' + h.r.params.design_effect);
  ok('lag=0 (include the symptom day) is accepted', (await top('headache', '&lag=0')).s === 200);
  ok('mission endpoint is inactive on Earth', (await raw('GET', '/api/mission')).j.active === false);

  console.log('--- spaceflight layer');
  const p = { ...meta.profile, mode: 'space', sex: 'female', age: 38, weight_kg: 62, height_cm: 168 };
  ok('switch to spaceflight mode', (await raw('PUT', '/api/profile', { body: p })).s === 200);
  ok('invalid mission dates rejected', (await raw('PUT', '/api/profile', { body: { ...p, mission_start: '2026-13-01' } })).s === 400);
  ok('return before start rejected', (await raw('PUT', '/api/profile', { body: { ...p, mission_start: '2026-05-01', mission_end: '2026-04-01' } })).s === 400);
  ok('space demo loads', (await raw('POST', '/api/demo', { body: { replace: true } })).s === 200);
  const sm = (await raw('GET', '/api/meta')).j, today = sm.today, sdays = (await raw('GET', '/api/days')).j;
  ok('space factors appear (8) and space symptoms come first', sm.factors.filter(f => f.sp).length === 8 && sm.symptoms[0].space, sm.symptoms.slice(0, 2).map(x => x.label).join(' / '));
  const ms = (await raw('GET', '/api/mission')).j.state;
  ok('mission clock: day, phase, days to return', ms.mission_day > 14 && ms.phase === 'cruise' && ms.days_to_return > 0, `day ${ms.mission_day}, ${ms.phase}, ${ms.days_to_return} days to return`);
  ok('radiation tracked against the career limit', ms.radiation.career_msv > 80 && ms.radiation.pct > 0 && ms.radiation.limit === 600, `${ms.radiation.career_msv} mSv = ${ms.radiation.pct}%`);
  ok('bone and kidney risk levels computed', ['low', 'moderate', 'high'].includes(ms.bone.level) && ['low', 'moderate', 'high'].includes(ms.kidney.level), `bone ${ms.bone.level}, kidney ${ms.kidney.level}`);
  ok('weight drift against pre-flight detected', ms.weight_change_pct < -4, ms.weight_change_pct + '%');
  const al = (await raw('GET', '/api/alerts?days=20')).j, byDate = Object.fromEntries(al.map(a => [a.date, a]));
  const has = (date, lvl, re) => (byDate[date]?.items || []).some(i => i.level === lvl && re.test(i.text));
  ok('persistent vision change escalates to RED (SANS watch)', has(today, 'RED', /Vision change logged on 2/));
  ok('solar particle event day is RED', has(add(today, -11), 'RED', /Solar particle event/));
  ok('cabin CO2 excursion (7400 ppm) is RED', has(add(today, -4), 'RED', /Cabin CO2 7400/));
  ok('headache with elevated CO2 links the two', al.some(a => a.items.some(i => /elevated CO2 is a possible contributor/.test(i.text))));
  const md2 = add(sm.profile.mission_start, 1);
  await raw('PUT', `/api/days/${md2}`, { body: { values: { sleep_h: 6 }, symptoms: { nausea: 4, dizziness: 4 } } });     // severe-ish symptoms on mission day 2
  const early = (await raw('GET', '/api/alerts?days=95')).j.find(a => a.date === md2);
  ok('space motion sickness on mission day 2 is expected (INFO), not a warning', early && early.items.some(i => i.type === 'symptom' && i.level === 'INFO' && /early space adaptation/.test(i.text)) && !early.items.some(i => i.type === 'symptom' && i.level === 'YELLOW'));
  const sa = await top('headache');
  ok('space analysis: window is in-flight days only, with forgetting', sa.r.params.space && sa.r.params.half_life === 45 && sa.r.n_days < sdays.length, `${sa.r.n_days} of ${sdays.length} days`);
  ok('space analysis: Cabin CO2 is a top-3 factor for headache', names(sa.f).includes('Cabin CO2'), names(sa.f).join(', '));
  ok('space analysis: Resistance exercise ranks first for muscle weakness', (await top('muscle_weakness')).f[0].l.startsWith('Resistance'));
  ok('space analysis: Social contact ranks first for loneliness', (await top('loneliness')).f[0].l.startsWith('Social'));
  const vc = sm.factors.find(f => f.k === 'vit_c_mg'), rawMean = sdays.filter(d => d.date > add(sm.profile.mission_start, 14) && d.values.vit_c_mg != null).map(d => d.values.vit_c_mg);
  ok('stored-food vitamin loss is applied in space mode (raw data untouched)', vc.normal < .8 * (rawMean.reduce((a, b) => a + b, 0) / rawMean.length), `normal ${vc.normal.toFixed(1)} mg vs logged ${(rawMean.reduce((a, b) => a + b, 0) / rawMean.length).toFixed(1)} mg`);
  ok('space references differ from Earth (vitamin D 25, exercise 120)', sm.factors.find(f => f.k === 'vit_d_ug').ref === 25 && sm.factors.find(f => f.k === 'exercise_min').ref === 120);

  console.log('--- organs, exercise, ignored nutrients, excess rule (Earth)');
  ok('back to Earth mode with a fresh demo', (await raw('PUT', '/api/profile', { body: { ...p, mode: 'earth' } })).s === 200 && (await raw('POST', '/api/demo', { body: { replace: true } })).s === 200);
  const em = (await raw('GET', '/api/meta')).j, en = em.exercise_normal, R0 = S.refFor(em.profile);
  ok('omega-3 is a tracked nutrient with an adult reference (1.1 g female)', em.factors.find(f => f.k === 'omega3_g')?.ref === 1.1);
  ok('exercise normal: usual intensity is Medium, usual type cardio, with a usual duration', en.ready && en.level.v === 3 && en.type.v === 1 && en.minutes.avg > 20 && en.minutes.avg < 60,
    `${en.level?.label} (${Math.round(en.level?.share * 100)}% of active days), ${en.type?.label}, ${en.minutes?.avg} min, ${en.per_week} days/week`);
  const od = add(em.today, -300);
  ok('exercise level 9 and type 4 are rejected', (await raw('PUT', `/api/days/${od}`, { body: { values: { exercise_level: 9 } } })).s === 400 && (await raw('PUT', `/api/days/${od}`, { body: { values: { exercise_type: 4 } } })).s === 400);

  const og = (await raw('GET', '/api/organs?days=60')).j, byK = k => og.organs.find(o => o.k === k);
  ok('nine organs', og.organs.length === 9 && ['brain', 'eyes', 'lungs', 'bone', 'muscles', 'skin', 'kidney', 'heart', 'liver'].every(k => byK(k)));
  ok('importance weights of every organ sum to 1', og.organs.every(o => Math.abs(o.weights.reduce((a, w) => a + w.w, 0) - 1) < 1e-9));
  ok('every bar and every plotted point stays within 0 to 100', og.organs.every(o => o.bar >= 0 && o.bar <= 100 && o.series.length > 0 && o.series.every(x => x.h >= 0 && x.h <= 100)), og.organs.map(o => `${o.l} ${o.bar}`).join(', '));
  ok('tissues differ in speed: muscles respond faster than bone', byK('muscles').perfect.at50 > byK('bone').perfect.at50, `${byK('muscles').perfect.at50} vs ${byK('bone').perfect.at50} points on a perfect day at 50`);
  let H = 0, Lo = 100; for (let i = 0; i < 3000; i++) { H = S.stepBar(H, 1, 30); Lo = S.stepBar(Lo, 0, 30); }
  ok('the bar can never pass 100 or fall below 0', H <= 100 && H > 99.9 && Lo >= 0 && Lo < .1);
  ok('a perfect day at 50 adds 50/tau; a day scoring below the bar lowers it', Math.abs(S.stepBar(50, 1, 60) - 50 - 50 / 60) < 1e-9 && S.stepBar(70, .4, 60) < 70);
  let Hs = 10; for (let i = 0; i < 2000; i++) Hs = S.stepBar(Hs, .8, 45);
  ok('a steady 80% diet settles at 80 (the bar is a smoothed average of daily scores)', Math.abs(Hs - 80) < .01);

  const toy = { d: { magnesium_mg: .5, zinc_mg: .5 }, fit: [1, 1, 1] }, mk = v => ({ v });
  const full = S.organDay(toy, mk({ magnesium_mg: R0.magnesium_mg, zinc_mg: R0.zinc_mg }), R0), half = S.organDay(toy, mk({ magnesium_mg: R0.magnesium_mg, zinc_mg: 0 }), R0), blank = S.organDay(toy, mk({ magnesium_mg: R0.magnesium_mg }), R0);
  ok('two nutrients, only one taken -> half the points', full.score === 1 && Math.abs(half.score - .5) < 1e-9);
  ok('a nutrient left blank is unknown, not skipped: scored on what was logged, with the coverage shown', blank.score === 1 && blank.coverage === .5);
  const Mg = S.FMAP.magnesium_mg, Na = S.FMAP.sodium_mg;
  ok('below the RDA the credit is proportional; at or above it is full; excess is never penalised', Math.abs(S.adequacy(Mg, .6 * R0.magnesium_mg, R0.magnesium_mg) - .6) < 1e-9 && S.adequacy(Mg, 3 * R0.magnesium_mg, R0.magnesium_mg) === 1);
  ok('sodium is a limit: full credit up to it, none at twice the limit', S.adequacy(Na, 2000, 2300) === 1 && S.adequacy(Na, 4600, 2300) === 0);
  const bone = S.ORGANS.find(o => o.k === 'bone'), heart = S.ORGANS.find(o => o.k === 'heart'), exRow = (o, v) => S.organDay(o, { v }, R0).rows.find(r => r.k === 'exercise').a;
  const ex = (min, lvl, type) => ({ exercise_min: min, exercise_level: lvl, exercise_type: type });
  ok('exercise type matters: strength suits bone, cardio suits the heart (same time and intensity)', exRow(bone, ex(30, 3, 2)) > exRow(bone, ex(30, 3, 1)) && exRow(heart, ex(30, 3, 1)) > exRow(heart, ex(30, 3, 2)));
  ok('exercise time and intensity both count; no exercise counts zero', exRow(bone, ex(20, 5, 2)) > exRow(bone, ex(20, 2, 2)) && exRow(bone, ex(30, 3, 2)) > exRow(bone, ex(10, 3, 2)) && exRow(bone, ex(0, 0, 0)) === 0);

  ok('symptom analysis: excess of an RDA nutrient is never an adverse direction', h.f.filter(f => f.t === 'rda').length > 10 && h.f.filter(f => f.t === 'rda').every(f => f.dir === 'low' && !f.tails.high));
  const ig3 = (await raw('GET', '/api/ignored?window=3')).j, ig7 = (await raw('GET', '/api/ignored?window=7')).j;
  ok('ignored nutrients: each is more than 1 SD under your own average, most ignored first, never sodium', [ig3, ig7].every(g => g.items.every((x, i) => x.z <= -1 && x.k !== 'sodium_mg' && (i === 0 || g.items[i - 1].z <= x.z))), `3 days: ${ig3.items.length}, 7 days: ${ig7.items.length} of ${ig3.checked} checked`);
  const allDays = (await raw('GET', '/api/days')).j, last3 = allDays.slice(-3);
  for (const d of last3) await raw('PUT', `/api/days/${d.date}`, { body: { values: { ...d.values, magnesium_mg: 60 }, symptoms: d.symptoms, note: d.note } });
  const ig3b = (await raw('GET', '/api/ignored?window=3')).j, ig7b = (await raw('GET', '/api/ignored?window=7')).j, m3 = ig3b.items.find(x => x.k === 'magnesium_mg'), m7 = ig7b.items.find(x => x.k === 'magnesium_mg');
  ok('magnesium skipped for 3 days is ignored, with the number of days it was low', m3 && m3.z <= -2 && m3.days_low === 3 && ig3b.items[0].k === 'magnesium_mg', m3 && `z ${m3.z}, ${m3.pct_ref}% of RDA`);
  ok('the intake window changes the list: over a week the same shortfall is diluted but still counted', ig7b.window === 7 && m7 && m7.days_low >= 3 && m7.z > m3.z, m7 && `z ${m7.z}`);
  ok('the symptom analysis returns the ignored list for its own window', (await top('headache', '&window=5')).r.ignored.window === 5);
  const og2 = (await raw('GET', `/api/organs?days=60&date=${last3[2].date}`)).j, brain = og2.organs.find(o => o.k === 'brain'), mgRow = brain.today.drivers.find(r => r.k === 'magnesium_mg');
  ok('brain breakdown for that day: magnesium is a shortfall, biggest losses listed first', mgRow.a < .25 && brain.today.drivers[0].lost >= mgRow.lost && brain.today.drivers.length === 10, `magnesium covered ${Math.round(mgRow.a * 100)}%, score ${Math.round(brain.today.score * 100)}%`);
  ok('the bars list what held them back over the last week', byK('bone').drag.length > 0 && og2.organs.find(o => o.k === 'muscles').drag.some(x => x.k === 'magnesium_mg' || x.k === 'protein_g' || x.k === 'potassium_mg' || x.k === 'exercise' || x.k === 'vit_d_ug'));

  console.log(`\n${pass} passed, ${fail} failed`);
  S.server.close(); S.db.close && S.db.close();
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('TEST CRASH', e); process.exit(1); });
