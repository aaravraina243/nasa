'use strict';
/**
 * Crew Health backend (single file, Node >= 18).      Run:  node server.js [--demo]
 * Storage : ONE SQLite file (crew_health.db). Driver: better-sqlite3 if installed, else Node's built-in node:sqlite (Node >= 22.5).
 * Frontend: App.jsx is served as-is (compiled in the browser), so there is no build step.
 *
 * Statistics (all per person, learned from the person's own daily logs)
 *   - "Normal for you" = mean of that person's own distribution for a factor (kernel density curve is returned for charts).
 *   - Benchmarks     : RDA (nutrients, protein, carbs...) / safe limit (sodium) / typical average (sleep, exercise, water).
 *   - Exposure       : mean of a factor over the last W days (default 3) ending on each day.
 *   - Bayesian odds ratio    : tail (below / above normal, cut-off fixed at 1.25 SD) vs normal band; normal-normal update on log OR.
 *   - Bayesian logistic regr.: Laplace approximation with Gaussian priors from the medical knowledge base:
 *        (a) continuous predictor (actual intake as z-score), (b) joint adjusted model.
 *   - Rules decide alerts; the statistics only explain.
 *   - Exposure window ends the day BEFORE the symptom day by default (lag 1; sleep = last night), so a symptom cannot "cause" its own exposure.
 *   - Overlapping windows are correlated: the data term is down-weighted by a design effect estimated from how clustered the symptom days are.
 *   - Spaceflight mode (profile.mode = 'space') adds a mission clock, adaptation window, space-only factors (radiation, CO2, noise, resistance
 *     exercise, mood, social contact), bone-loss / kidney-stone / radiation / CO2 / SANS rules, stored-food vitamin loss, and
 *     down-weights old days (half-life) because "normal" drifts as the body adapts. Earth behaviour is unchanged.
 * Kidney diagnosis (PatchCore, spaceflight mode only; other modes get HTTP 403): POST /api/kidney/diagnose sends the uploaded urine images to run_patchcore.py (a Python process using the trained
 *   model), stores the probability + verdict in the kidney_tests table, and GET /api/kidney/tests?month=YYYY-MM lists the history.
 *   Env: PYTHON (interpreter), KIDNEY_MODEL (path to patchcore_model.pt, default ./models/patchcore_model.pt), KIDNEY_SCRIPT, KIDNEY_TIMEOUT_MS.
 * Eye diagnosis (same design; swollen optic nerve / papilledema from retinal photos, spaceflight mode only): /api/eye/status, /api/eye/tests, /api/eye/diagnose,
 *   table eye_tests, run_eye.py / eye_service.py. Env: EYE_MODEL (default ./models/eye_model.pt), EYE_SCRIPT, EYE_URL (default http://127.0.0.1:5002).
 * Every reference number is a placeholder from public nutrition knowledge: verify before real use. Not medical advice.
 */
const http = require('http'), fs = require('fs'), path = require('path'), os = require('os'), { spawn } = require('child_process');
// Change '127.0.0.1' to '0.0.0.0'
const PORT = +process.env.PORT || 3000, HOST = process.env.HOST || '0.0.0.0';
const LOOPBACK = ['127.0.0.1', 'localhost', '::1'].includes(HOST);     // when bound to loopback, only loopback Host headers are served (blocks DNS rebinding)
const HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', HOST.toLowerCase(), ...String(process.env.ALLOWED_HOSTS || '').toLowerCase().split(',').filter(Boolean)]);
const DB_FILE = process.env.DB_FILE || path.join(__dirname, 'crew_health.db');

/* ============================== database ============================== */
function openDb(f) {
  try { return new (require('better-sqlite3'))(f); } catch (e) {
    try { const { DatabaseSync } = require('node:sqlite'); return new DatabaseSync(f); }
    catch (e2) { throw new Error('No SQLite driver: run "npm install" (better-sqlite3) or use Node >= 22.5'); }
  }
}

const db = openDb(DB_FILE);

db.exec(`
CREATE TABLE IF NOT EXISTS profile(
  id INTEGER PRIMARY KEY CHECK(id=1),
  name TEXT,
  age INTEGER,
  sex TEXT,
  weight_kg REAL,
  height_cm REAL,
  mode TEXT
);

CREATE TABLE IF NOT EXISTS days(
  date TEXT PRIMARY KEY,
  note TEXT
);

CREATE TABLE IF NOT EXISTS entries(
  date TEXT,
  factor TEXT,
  value REAL,
  PRIMARY KEY(date, factor)
);

CREATE TABLE IF NOT EXISTS symptom_logs(
  date TEXT,
  symptom TEXT,
  severity INTEGER,
  PRIMARY KEY(date, symptom)
);

CREATE TABLE IF NOT EXISTS symptom_kb(
  key TEXT PRIMARY KEY,
  label TEXT,
  keywords TEXT,
  red INTEGER,
  links TEXT
);
`);

{
  const cols = db.prepare('PRAGMA table_info(profile)').all().map(c => c.name);

  [
    ['mission_start', 'TEXT'],
    ['mission_end', 'TEXT'],
    ['career_dose_msv', 'REAL DEFAULT 0'],
    ['career_limit_msv', 'REAL DEFAULT 600']
  ].forEach(([c, t]) => {
    if (!cols.includes(c)) {
      db.exec(`ALTER TABLE profile ADD COLUMN ${c} ${t}`);
    }
  });
}

db.prepare(`
  INSERT OR IGNORE INTO profile(id, name, age, sex, weight_kg, height_cm, mode)
  VALUES (1, 'Crew member', 35, 'male', 75, 178, 'earth')
`).run();

const tx = fn => {
  db.exec('BEGIN');
  try {
    const r = fn();
    db.exec('COMMIT');
    return r;
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  }
};
/* ============================== knowledge base ============================== */
// key, label, unit, group, benchmark type (rda | limit | avg), adverse direction, input step
const FACTORS = [
  ['kcal', 'Calories', 'kcal', 'Energy & macros', 'rda', 'low', 10], ['protein_g', 'Protein', 'g', 'Energy & macros', 'rda', 'low', 1],
  ['carb_g', 'Carbohydrates', 'g', 'Energy & macros', 'rda', 'low', 1], ['fat_g', 'Fat', 'g', 'Energy & macros', 'rda', 'low', 1],
  ['fiber_g', 'Fibre', 'g', 'Energy & macros', 'rda', 'low', .5], ['omega3_g', 'Omega-3 fatty acids', 'g', 'Energy & macros', 'rda', 'low', .1],
  ['vit_a_ug', 'Vitamin A', 'µg', 'Vitamins', 'rda', 'low', 10], ['vit_c_mg', 'Vitamin C', 'mg', 'Vitamins', 'rda', 'low', 1],
  ['vit_d_ug', 'Vitamin D', 'µg', 'Vitamins', 'rda', 'low', .5], ['vit_e_mg', 'Vitamin E', 'mg', 'Vitamins', 'rda', 'low', .5],
  ['vit_k_ug', 'Vitamin K', 'µg', 'Vitamins', 'rda', 'low', 1], ['vit_b1_mg', 'Vitamin B1 (thiamin)', 'mg', 'Vitamins', 'rda', 'low', .1],
  ['vit_b2_mg', 'Vitamin B2 (riboflavin)', 'mg', 'Vitamins', 'rda', 'low', .1], ['vit_b3_mg', 'Vitamin B3 (niacin)', 'mg', 'Vitamins', 'rda', 'low', .5],
  ['vit_b6_mg', 'Vitamin B6', 'mg', 'Vitamins', 'rda', 'low', .1], ['folate_ug', 'Folate (B9)', 'µg', 'Vitamins', 'rda', 'low', 5],
  ['vit_b12_ug', 'Vitamin B12', 'µg', 'Vitamins', 'rda', 'low', .1],
  ['calcium_mg', 'Calcium', 'mg', 'Minerals', 'rda', 'low', 10], ['iron_mg', 'Iron', 'mg', 'Minerals', 'rda', 'low', .5],
  ['magnesium_mg', 'Magnesium', 'mg', 'Minerals', 'rda', 'low', 5], ['zinc_mg', 'Zinc', 'mg', 'Minerals', 'rda', 'low', .5],
  ['potassium_mg', 'Potassium', 'mg', 'Minerals', 'rda', 'low', 20], ['sodium_mg', 'Sodium', 'mg', 'Minerals', 'limit', 'high', 20],
  ['sleep_h', 'Sleep', 'h', 'Lifestyle', 'avg', 'low', .25], ['exercise_min', 'Exercise', 'min', 'Lifestyle', 'avg', 'low', 5],
  ['water_l', 'Water', 'L', 'Lifestyle', 'avg', 'low', .1],
  ['spo2_pct', 'Blood oxygen at rest (SpO2)', '%', 'Vitals', 'avg', 'low', 1],
  ['weight_kg', 'Body weight', 'kg', 'Vitals', 'pre', 'low', .1],
  // spaceflight-only factors (8th value = 1): hidden and ignored in Earth mode
  ['resist_min', 'Resistance exercise (ARED / weights)', 'min', 'Space countermeasures', 'avg', 'low', 5, 1],
  ['aerobic_min', 'Aerobic exercise (treadmill / cycle)', 'min', 'Space countermeasures', 'avg', 'low', 5, 1],
  ['radiation_msv', 'Radiation dose', 'mSv', 'Space environment', 'avg', 'high', .01, 1],
  ['co2_ppm', 'Cabin CO2', 'ppm', 'Space environment', 'limit', 'high', 50, 1],
  ['noise_dba', 'Cabin noise', 'dBA', 'Space environment', 'limit', 'high', 1, 1],
  ['mood', 'Mood (1 low to 5 high)', 'score', 'Mind and crew', 'avg', 'low', .5, 1],
  ['social_min', 'Social contact (crew, family, calls)', 'min', 'Mind and crew', 'avg', 'low', 5, 1],
].map(([k, l, u, g, t, bad, step, sp]) => ({ k, l, u, g, t, bad, step, sp: !!sp }));
const SPECIAL = ['exercise_type', 'urine_color'];
// Urine colour -> dehydration band, with the body-mass loss for that band (NATA position statement, 2000, Table 2).
// avg = midpoint of the loss range, used to estimate water needed: litres = body weight (kg) x avg / 100 (1 kg of water = 1 L).
// Colours 7-8 are "more than 5%" (no upper bound given): 6% is an assumed midpoint (5-7%).
const URINE_BANDS = {
  well: { s: 'Well hydrated', loss: '+1% to -1%', avg: 0, med: false },
  minimal: { s: 'Minimal dehydration', loss: '1% to 3%', avg: 2, med: false },
  significant: { s: 'Significant dehydration', loss: '3% to 5%', avg: 4, med: false },
  serious: { s: 'Serious dehydration', loss: 'more than 5%', avg: 6, med: true } };
const URINE = [['Very pale yellow', 'well', '#fbf8d0'], ['Pale yellow', 'well', '#f7efa0'], ['Light yellow', 'minimal', '#f3e36a'], ['Yellow', 'minimal', '#edd242'],
  ['Dark yellow', 'significant', '#e5be2a'], ['Amber', 'significant', '#d9a31c'], ['Dark amber', 'serious', '#c48516'], ['Brown / orange', 'serious', '#a5640f']]
  .map(([l, band, h], i) => ({ n: i + 1, l, h, ...URINE_BANDS[band], a: [1, 1, 1, .75, .5, .3, .15, 0][i] }));     // a: hydration adequacy score; saved with the day, not analysed as a factor
/** Litres of water to drink over the next few hours for a urine-colour band and body weight (kg); null if weight unknown. */
const waterNeedL = (uo, kg) => kg > 0 ? Math.round(kg * uo.avg / 100 * 10) / 10 : null;
const FKEYS = new Set([...FACTORS.map(f => f.k), ...SPECIAL]);
const vis = p => f => !f.sp || p.mode === 'space';       // which factors this person sees
// plausible ranges: anything outside is a typo, and one absurd value would wreck the statistics
const BOUNDS = { kcal: [0, 12000], protein_g: [0, 500], carb_g: [0, 1500], fat_g: [0, 600], fiber_g: [0, 200], vit_a_ug: [0, 30000], vit_c_mg: [0, 5000], vit_d_ug: [0, 500], vit_e_mg: [0, 1000],
  vit_k_ug: [0, 5000], vit_b1_mg: [0, 100], vit_b2_mg: [0, 100], vit_b3_mg: [0, 500], vit_b6_mg: [0, 500], folate_ug: [0, 5000], vit_b12_ug: [0, 5000], calcium_mg: [0, 6000], iron_mg: [0, 200],
  magnesium_mg: [0, 2000], zinc_mg: [0, 200], potassium_mg: [0, 12000], sodium_mg: [0, 15000], sleep_h: [0, 24], exercise_min: [0, 1440], water_l: [0, 15], spo2_pct: [50, 100],
  weight_kg: [20, 300], resist_min: [0, 600], aerobic_min: [0, 600], radiation_msv: [0, 100], co2_ppm: [0, 50000], noise_dba: [20, 140], mood: [1, 5], social_min: [0, 1440],
  omega3_g: [0, 50], exercise_type: [0, 3], urine_color: [1, 8] };

/** Benchmarks per person. Adult RDA/AI values (rounded); space mode changes a few (more D/Ca/K and protein, less iron). */
function refFor(p) {
  const male = p.sex !== 'female', other = p.sex === 'other', age = p.age, w = p.weight_kg, space = p.mode === 'space';
  const sx = (m, f) => other ? (m + f) / 2 : male ? m : f;
  const bmr = 10 * w + 6.25 * p.height_cm - 5 * age + (other ? -78 : male ? 5 : -161), kcal = Math.round(bmr * (space ? 1.7 : 1.5) / 10) * 10;   // ~2 h/day of exercise in flight
  return {
    kcal, protein_g: Math.round(w * (space ? 1.2 : .8)), carb_g: 130, fat_g: Math.round(kcal * .3 / 9), fiber_g: Math.round(kcal / 1000 * 14), omega3_g: sx(1.6, 1.1),
    vit_a_ug: sx(900, 700), vit_c_mg: sx(90, 75), vit_d_ug: space ? 25 : age > 70 ? 20 : 15, vit_e_mg: 15, vit_k_ug: space ? 80 : sx(120, 90),
    vit_b1_mg: sx(1.2, 1.1), vit_b2_mg: sx(1.3, 1.1), vit_b3_mg: sx(16, 14), vit_b6_mg: age > 50 ? sx(1.7, 1.5) : 1.3, folate_ug: 400, vit_b12_ug: 2.4,
    calcium_mg: space || (age > 50 && !male) || age > 70 ? 1200 : 1000, iron_mg: space ? 10 : sx(8, age > 50 ? 8 : 18),
    magnesium_mg: sx(age > 30 ? 420 : 400, age > 30 ? 320 : 310), zinc_mg: sx(11, 8), potassium_mg: sx(3400, 2600), sodium_mg: 2300,
    sleep_h: 7.5, exercise_min: space ? 120 : 30, water_l: sx(2.7, 2.2), spo2_pct: 97, weight_kg: w,   // weight: your profile (pre-flight) weight
    resist_min: 30, aerobic_min: 30, radiation_msv: .6, co2_ppm: 4000, noise_dba: 60, mood: 3.5, social_min: 45,   // spaceflight placeholders   // "typical average" benchmarks
  };
}
/** 0 = fine, 1 = below benchmark, 2 = clearly off (RDA < 70%, average < 80% / > 115%, limit exceeded) */
function flag(f, v, ref) {
  if (v == null) return 0;
  if (f.t === 'pre') return v < .95 * ref ? 2 : v < .98 * ref ? 1 : 0;      // body mass vs the pre-flight / profile weight
  if (f.t === 'limit') return v > ref ? 2 : 0;
  if (f.bad === 'high') return v > ref * 1.15 ? 2 : v > ref ? 1 : 0;
  return v < (f.t === 'rda' ? .7 : .8) * ref ? 2 : v < ref ? 1 : 0;
}

// key, label, keywords, red flag, links {factor: [adverse direction, prior log-odds-ratio strength]}
const SYMPTOMS = [
  ['fatigue', 'Fatigue / low energy', ['tired', 'tiredness', 'exhausted', 'low energy', 'fatigue', 'sluggish', 'drained', 'weak'], 0, { iron_mg: ['low', .5], sleep_h: ['low', .6], vit_b12_ug: ['low', .4], folate_ug: ['low', .3], magnesium_mg: ['low', .3], vit_d_ug: ['low', .3], kcal: ['low', .3] }],
  ['headache', 'Headache', ['headache', 'head pain', 'migraine', 'throbbing head', 'head hurts'], 0, { water_l: ['low', .6], magnesium_mg: ['low', .5], sleep_h: ['low', .4], vit_b2_mg: ['low', .3] }],
  ['muscle_cramps', 'Muscle cramps', ['cramp', 'cramps', 'leg cramp', 'muscle spasm', 'charley horse', 'twitching'], 0, { magnesium_mg: ['low', .6], potassium_mg: ['low', .5], calcium_mg: ['low', .3], water_l: ['low', .4] }],
  ['dizziness', 'Dizziness / lightheadedness', ['dizzy', 'dizziness', 'lightheaded', 'light headed', 'vertigo', 'spinning'], 0, { water_l: ['low', .5], iron_mg: ['low', .4], vit_b12_ug: ['low', .3], kcal: ['low', .3] }],
  ['nausea', 'Nausea', ['nausea', 'nauseous', 'queasy', 'sick to my stomach', 'vomit', 'feel sick'], 0, { water_l: ['low', .3], vit_b6_mg: ['low', .3], kcal: ['low', .2] }],
  ['brain_fog', 'Brain fog / poor focus', ['brain fog', 'foggy', 'cant focus', 'cannot concentrate', 'concentration', 'forgetful', 'memory'], 0, { sleep_h: ['low', .6], vit_b12_ug: ['low', .4], iron_mg: ['low', .3], water_l: ['low', .3], folate_ug: ['low', .3] }],
  ['insomnia', 'Poor sleep / insomnia', ['insomnia', 'cant sleep', 'cannot sleep', 'sleepless', 'restless', 'wake up at night', 'trouble sleeping'], 0, { magnesium_mg: ['low', .4], exercise_min: ['low', .4], calcium_mg: ['low', .2] }],
  ['low_mood', 'Low mood / anxiety', ['low mood', 'depressed', 'sad', 'anxious', 'anxiety', 'irritable', 'stressed', 'mood'], 0, { sleep_h: ['low', .5], vit_d_ug: ['low', .4], folate_ug: ['low', .3], vit_b12_ug: ['low', .3], magnesium_mg: ['low', .3], exercise_min: ['low', .4] }],
  ['bone_joint_pain', 'Bone / joint pain', ['bone pain', 'joint pain', 'aching joints', 'sore joints', 'achy', 'stiff joints'], 0, { vit_d_ug: ['low', .5], calcium_mg: ['low', .4], magnesium_mg: ['low', .2], exercise_min: ['low', .2] }],
  ['muscle_weakness', 'Muscle weakness / soreness', ['muscle weakness', 'weak muscles', 'sore muscles', 'muscle pain', 'muscle ache'], 0, { protein_g: ['low', .5], vit_d_ug: ['low', .4], magnesium_mg: ['low', .3], potassium_mg: ['low', .3] }],
  ['bruising_bleeding', 'Easy bruising / bleeding gums', ['bruise', 'bruising', 'bleeding gums', 'gum bleeding', 'slow healing', 'nosebleed'], 0, { vit_c_mg: ['low', .6], vit_k_ug: ['low', .5] }],
  ['mouth_sores', 'Mouth sores / cracked lips', ['mouth sores', 'mouth ulcer', 'ulcers', 'cracked lips', 'sore tongue', 'cracked corners'], 0, { vit_b2_mg: ['low', .5], iron_mg: ['low', .3], vit_b12_ug: ['low', .3], folate_ug: ['low', .3] }],
  ['hair_skin', 'Hair loss / dry skin / brittle nails', ['hair loss', 'hair thinning', 'dry skin', 'brittle nails', 'rash', 'skin rash', 'itchy skin'], 0, { zinc_mg: ['low', .5], iron_mg: ['low', .4], protein_g: ['low', .4], vit_a_ug: ['low', .2] }],
  ['vision_eye', 'Blurry vision / dry eyes', ['blurry vision', 'blurred vision', 'dry eyes', 'night vision', 'vision change', 'eye strain'], 0, { vit_a_ug: ['low', .5], vit_b2_mg: ['low', .2] }],
  ['tingling', 'Tingling / numbness', ['tingling', 'numbness', 'pins and needles', 'numb', 'tingle'], 0, { vit_b12_ug: ['low', .6], vit_b6_mg: ['low', .3], folate_ug: ['low', .3], magnesium_mg: ['low', .2] }],
  ['palpitations', 'Palpitations / racing heart', ['palpitations', 'racing heart', 'heart pounding', 'irregular heartbeat', 'heart flutter'], 0, { magnesium_mg: ['low', .4], potassium_mg: ['low', .4], iron_mg: ['low', .3], water_l: ['low', .3], sleep_h: ['low', .3] }],
  ['constipation', 'Constipation / bloating', ['constipation', 'hard stool', 'bloated', 'bloating'], 0, { fiber_g: ['low', .6], water_l: ['low', .5], magnesium_mg: ['low', .3], exercise_min: ['low', .3] }],
  ['frequent_infections', 'Frequent colds / infections', ['frequent colds', 'getting sick', 'infection', 'sore throat', 'immune', 'cold', 'flu'], 0, { zinc_mg: ['low', .5], sleep_h: ['low', .5], vit_d_ug: ['low', .4], vit_c_mg: ['low', .3] }],
  ['back_pain', 'Back pain', ['back pain', 'backache', 'lower back', 'sore back'], 0, { exercise_min: ['low', .3], vit_d_ug: ['low', .3] }],
  ['space_motion_sickness', 'Space motion sickness', ['space sickness', 'space motion sickness', 'motion sickness'], 0, {}],
  ['flank_pain', 'Flank / kidney-area pain', ['flank pain', 'kidney pain', 'kidney stone', 'side pain', 'pain when urinating'], 1, { water_l: ['low', .6], sodium_mg: ['high', .4] }],
  ['chest_pain', 'Chest pain / pressure', ['chest pain', 'chest tightness', 'chest pressure', 'pressure in chest'], 1, {}],
  ['breathlessness', 'Shortness of breath', ['short of breath', 'shortness of breath', 'cant breathe', 'difficulty breathing', 'breathless'], 1, {}],
  ['fainting', 'Fainting / passing out', ['fainted', 'fainting', 'passed out', 'blackout', 'lost consciousness'], 1, {}],
];
// Spaceflight links: these factors exist only in space mode, so Earth analysis is unchanged.
const SPACE_LINKS = {
  headache: { co2_ppm: ['high', .6], noise_dba: ['high', .3] }, fatigue: { co2_ppm: ['high', .3], noise_dba: ['high', .3] },
  brain_fog: { co2_ppm: ['high', .5], noise_dba: ['high', .3] }, insomnia: { noise_dba: ['high', .5], co2_ppm: ['high', .3] },
  low_mood: { social_min: ['low', .6] }, muscle_weakness: { resist_min: ['low', .6], aerobic_min: ['low', .2] }, back_pain: { resist_min: ['low', .5] },
  bone_joint_pain: { resist_min: ['low', .4] }, palpitations: { co2_ppm: ['high', .3] },
};
const OMEGA_LINKS = { brain_fog: ['low', .3], low_mood: ['low', .3], vision_eye: ['low', .3], hair_skin: ['low', .25] };
SYMPTOMS.forEach(s => { if (OMEGA_LINKS[s[0]]) s[4].omega3_g = OMEGA_LINKS[s[0]]; });
SYMPTOMS.forEach(s => Object.assign(s[4], SPACE_LINKS[s[0]] || {}));
SYMPTOMS.push(
  ['facial_congestion', 'Nasal congestion / facial puffiness (fluid shift)', ['nasal congestion', 'stuffy nose', 'blocked nose', 'puffy face', 'facial puffiness', 'head fullness', 'sinus pressure'], 0, {}],
  ['loneliness', 'Loneliness / isolation / homesick', ['lonely', 'loneliness', 'isolated', 'isolation', 'homesick', 'miss home', 'miss my family'], 0, { social_min: ['low', .6] }],
  ['orthostatic', 'Dizzy on standing (orthostatic)', ['dizzy when standing', 'dizzy on standing', 'dizzy standing up', 'orthostatic', 'faint on standing'], 0, { water_l: ['low', .5] }],
  ['eye_flashes', 'Light flashes in the eyes', ['light flashes', 'flashes of light', 'seeing flashes', 'cosmic ray flashes', 'floaters'], 0, {}],
  ['taste_change', 'Food tastes bland / reduced smell', ['food tastes bland', 'no taste', 'taste changes', 'cannot smell', 'reduced smell', 'lost appetite'], 0, {}],
  ['vision_change', 'Persistent or sudden vision change', ['sudden vision loss', 'sudden blurry vision', 'double vision', 'vision suddenly', 'vision getting worse', 'blurry for days'], 1, {}],
  ['radiation_signs', 'Possible acute radiation signs (vomiting, skin redness)', ['radiation sickness', 'radiation exposure', 'skin redness', 'vomiting after solar storm'], 1, {}],
);
const SPACE_SYMS = new Set(['space_motion_sickness', 'facial_congestion', 'loneliness', 'orthostatic', 'eye_flashes', 'taste_change', 'vision_change', 'radiation_signs']);
{ const ins = db.prepare('INSERT OR REPLACE INTO symptom_kb VALUES(?,?,?,?,?)');
  tx(() => SYMPTOMS.forEach(([k, l, kw, red, links]) => ins.run(k, l, JSON.stringify(kw), red, JSON.stringify(links)))); }

/* ============================== math ============================== */
const clamp = (x, a, b) => Math.min(b, Math.max(a, x)), r2 = x => +x.toFixed(2), r3 = x => +x.toFixed(3);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const sdev = (a, m = mean(a)) => a.length > 1 ? Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)) : 0;
const sig = e => 1 / (1 + Math.exp(-e)), dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0), quad = (M, v) => v.reduce((s, x, i) => s + x * dot(M[i], v), 0);
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + .3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x)); }
const phi = z => .5 * (1 + erf(z / Math.SQRT2));
const todayISO = () => new Date().toLocaleDateString('en-CA');
const addDays = (d, n) => { const t = new Date(d + 'T00:00:00Z'); t.setUTCDate(t.getUTCDate() + n); return t.toISOString().slice(0, 10); };
const httpErr = (status, message) => Object.assign(new Error(message), { status });

function inv(A) {                                   // Gauss-Jordan inverse (matrices are small and positive definite)
  const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => +(i === j))]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]; const d = M[c][c]; for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map(r => r.slice(n));
}
/** Bayesian log odds ratio from a 2x2 table (weighted counts allowed): Haldane-Anscombe estimate (variance = sum 1/cell, times `deff`
 *  because consecutive W-day exposure windows overlap and are correlated) combined with a normal prior. */
function postLogOR(a, b, c, d, m0, s0, deff = 1) {
  const A = a + .5, B = b + .5, C = c + .5, D = d + .5, y = Math.log(A * D / (B * C)), vd = (1 / A + 1 / B + 1 / C + 1 / D) * deff, v0 = s0 * s0, prec = 1 / v0 + 1 / vd;
  const m = (m0 / v0 + y / vd) / prec, s = Math.sqrt(1 / prec);
  return { mean: m, sd: s, p: phi(m / s), w: (1 / vd) / prec };        // p = P(OR > 1); w = weight of the data vs the prior
}
/** Bayesian logistic regression: MAP by damped Newton, posterior ~ N(beta_hat, H^-1) (Laplace approximation). Gaussian priors mu/sg.
 *  wl[i] = weight of day i in the likelihood (old days count less in space; 1/W allows for overlapping windows). */
function gradHess(X, y, b, mu, sg, wl) {
  const p = b.length, g = b.map((v, j) => -(v - mu[j]) / sg[j] ** 2), H = Array.from({ length: p }, (_, j) => Array.from({ length: p }, (_, k) => j === k ? 1 / sg[j] ** 2 : 0));
  X.forEach((x, i) => { const pr = sig(dot(x, b)), w = pr * (1 - pr) * wl[i], e = (y[i] - pr) * wl[i];
    for (let j = 0; j < p; j++) { g[j] += e * x[j]; for (let k = 0; k < p; k++) H[j][k] += w * x[j] * x[k]; } });
  return { g, H };
}
function logPost(X, y, b, mu, sg, wl) {
  let s = 0; X.forEach((x, i) => { const e = dot(x, b); s += wl[i] * (y[i] * e - (e > 30 ? e : Math.log1p(Math.exp(e)))); });
  b.forEach((v, j) => s -= .5 * ((v - mu[j]) / sg[j]) ** 2); return s;
}
function blr(X, y, mu, sg, wl = X.map(() => 1)) {
  let b = mu.slice(); b[0] = 0; let cur = logPost(X, y, b, mu, sg, wl);
  for (let it = 0; it < 60; it++) {
    const { g, H } = gradHess(X, y, b, mu, sg, wl), st = inv(H).map(r => dot(r, g)); let t = 1, nb = b, ok = false;
    while (t > 1e-5) { nb = b.map((v, j) => v + t * st[j]); const lp = logPost(X, y, nb, mu, sg, wl); if (lp >= cur - 1e-9) { cur = lp; ok = true; break; } t /= 2; }
    if (!ok) break;                                   // no improving step: keep the current point (the earlier version moved to a worse one)
    const ch = Math.max(...st.map(Math.abs)) * t; b = nb; if (ch < 1e-7) break;
  }
  return { b, cov: inv(gradHess(X, y, b, mu, sg, wl).H) };
}
const CEN = [-2, -1, 0, 1, 2], SB = z => [1, z, ...CEN.map(c => Math.exp(-((z - c) ** 2) / 2))];   // linear term + 5 Gaussian bumps (ridge prior = smoothing penalty)

function quant(s, q) { const i = (s.length - 1) * q, l = Math.floor(i); return s[l] + (s[Math.min(l + 1, s.length - 1)] - s[l]) * (i - l); }
function kde(vs, m, s) {                            // Gaussian kernel density, Silverman bandwidth
  const so = [...vs].sort((a, b) => a - b), sg = Math.min(s, (quant(so, .75) - quant(so, .25)) / 1.34) || s || 1, h = Math.max(.9 * sg * vs.length ** -.2, 1e-6);
  const lo = Math.max(0, m - 3 * s), hi = m + 3 * s, x = [], y = [];
  for (let i = 0; i <= 48; i++) { const v = lo + (hi - lo) * i / 48; let d = 0; for (const a of vs) d += Math.exp(-.5 * ((v - a) / h) ** 2);
    x.push(r3(v)); y.push(+(d / (vs.length * h * Math.sqrt(2 * Math.PI))).toExponential(3)); }
  return { x, y };
}
function stats(f, days) {                           // the person's own distribution for one factor
  const vs = days.map(d => d.v[f.k]).filter(Number.isFinite), n = vs.length;
  if (n < 5) return { n, normal: null, sd: null, kde: null };
  const m = mean(vs), s = Math.max(sdev(vs, m), Math.abs(m) * .03, 1e-6);
  return { n, normal: r3(m), sd: r3(s), kde: kde(vs, m, s) };
}

// One-time clean-up of removed data: keep "no exercise" days as type 0, then drop the old intensity rows and urine output.
db.exec(`
INSERT OR IGNORE INTO entries(date,factor,value) SELECT date,'exercise_type',0 FROM entries WHERE factor='exercise_level' AND value=0;
DELETE FROM entries WHERE factor IN ('exercise_level','aerobic_level','resist_level','urine_l','resting_hr');
`);

/* ============================== data access ============================== */
const getProfile = () => db.prepare('SELECT * FROM profile WHERE id=1').get();
const dayDiff = (a, b) => Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 864e5);
const validDate = s => { if (!/^\d{4}-\d{2}-\d{2}$/.test(String(s))) return false; const t = Date.parse(s + 'T00:00:00Z'); return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === s; };
function load() {
  const days = new Map();
  for (const r of db.prepare('SELECT date,note FROM days ORDER BY date').all()) days.set(r.date, { date: r.date, note: r.note || '', v: {}, s: {} });
  for (const r of db.prepare('SELECT date,factor,value FROM entries').all()) { const d = days.get(r.date); if (d) d.v[r.factor] = r.value; }
  for (const r of db.prepare('SELECT date,symptom,severity FROM symptom_logs').all()) { const d = days.get(r.date); if (d) d.s[r.symptom] = r.severity; }
  return [...days.values()];
}
function meta() {
  const p = getProfile(),
        days = loadEff(p),
        R = refFor(p),
        nd = normDays(days, p),
        space = p.mode === 'space';

  const split = splitSpace(days, p);

  const syms = db.prepare('SELECT key,label,red FROM symptom_kb')
    .all()
    .map(r => ({
      key: r.key,
      label: r.label,
      red: !!r.red,
      space: SPACE_SYMS.has(r.key)
    }));

  if (space) syms.sort((a, b) => b.space - a.space);

  return {
    profile: p,
    today: todayISO(),
    count: days.length,
    latest: days.length ? days[days.length - 1].date : null,

    space_start: split.start,
    space_age: split.age,
    space_ready_30: split.ready30,
    space_ready_60: split.ready60,

    factors: FACTORS.filter(vis(p)).map(f => {
      const active = stats(f, nd);
      const earth = split.start ? stats(f, split.earth) : null;
      const sp = split.start ? stats(f, split.space) : null;
      const spv = split.start ? split.space.map(d => d.v[f.k]).filter(Number.isFinite) : [];

      return {
        ...f,
        ref: R[f.k],

        // Active baseline used for scoring/rules.
        ...active,

        // Explicit earth/space chart datasets.
        active,
        earth,
        space: sp,

        earth_normal: earth && earth.normal != null ? earth.normal : null,
        space_normal: sp && sp.normal != null ? sp.normal : null,
        space_avg: spv.length ? r3(mean(spv)) : null
      };
    }),

    symptoms: syms,
    urine: URINE,
    exercise_normal: exerciseNormal(nd)
  };
}
/** Profile weight follows the most recent logged body weight (so logging weight on a day also updates the Profile page). */
function syncProfileWeight() {
  const r = db.prepare("SELECT value FROM entries WHERE factor='weight_kg' ORDER BY date DESC LIMIT 1").get();
  if (r && Number.isFinite(r.value) && r.value >= 30 && r.value <= 250) db.prepare('UPDATE profile SET weight_kg=? WHERE id=1').run(r.value);   // same 30-250 kg limits as the profile form
}
function saveDay(date, b) {
  const kb = new Set(db.prepare('SELECT key FROM symptom_kb').all().map(r => r.key));
  tx(() => {
    db.prepare('INSERT INTO days(date,note) VALUES(?,?) ON CONFLICT(date) DO UPDATE SET note=excluded.note').run(date, String(b.note || '').slice(0, 500));
    db.prepare('DELETE FROM entries WHERE date=?').run(date); db.prepare('DELETE FROM symptom_logs WHERE date=?').run(date);
    const ei = db.prepare('INSERT INTO entries VALUES(?,?,?)'), si = db.prepare('INSERT INTO symptom_logs VALUES(?,?,?)');
    for (const [k, v] of Object.entries(b.values || {})) {
      if (!FKEYS.has(k) || v === '' || v == null) continue;
      const x = +v, [lo, hi] = BOUNDS[k] || [0, 1e5]; if (!Number.isFinite(x) || x < lo || x > hi) throw httpErr(400, `${k}: ${v} is outside the plausible range ${lo} to ${hi}`); ei.run(date, k, SPECIAL.includes(k) ? Math.round(x) : x);
    }
    for (const [k, v] of Object.entries(b.symptoms || {})) { const s = Math.round(+v); if (kb.has(k) && s >= 1 && s <= 5) si.run(date, k, s); }
    syncProfileWeight();
  });
}

/* ============================== symptom keyword matching (from the symptom_kb table) ============================== */
const norm = s => String(s).toLowerCase().replace(/'/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
function lev(a, b) { const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]); for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)); return d[a.length][b.length]; }
const STOPW = new Set('i my me a an the and or of in on at to is am are was be been have has had feel feels feeling felt very bit some with get getting often really lot too so it its that this when after before all day days been'.split(' '));
const SYN = { hurts: 'pain', hurt: 'pain', hurting: 'pain', aching: 'pain', ache: 'pain', aches: 'pain', painful: 'pain', cramping: 'cramp', cramps: 'cramp', tiredness: 'tired', exhaustion: 'exhausted',
  dizzy: 'dizziness', nauseous: 'nausea', queasy: 'nausea', vomiting: 'vomit', headaches: 'headache', tingly: 'tingling', stressed: 'stress', anxious: 'anxiety' };
const toks = s => norm(s).split(' ').filter(t => t && !STOPW.has(t)).map(t => SYN[t] || t);
const stem = t => { for (; ;) { const n = t.replace(/(ness|ing|ed|es|s)$/, ''); if (n === t || n.length < 3) return t; t = n; } };   // tiredness -> tir, weakness -> weak
const sameTok = (x, t) => { const a = stem(x), b = stem(t); return a === b || (a.length >= 4 && b.length >= 4 && lev(a, b) <= 1); };   // typo tolerance; no prefix rule ("nose" is not "nosebleeds")
/** Score = share of a keyword phrase's information (IDF weight) found in the text, so a generic word like "pain" counts little and "chest" counts a lot.
 *  >= 2/3 is a match; 1/4 to 2/3 is returned as a weak "did you mean" suggestion. */
function match(q) {
  const qt = toks(q); if (!qt.length) return [];
  const rows = db.prepare('SELECT * FROM symptom_kb').all().map(r => ({ r, phrases: [...JSON.parse(r.keywords), r.label].map(toks).filter(t => t.length) }));
  const df = {}; rows.forEach(({ phrases }) => new Set(phrases.flat()).forEach(t => df[t] = (df[t] || 0) + 1));
  const idf = t => Math.log(1 + rows.length / (df[t] || 1)), out = [];
  for (const { r, phrases } of rows) {
    let best = 0;
    for (const pt of phrases) { const tot = pt.reduce((s, t) => s + idf(t), 0), got = pt.filter(t => qt.some(x => sameTok(x, t))).reduce((s, t) => s + idf(t), 0); best = Math.max(best, got / tot); }
    if (best >= .25) out.push({ key: r.key, label: r.label, red: !!r.red, score: r2(best), weak: best < .66 });
  }
  return out.sort((a, b) => b.score - a.score).slice(0, 8);
}

/* ============================== symptom analysis (Bayesian) ============================== */
const SD_CUT = 1.25;        // below / above normal = more than 1.25 SD from your own normal (fixed)
const NUTRIENT_GROUPS = ['Energy & macros', 'Vitamins', 'Minerals'];
function analyze(q) {
  if (!q.symptom) throw httpErr(400, 'symptom is required');

  const sym = db.prepare('SELECT * FROM symptom_kb WHERE key=?').get(q.symptom);
  if (!sym) throw httpErr(404, 'Unknown symptom');

  const links = JSON.parse(sym.links),
        p = getProfile(),
        R = refFor(p),
        space = p.mode === 'space';

  const allDays = loadEff(p);
  const split = splitSpace(allDays, p);

  const days = analysisDays(allDays, p),
        N = days.length;

  const W = clamp(Math.round(+q.window || 3), 1, 21),
        C = SD_CUT,   // fixed at 1.25 SD (no longer user-selectable)
        MS = clamp(Math.round(+q.minSev || 1), 1, 5),
        LG = q.lag === undefined ? 1 : clamp(Math.round(+q.lag) || 0, 0, 3);

  const by = new Map(days.map(d => [d.date, d]));
  const byAll = new Map(allDays.map(d => [d.date, d]));

  const asOf = allDays.length ? allDays[allDays.length - 1].date : null;

  // No half-life: all days are weighted equally.
  const wt = days.map(() => 1);

  const expo = (k, date, lag = k === 'sleep_h' ? 0 : LG) => {
    const v = [];

    for (let i = 0; i < W; i++) {
      const d = byAll.get(addDays(date, -(i + lag)));
      if (d && Number.isFinite(d.v[k])) v.push(d.v[k]);
    }

    return v.length >= Math.ceil(W / 2) ? mean(v) : null;
  };

  const y = days.map(d => (d.s[q.symptom] || 0) >= MS ? 1 : 0),
        nSym = y.reduce((a, b) => a + b, 0),
        enough = N >= 14 && nSym >= 3 && N - nSym >= 3;

  const flagSets = {};
  let model = null;

  const ym = nSym / Math.max(N, 1),
        rhoY = k => {
          let n = 0, sxy = 0;

          days.forEach((d, i) => {
            const o = by.get(addDays(d.date, k));
            if (o) {
              sxy += (y[i] - ym) * ((o.s[q.symptom] || 0) >= MS ? 1 : 0) - (y[i] - ym) * ym;
              n++;
            }
          });

          const v = ym * (1 - ym) * n;
          return v > 0 ? Math.max(0, sxy / v) : 0;
        };

  const DEFF = 1 + 2 * Array.from({ length: W - 1 }, (_, i) => i + 1)
    .reduce((a, k) => a + (W - k) / W * rhoY(k), 0);

  const rows = FACTORS.filter(vis(p)).map(f => {
    const L = links[f.k],
          xs = days.map(d => expo(f.k, d.date)),
          idx = xs.map((v, i) => v == null ? -1 : i).filter(i => i >= 0),
          ref = R[f.k];

    const r = {
      k: f.k,
      l: f.l,
      u: f.u,
      g: f.g,
      t: f.t,
      bad: f.bad,
      ref,
      link: L ? { dir: L[0], w: L[1] } : null,
      n: idx.length
    };

    if (idx.length < 10) return r;

    const vals = idx.map(i => xs[i]),
          mx = mean(vals),
          sx = Math.max(sdev(vals, mx), Math.abs(mx) * .03, 1e-6);

    r.normal = r3(mx);
    r.sd = r3(sx);
    r.low_cut = r3(mx - C * sx);
    r.high_cut = r3(mx + C * sx);

    const cur = asOf ? expo(f.k, asOf, 0) : null;

    if (cur != null) {
      const z = (cur - mx) / sx;

      r.now = {
        x: r3(cur),
        z: r2(z),
        vs_ref: +((cur - ref) / ref * 100).toFixed(1),
        vs_normal: +((cur - mx) / mx * 100).toFixed(1),
        band: z <= -C ? 'low' : z >= C ? 'high' : 'normal',
        below_ref: f.bad === 'high' ? cur > ref : cur < ref
      };
    }

    if (!enough) return r;

    const zs = idx.map(i => (xs[i] - mx) / sx),
          yy = idx.map(i => y[i]),
          wI = idx.map(i => wt[i]),
          wL = wI.map(w => w / DEFF);

    const cnt = (isIn, isOut) => {
      let a = 0, b = 0, c = 0, d = 0;

      zs.forEach((z, j) => {
        const w = wI[j];

        if (isIn(z, j)) yy[j] ? a += w : b += w;
        else if (isOut(z, j)) yy[j] ? c += w : d += w;
      });

      return [a, b, c, d];
    };

    const mk = ([a, b, c, d], hyp, pr) => {
      if (a + b < 3 || c + d < 3) return null;

      const P = postLogOR(a, b, c, d, hyp ? pr : 0, hyp ? .5 : .7, DEFF);

      return {
        a: r2(a),
        b: r2(b),
        c: r2(c),
        d: r2(d),
        or: r2(Math.exp(P.mean)),
        lo: r2(Math.exp(P.mean - 1.96 * P.sd)),
        hi: r2(Math.exp(P.mean + 1.96 * P.sd)),
        p: r3(P.p),
        w: r2(P.w)
      };
    };

    const tail = dir => mk(
      cnt(
        z => dir === 'low' ? z <= -C : z >= C,
        z => Math.abs(z) < C
      ),
      L && L[0] === dir,
      L && L[1]
    );

    const offRef = j => flag(f, xs[idx[j]], ref) === 2;

    r.tails = {
      low: tail('low'),
      high: f.t === 'rda' ? null : tail('high'),
      ref: mk(
        cnt(
          (z, j) => offRef(j),
          (z, j) => !offRef(j)
        ),
        L && L[0] === f.bad,
        L && L[1]
      )
    };

    const hd = L ? L[0] : f.bad,
          od = hd === 'low' ? 'high' : 'low',
          T = r.tails;

    r.dir = T[od] && T[od].p >= .95 && T[od].or >= 1.5 && !(T[hd] && T[hd].p >= T[od].p)
      ? od
      : hd;

    const bt = T[r.dir];

    if (bt) {
      r.evidence = r.dir !== hd
        ? 'unexpected direction (data-driven)'
        : bt.w < .35
          ? 'hypothesis (mostly prior knowledge)'
          : bt.p >= .975 && nSym >= 5
            ? 'likely (supported by your data)'
            : bt.p >= .8
              ? 'possible'
              : 'weak';
    }

    flagSets[f.k] = new Set(
      idx.filter((i, j) => r.dir === 'low' ? zs[j] <= -C : zs[j] >= C)
    );

    const pm = L ? (L[0] === 'low' ? -L[1] : L[1]) : 0,
          ps = L ? .5 : .7;

    const fit = blr(zs.map(z => [1, z]), yy, [0, pm], [3, ps], wL),
          b1 = fit.b[1],
          s1 = Math.sqrt(fit.cov[1][1]);

    r.cont = {
      beta: r3(b1),
      sd: r3(s1),
      or_low: r2(Math.exp(-b1)),
      lo: r2(Math.exp(-b1 - 1.96 * s1)),
      hi: r2(Math.exp(-b1 + 1.96 * s1)),
      p_low: r3(phi(-b1 / s1))
    };

    return r;
  });

  const jf = rows.filter(r => r.cont && r.n >= .6 * N);

  if (enough && jf.length > 1) {
    const Z = days.map(d => [
      1,
      ...jf.map(r => {
        const x = expo(r.k, d.date);
        return x == null ? 0 : (x - r.normal) / r.sd;
      })
    ]);

    const jm = blr(
      Z,
      y,
      [0, ...jf.map(r => r.link ? (r.link.dir === 'low' ? -r.link.w : r.link.w) : 0)],
      [3, ...jf.map(() => .5)],
      wt.map(w => w / DEFF)
    );

    jf.forEach((r, j) => {
      const b = jm.b[j + 1],
            s = Math.sqrt(jm.cov[j + 1][j + 1]);

      r.adj = {
        beta: r3(b),
        or_low: r2(Math.exp(-b)),
        lo: r2(Math.exp(-b - 1.96 * s)),
        hi: r2(Math.exp(-b + 1.96 * s)),
        p_low: r3(phi(-b / s))
      };
    });

    if (asOf) {
      const zn = [1, ...jf.map(r => r.now ? r.now.z : 0)],
            e = dot(zn, jm.b),
            s = Math.sqrt(quad(jm.cov, zn));

      model = {
        base_rate: r3(nSym / N),
        risk_now: r3(sig(e)),
        risk_lo: r3(sig(e - 1.96 * s)),
        risk_hi: r3(sig(e + 1.96 * s)),
        n_predictors: jf.length
      };
    }
  }

  rows.forEach(r => {
    if (r.cont) {
      const lowSide = r.dir === 'low',
            ps = [
              r.tails[r.dir] ? r.tails[r.dir].p : null,
              lowSide ? r.cont.p_low : 1 - r.cont.p_low,
              r.adj ? (lowSide ? r.adj.p_low : 1 - r.adj.p_low) : null
            ].filter(v => v != null);

      r.score = r3(mean(ps));

      if (!r.link) r.score = r3(.5 + (r.score - .5) * .5);
    } else if (!enough && r.link && r.now) {
      r.dir = r.link.dir;
      r.evidence = 'hypothesis (medical-knowledge prior only)';
      r.score = r.now.band === r.dir || r.now.below_ref ? r2(.5 + r.link.w / 2) : .3;
    }
  });

  const jac = (A, B) => {
    let i = 0;
    A.forEach(x => B.has(x) && i++);
    return i / (A.size + B.size - i || 1);
  };

  const top6 = rows.filter(r => r.score != null)
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);

  top6.forEach(r => {
    if (flagSets[r.k]) {
      r.confounded_with = top6
        .filter(o => o !== r && flagSets[o.k] && jac(flagSets[r.k], flagSets[o.k]) >= .8)
        .map(o => o.k);
    }
  });

  rows.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));

  const notes = [];

  if (!enough) {
    notes.push('Not enough data for the statistics yet: log at least 14 days, with this symptom on 3 or more of them and 3 or more symptom-free days. Showing medical-knowledge hypotheses compared with your current intake.');
  }

  if (sym.red) {
    notes.push('This is a red-flag symptom. Statistics never replace medical care: seek medical help promptly.');
  }

  if (space && split.start) {
    notes.push(
      split.ready60
        ? 'Space baseline is being used because more than 60 space days have passed.'
        : 'Earth baseline is used for the first 60 space days. Blue space curves appear after 30 space days when enough space data exists.'
    );
  }

  const top = enough ? rows.find(r => r.score >= .8) : null;

  // Table 1: nutrient deficiencies that medical knowledge links to this symptom (symptom_kb.links = {nutrient: [direction, weight]}).
  const causes = FACTORS.filter(f => NUTRIENT_GROUPS.includes(f.g) && links[f.k] && links[f.k][0] === 'low')
    .map(f => ({ k: f.k, l: f.l, u: f.u, w: links[f.k][1] }))
    .sort((a, b) => b.w - a.w);

  return {
    symptom: {
      key: sym.key,
      label: sym.label,
      red: !!sym.red
    },

    params: {
      window: W,
      cutoff: C,
      min_sev: MS,
      lag: LG,
      space,
      design_effect: r2(DEFF)
    },

    as_of: asOf,
    n_days: N,
    n_symptom_days: nSym,
    enough,
    notes,
    model,
    top: top ? top.k : null,
    causes,
    factors: rows,

    space_start: split.start,
    space_age: split.age,
    space_ready_30: split.ready30,
    space_ready_60: split.ready60
  };
}
/* ============================== spaceflight layer (modified) ==============================
Mission page is removed. Mission dates remain in profile because they are used as the
space-start reference.

Space transition rule:
- Space days start from mission_start.
- Earth baseline is used until SPACE_SHIFT_DAYS space days have passed.
- After that, space baseline is used.
- Blue space overlays are exposed after SPACE_OVERLAY_DAYS space days.
- No half-life weighting: all days have equal weight.
============================================================================================ */

const SPACE_SHIFT_DAYS = 60;
const SPACE_OVERLAY_DAYS = 30;

const SPACE_ONLY_KEYS = new Set([
'resist_min',
'aerobic_min',
'radiation_msv',
'co2_ppm',
'noise_dba',
'mood',
'social_min'
]);


/** Days as analysed: logged values are used unchanged.
Stored-food vitamin loss has been removed because food is treated as nutritionally constant. */
function loadEff(_p = getProfile()) {
  return load();
}

/** Hidden space-start date.
    Mission UI is removed, but mission_start remains in profile and is used internally. */
function spaceStart(p, days) {
  if (p.mode !== 'space') return null;

  if (p.mission_start) return p.mission_start;

  // Fallback only if mission_start is missing:
  // use first logged space-only factor as space start.
  const first = days.find(d =>
    Object.entries(d.v).some(([k, v]) => SPACE_ONLY_KEYS.has(k) && Number.isFinite(v))
  );

  return first ? first.date : null;
}

function splitSpace(days, p) {
  const start = spaceStart(p, days);
  const latest = days.length ? days[days.length - 1].date : null;

  if (!start || !latest) {
    return {
      start: null,
      age: 0,
      earth: days,
      space: [],
      ready30: false,
      ready60: false
    };
  }

  const earth = days.filter(d => d.date < start);
  const space = days.filter(d => d.date >= start);

  const age = latest >= start ? dayDiff(start, latest) + 1 : 0;

  return {
    start,
    age,
    earth,
    space,
    ready30: age > SPACE_OVERLAY_DAYS,
    ready60: age > SPACE_SHIFT_DAYS
  };
}

/** "Normal for you".
    Earth baseline is used during the first 60 space days.
    After 60 space days, space baseline is used. */
function normDays(days, p) {
  if (p.mode !== 'space') return days;

  const s = splitSpace(days, p);
  if (!s.start) return days;

  if (s.ready60 && s.space.length >= 10) return s.space;

  return s.earth.length ? s.earth : s.space;
}

/** Symptom analysis uses the same active baseline as normDays. */
function analysisDays(days, p) {
  return normDays(days, p);
}

const medianOf = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
/* ============================== exercise, organ grading ============================== */
// Exercise is logged as type (0 none, 1 cardio, 2 strength, 3 mixed) and minutes. Every minute counts the same (no intensity multiplier).
const ETYPES = ['none', 'cardio', 'strength', 'mixed'];
function exerciseOf(d) {
  const v = d.v, hasRM = Number.isFinite(v.resist_min) || Number.isFinite(v.aerobic_min);
  let min = Number.isFinite(v.exercise_min) ? v.exercise_min : hasRM ? (v.resist_min || 0) + (v.aerobic_min || 0) : null;
  if (min == null && v.exercise_type === 0) min = 0;
  if (min == null) return { min: null };
  let type = v.exercise_type > 0 ? v.exercise_type : hasRM ? (v.resist_min > 0 && v.aerobic_min > 0 ? 3 : v.resist_min > 0 ? 2 : v.aerobic_min > 0 ? 1 : 0) : 0;
  if (v.exercise_type === 0 || min === 0) type = 0;      // logged as "no exercise"
  else if (!type) type = 3;                              // minutes without a type: assume mixed
  return { min, type };
}
/** "Normal exercise for you": the most common type on days you exercised, and your usual duration. */
function exerciseNormal(days) {
  const logged = days.map(d => exerciseOf(d)).filter(e => e.min != null), n = logged.length;
  if (n < 5) return { ready: false, days: n };
  const act = logged.filter(e => e.type > 0);
  const mode = a => { const c = {}; a.forEach(x => c[x] = (c[x] || 0) + 1); const b = Object.entries(c).sort((x, y) => y[1] - x[1] || x[0] - y[0])[0]; return b ? { v: +b[0], share: r2(b[1] / a.length) } : null; };
  const ty = mode(act.map(e => e.type)), mins = act.map(e => e.min);
  return { ready: true, days: n, active_share: r2(act.length / n), per_week: r2(act.length / n * 7),
    type: ty && { ...ty, label: ETYPES[ty.v] },
    minutes: mins.length ? { avg: r2(mean(mins)), sd: r2(sdev(mins)), median: r2(medianOf(mins)) } : null };
}

/* ---- organ health bars: daily points ----
   Each calendar day of the period earns at most 1 point per organ, so a stay of x days gives a bar of at most x (30 days -> 30, 60 days -> 60).
   Day points  =  sum(importance_i * v_i)  -  (noise penalty + CO2 penalty + radiation penalty)
     e.g. 0.17*x + 0.32*y - 0.17 - 0.02 - 0.9, where x and y are the day's variables and the three constants are today's environment penalties.
   Importance weights of an organ add up to 1, so the positive part is 0..1.
   Daily variable v = today's intake / average(RDA, your normal), held between 0 and 1 (excess earns nothing extra).
     Your normal = EARTH average; in space mode the earth average is used for the first 60 space days, then the SPACE average takes over (>= 10 space days needed, else earth stays).
     With no normal yet (fewer than 5 days) the RDA alone is used. Limits such as sodium: 1 up to the limit, falling to 0 at twice the limit. Hydration uses its urine-colour score.
   Environment penalties (space mode only): each of noise, CO2 and radiation has a severity 0..1 from its level (tables below), multiplied by the organ's cap for that factor.
     An organ's caps add up to 1, so the day's points run from 1 down to -1. Mental health has only noise (cap 1).
   A day with less than 50% of its driver weight logged, or no log at all, earns 0 points but still counts as a day of the stay.
   Weights, caps and level tables are hand-set conventions anchored to the cited limits, not fitted values. */
const FMAP = Object.fromEntries(FACTORS.map(f => [f.k, f])), labelOf = k => k === 'exercise' ? 'Exercise' : k === 'hydration' ? 'Hydration (urine colour)' : FMAP[k].l;
const ORGANS = [
  { k: 'brain', l: 'Brain', fit: [1, .7, .9], d: { omega3_g: .20, vit_b12_ug: .13, sleep_h: .13, magnesium_mg: .11, exercise: .10, folate_ug: .09, zinc_mg: .07, iron_mg: .07, vit_b6_mg: .06, vit_d_ug: .04 },
    why: 'Omega-3 builds neuron membranes; B12, folate and B6 make neurotransmitters; magnesium and zinc tune signalling; iron carries oxygen. Sleep and aerobic exercise drive repair and growth factors.' },
  { k: 'eyes', l: 'Eyes', fit: [1, 1, 1], d: { vit_a_ug: .30, omega3_g: .15, vit_c_mg: .15, vit_e_mg: .12, zinc_mg: .12, vit_b2_mg: .08, vit_b12_ug: .08 },
    why: 'Vitamin A forms the light-sensing pigment; vitamins C and E, zinc and omega-3 protect the retina; B2 and B12 support the lens and optic nerve.' },
  { k: 'bone', l: 'Bone', fit: [.5, 1, .9], d: { calcium_mg: .22, vit_d_ug: .20, exercise: .20, vit_k_ug: .12, magnesium_mg: .10, protein_g: .09, potassium_mg: .04, vit_c_mg: .03 },
    why: 'Calcium and vitamin D supply the mineral; vitamin K and magnesium direct it into bone; protein builds the matrix; loading exercise (strength, impact) is the signal that keeps bone.' },
  { k: 'muscles', l: 'Muscles', fit: [.4, 1, .9], d: { protein_g: .30, exercise: .25, vit_d_ug: .10, kcal: .10, magnesium_mg: .08, potassium_mg: .08, zinc_mg: .05, vit_b6_mg: .04 },
    why: 'Protein and enough energy are the raw material; vitamin D, magnesium, potassium, zinc and B6 support contraction and repair; strength training is the signal to build.' },
  { k: 'kidney', l: 'Kidneys', fit: [1, 1, 1], d: { hydration: .20, water_l: .25, sodium_mg: .12, potassium_mg: .10, magnesium_mg: .08, exercise: .08, calcium_mg: .07, vit_b6_mg: .05, vit_d_ug: .05 },
    why: 'Water and hydration (urine colour) are the main drivers. Keeping sodium at or under its limit, and enough potassium, magnesium and calcium, protects against stones and pressure damage; exercise helps blood-pressure control.' },
  { k: 'heart', l: 'Heart', fit: [1, .6, .9], d: { exercise: .22, omega3_g: .18, potassium_mg: .14, magnesium_mg: .12, fiber_g: .12, sodium_mg: .10, sleep_h: .08, folate_ug: .04 },
    why: 'Aerobic exercise is the strongest protector; omega-3, potassium and magnesium support rhythm and pressure; fibre helps cholesterol; low sodium helps pressure; sleep supports recovery.' },
  { k: 'mental', l: 'Mental health', space: true, fit: [1, .8, .9], d: { mood: .45, social_min: .30, exercise: .25 },
    why: 'Mood is the direct reading of how you are doing; social contact with the crew, family and calls protects against isolation; regular exercise, cardio most of all, supports mood and stress resistance.' },
];
ORGANS.forEach(o => { const t = Object.values(o.d).reduce((a, b) => a + b, 0); if (Math.abs(t - 1) > 1e-9) throw new Error(`organ ${o.k}: weights sum to ${t}`); Object.keys(o.d).forEach(k => { if (k !== 'exercise' && k !== 'hydration' && !FMAP[k]) throw new Error(`organ ${o.k}: unknown driver ${k}`); }); });

const ENV = [
  // [lower bound, severity 0..1]; the last bound reached applies. Noise: WHO/EPA/OSHA bands (0-30 quiet, 55 annoyance, 70-75 chronic stress, 85 hearing damage, 120+ pain).
  { k: 'noise', f: 'noise_dba', l: 'Cabin noise', u: 'dBA', tiers: [[0, 0], [30.01, .05], [45, .25], [55, .60], [70, .75], [85, .90], [100, .95], [120, 1]] },
  // CO2 (ppm at 1 atm): 1,000 Health Canada / first cognitive effects, 1,500 symptoms, 2,500 decision-making, 3,290 ISS headache reference (2.5 mmHg),
  // 3,950 NASA 1-hour limit (3 mmHg), 4,605 (3.5 mmHg), 5,921 (4.5 mmHg), 6,579 (5 mmHg), 8,000 NRC submarine 90-day ceiling.
  { k: 'co2', f: 'co2_ppm', l: 'Cabin CO2', u: 'ppm', tiers: [[0, 0], [1000, .08], [1500, .20], [2500, .40], [3290, .55], [3950, .70], [4605, .80], [5921, .90], [6579, .95], [8000, 1]] },
  // Radiation (mSv per day): 1.37 = rate that uses up NASA's 1-year blood-forming-organ limit (500 mSv), 8.33 = rate that uses up the 30-day limit (250 mSv).
  { k: 'radiation', f: 'radiation_msv', l: 'Radiation', u: 'mSv/day', tiers: [[0, 0], [.2, .10], [.4, .25], [.7, .45], [1.37, .70], [3, .85], [8.33, 1]] },
];
const CAPS = {                                         // max daily penalty per organ and environment factor; each organ sums to 1
  brain: { noise: .25, co2: .40, radiation: .35 }, eyes: { noise: .15, co2: .30, radiation: .55 }, bone: { noise: .10, co2: .30, radiation: .60 },
  muscles: { noise: .15, co2: .35, radiation: .50 }, kidney: { noise: .20, co2: .30, radiation: .50 }, heart: { noise: .30, co2: .30, radiation: .40 },
  mental: { noise: 1 },
};
Object.entries(CAPS).forEach(([k, c]) => { const t = Object.values(c).reduce((a, b) => a + b, 0); if (Math.abs(t - 1) > 1e-9) throw new Error(`organ ${k}: penalty caps sum to ${t}`); });
const sevOf = (tiers, x) => tiers.reduce((s, [lo, v]) => x >= lo ? v : s, 0);
/** Today's negative constants for one organ: severity of each environment level times the organ's cap. A reading that was not logged costs nothing. */
function envPenalty(organ, v) {
  const caps = CAPS[organ] || {};
  return ENV.filter(e => caps[e.k]).map(e => { const x = v[e.f], ok = Number.isFinite(x), sev = ok ? sevOf(e.tiers, x) : 0;
    return { k: e.k, l: e.l, u: e.u, x: ok ? r3(x) : null, sev, cap: caps[e.k], penalty: r3(caps[e.k] * sev), missing: !ok }; });
}

const adequacyOne = (f, x, ref) => f.t === 'limit' ? (x <= ref ? 1 : Math.max(0, 1 - (x - ref) / ref)) : clamp(x / ref, 0, 1);
/** Daily variable v = intake / average(RDA, your normal), held in 0..1. Limits use the limit alone. */
const adequacy = (f, x, ref, normal = null) =>
  f.t !== 'limit' && normal != null && normal > 0 ? clamp(x / ((ref + normal) / 2), 0, 1) : adequacyOne(f, x, ref);

function organDay(o, d, R, norm = {}, space = false) {
  const rows = []; let cov = 0, sum = 0;
  for (const [k, w] of Object.entries(o.d)) {
    let a = null, x = null, ref = null, info = '', u = '', normal = null;
    if (k === 'hydration') {
      const h = URINE[d.v.urine_color - 1]; ref = 3; u = 'colour';
      if (h) { x = h.n; a = h.a; info = `Colour ${h.n}: ${h.s.toLowerCase()}`; }
    } else if (k === 'exercise') {
      const e = exerciseOf(d), f = FMAP.exercise_min; ref = R.exercise_min; u = 'min'; normal = norm.exercise_min ?? null;
      if (e.min != null) { x = e.min;
        a = e.type === 0 ? 0 : adequacy(f, e.min, ref, normal) * o.fit[e.type - 1];       // minutes against average(RDA, normal), then how well this kind of exercise suits the organ
        info = e.type === 0 ? 'No exercise' : `${r2(e.min)} min, ${ETYPES[e.type]}`; }
    } else { const f = FMAP[k]; ref = R[k]; u = f.u; x = d.v[k]; normal = f.t === 'limit' ? null : norm[k] ?? null; if (Number.isFinite(x)) a = adequacy(f, x, ref, normal); else x = null; }
    rows.push({ k, l: labelOf(k), u, w, x: x == null ? null : r3(x), ref, normal: normal == null ? null : r3(normal), a: a == null ? null : r3(a), lost: a == null ? null : r3(w * (1 - a)), info, missing: a == null });
    if (a != null) { cov += w; sum += w * a; }
  }
  const score = cov >= .5 ? sum / cov : null;
  const penalties = space ? envPenalty(o.k, d.v) : [], penalty = penalties.reduce((s, x) => s + x.penalty, 0);
  return { rows, coverage: r2(cov), score, penalties, penalty: r3(penalty), points: score == null ? null : r3(score - penalty) };
}
function organsFor(q) {
  const p = getProfile(), R = refFor(p), days = loadEff(p), N = days.length, space = p.mode === 'space';
  if (!N) return { as_of: null, organs: [], exercise_normal: exerciseNormal([]) };
  const sp = splitSpace(days, p), inSpace = space && !!sp.start, latest = days[N - 1].date, date = validDate(q.date) ? q.date : latest;
  // length of the stay: the days asked for, else the whole stay in space mode (since mission start) or 30 days on earth
  // space mode: the bar is fixed to the planned stay (mission start to planned return) and counts DOWN from full; it never depends on a "days counted" box
  const asked = Math.floor(+q.days), missionLen = inSpace && p.mission_end && p.mission_end >= sp.start ? dayDiff(sp.start, p.mission_end) + 1 : 0;
  const W = inSpace ? clamp(missionLen || sp.age, 1, 3650) : asked >= 1 ? clamp(asked, 1, 365) : 30;
  const start = inSpace ? sp.start : [addDays(latest, -(W - 1)), days[0].date].reduce((a, b) => a > b ? a : b);
  const count = inSpace ? clamp(dayDiff(start, latest) + 1, 0, W) : Math.max(0, dayDiff(start, latest) + 1);
  // normals: earth average always; in space mode the space average replaces it after SPACE_SHIFT_DAYS space days (needs >= 10 space days)
  const baseDays = inSpace ? sp.earth : days, spaceOk = inSpace && sp.space.length >= 10, earthN = {}, spaceN = {};
  for (const k of new Set([...ORGANS.flatMap(o => Object.keys(o.d)), 'exercise_min'])) { const f = FMAP[k === 'exercise' ? 'exercise_min' : k];
    if (f) { earthN[f.k] = stats(f, baseDays).normal; if (spaceOk) spaceN[f.k] = stats(f, sp.space).normal; } }
  const mixedN = { ...earthN }; Object.entries(spaceN).forEach(([k, v]) => { if (v != null) mixedN[k] = v; });
  const onSpaceNormal = d => spaceOk && dayDiff(sp.start, d.date) + 1 > SPACE_SHIFT_DAYS, normFor = d => onSpaceNormal(d) ? mixedN : earthN;
  const byDate = new Map(days.map(d => [d.date, d]));
  const organs = ORGANS.filter(o => !o.space || space).map(o => {
    const base = { k: o.k, l: o.l, why: o.why, weights: Object.entries(o.d).map(([k, w]) => ({ k, l: labelOf(k), w })), caps: CAPS[o.k] || {} };
    let tot = 0, scored = 0, lostPts = 0; const series = [], pen = {}, rowsBy = {};
    for (let i = 0; i < count; i++) {
      const dt = addDays(start, i), d = byDate.get(dt), od = d ? organDay(o, d, R, normFor(d), inSpace) : null, ok = !!od && od.points != null, pts = ok ? od.points : 0;
      tot += pts; lostPts += 1 - pts; if (ok) scored++;     // a full day (1 point) loses nothing; 0.7 loses 0.3; an unlogged day loses 1
      series.push({ date: dt, points: r3(pts), total: r2(tot), logged: !!d, scored: ok });
      if (ok) { od.penalties.forEach(x => { pen[x.k] = (pen[x.k] || 0) + x.penalty; }); od.rows.forEach(r => { if (!r.missing) (rowsBy[r.k] = rowsBy[r.k] || []).push(r); }); }
    }
    const drag = Object.keys(o.d).map(k => { const rs = rowsBy[k]; if (!rs) return null; const a = mean(rs.map(r => r.a));
      return { k, l: labelOf(k), w: o.d[k], a: r2(a), lost: r2(rs.reduce((s, r) => s + r.lost, 0)), days: rs.length }; }).filter(x => x && x.lost > .004).sort((a, b) => b.lost - a.lost).slice(0, 3);
    const env = ENV.filter(e => CAPS[o.k] && CAPS[o.k][e.k]).map(e => ({ k: e.k, l: e.l, cap: CAPS[o.k][e.k], lost: r2(pen[e.k] || 0) })).sort((a, b) => b.lost - a.lost);
    const dd = byDate.get(date), od = dd ? organDay(o, dd, R, normFor(dd), inSpace && date >= sp.start) : null;
    const left = clamp(W - lostPts, 0, W);       // space mode: starts full (W) and only goes down
    return { ...base, total: inSpace ? r2(left) : r2(tot), max: inSpace ? W : count, pct: inSpace ? r2(left / W * 100) : count ? r2(tot / count * 100) : null, n_scored: scored, series, drag, env,
      today: od ? { date, points: od.points, score: od.score == null ? null : r3(od.score), coverage: od.coverage, penalty: od.penalty, penalties: od.penalties,
        baseline: dd && inSpace && onSpaceNormal(dd) ? 'space' : 'earth', drivers: [...od.rows].sort((a, b) => (b.lost ?? -1) - (a.lost ?? -1)) } : null };
  });
  return { as_of: latest, date, n_days: N, window_days: W, period_start: start, period_days: count, mode: p.mode, exercise_normal: exerciseNormal(normDays(days, p)), organs };
}

/* ============================== alerts and reminders ============================== */
function alertsFor(n) {
  const p = getProfile(),
        R = refFor(p),
        days = loadEff(p),
        by = new Map(days.map(d => [d.date, d])),
        vf = FACTORS.filter(vis(p)),
        nd = normDays(days, p),
        S = {};

  vf.forEach(f => S[f.k] = stats(f, nd));

  const kb = Object.fromEntries(
    db.prepare('SELECT key,label,red FROM symptom_kb')
      .all()
      .map(r => [r.key, r])
  );

  const urgent = p.mode === 'space'
    ? 'contact the flight surgeon / ground medical team'
    : 'seek medical care promptly';

  const cmp = (v, ref, w) =>
    `${Math.round(Math.abs(v - ref) / ref * 100)}% ${v < ref ? 'below' : 'above'} ${w}`;

  const WORD = {
    rda: 'RDA',
    limit: 'limit',
    avg: 'average',
    pre: 'baseline weight'
  };

  const rank = { RED: 0, YELLOW: 1, INFO: 2, GREEN: 3 },
        out = [],
        end = todayISO();

  for (let i = 0; i < n; i++) {
    const date = addDays(end, -i),
          d = by.get(date);

    if (!d) {
      out.push({
        date,
        status: 'MISSING',
        items: [{
          level: 'INFO',
          type: 'reminder',
          text: 'No log for this day yet. Add it so your normal stays accurate.'
        }]
      });

      continue;
    }

    const items = [],
          add = (level, type, text, extra = {}) => items.push({ level, type, text, ...extra });

    for (const [k, sev] of Object.entries(d.s)) {
      const s = kb[k] || { label: k };

      if (s.red) {
        add('RED', 'symptom', `${s.label} reported. This can be serious: ${urgent}.`);
      } else if (sev >= 5) {
        add('RED', 'symptom', `Severe ${s.label.toLowerCase()} (5/5): ${urgent}.`);
      } else if (sev >= 4) {
        add('YELLOW', 'symptom', `${s.label} rated ${sev}/5. Monitor closely.`);
      }

      let c = 0;
      for (let j = 0; j < 5; j++) {
        if (by.get(addDays(date, -j))?.s[k]) c++;
      }

      if (c >= 3 && !s.red) {
        add('YELLOW', 'symptom', `${s.label} on ${c} of the last 5 days. If it continues, talk to a clinician.`);
      }
    }

    for (const f of vf) {
      const v = d.v[f.k];
      if (v == null) continue;

      const ref = R[f.k],
            st = S[f.k],
            lvl = flag(f, v, ref),
            z = st.sd ? (v - st.normal) / st.sd : null,
            offNormal = z != null && (f.bad === 'high' ? z >= 1 : z <= -1);

      if (lvl < 2 && !offNormal) continue;

      add(
        lvl === 2 ? 'YELLOW' : 'INFO',
        'benchmark',
        `${f.l} ${r2(v)} ${f.u}: ${cmp(v, ref, WORD[f.t])} (${r2(ref)})` +
        (st.normal != null ? `; ${cmp(v, st.normal, 'your normal')} (${r2(st.normal)}).` : '.'),
        {
          factor: f.k,
          value: v,
          ref,
          normal: st.normal,
          z: z == null ? null : r2(z)
        }
      );
    }

    const uc = d.v.urine_color, uo = URINE[uc - 1];

    if (uo) {
      const kg = d.v.weight_kg ?? p.weight_kg, L = waterNeedL(uo, kg);
      add(
        uo.med ? 'RED' : uc >= 5 ? 'YELLOW' : uc >= 3 ? 'INFO' : 'GREEN',
        'hydration',
        `Urine colour ${uc}: ${uo.s.toLowerCase()}, about ${uo.loss} of body mass lost to dehydration. ` +
        (uo.avg ? `Drink about ${L} L water over the next few hours${uo.med ? ' and seek medical attention' : ''}.` : 'No extra water needed.'),
        { factor: 'urine_color', value: uc, loss: uo.loss, extra_l: L }
      );
    }

    items.sort((a, b) => rank[a.level] - rank[b.level]);

    out.push({
      date,
      status: items.some(x => x.level === 'RED')
        ? 'RED'
        : items.some(x => x.level === 'YELLOW')
          ? 'YELLOW'
          : items.some(x => x.level !== 'GREEN')
            ? 'INFO'
            : 'GREEN',
      items
    });
  }

  return out;
}

/* ============================== demo data (planted effects so the statistics have something to find) ============================== */
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function clearAll() { tx(() => ['entries', 'symptom_logs', 'days'].forEach(t => db.exec(`DELETE FROM ${t}`))); }
function demo(n = 90) {
  const today = todayISO(), PRE = 20;                                       // space demo: 20 pre-flight days, then the mission
  if (getProfile().mode === 'space') db.prepare('UPDATE profile SET mission_start=?, mission_end=?, career_dose_msv=?, career_limit_msv=? WHERE id=1').run(addDays(today, -(n - 1 - PRE)), addDays(today, 150), 80, 600);
  const p = getProfile(), space = p.mode === 'space', R = refFor(p), rnd = mulberry(42), gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const FR = { kcal: .95, protein_g: 1.1, carb_g: 1.7, fat_g: 1, fiber_g: .65, vit_a_ug: .8, vit_c_mg: .95, vit_d_ug: .5, vit_e_mg: .7, vit_k_ug: 1, vit_b1_mg: 1.05, vit_b2_mg: 1.1,
    vit_b3_mg: 1.2, vit_b6_mg: .95, folate_ug: .85, vit_b12_ug: 1.3, calcium_mg: .75, iron_mg: .85, magnesium_mg: .72, zinc_mg: .85, potassium_mg: .62, sodium_mg: 1.25 };
  const LS = { sleep_h: [6.9, .9], exercise_min: [30, 22], water_l: [2.2, .5], spo2_pct: [97.5, .8], weight_kg: [R.weight_kg, .4], resist_min: [10, 12], aerobic_min: [20, 15],
    radiation_msv: [.01, .002], co2_ppm: [450, 40], noise_dba: [45, 3], mood: [3.7, .5], social_min: [60, 30] };
  const LF = { sleep_h: [6.4, .9], exercise_min: [105, 25], spo2_pct: [96.5, .9], resist_min: [35, 15], aerobic_min: [55, 20], radiation_msv: [.55, .08],
    co2_ppm: [3300, 550], noise_dba: [58, 3], mood: [3.6, .5], social_min: [40, 25] };                    // in-flight overrides
  const vf = FACTORS.filter(vis(p)).filter(f => f.k !== 'omega3_g'), fl = t => space && t >= PRE, S = {};
  vf.forEach(f => { let e = 0; const E = Array.from({ length: n }, () => (e = .35 * e + Math.sqrt(1 - .35 ** 2) * gauss()));
    S[f.k] = E.map((z, t) => { const [m, s] = FR[f.k] ? [R[f.k] * FR[f.k], R[f.k] * FR[f.k] * .24] : (fl(t) && LF[f.k]) || LS[f.k];
      const mm = f.k === 'weight_kg' && fl(t) ? R.weight_kg * (1 - .06 * clamp((t - PRE) / 60, 0, 1)) : m, v = Math.max(0, mm + s * z); return f.k === 'mood' ? clamp(v, 1, 5) : f.k === 'spo2_pct' ? Math.min(100, v) : v; }); });
  // planted effects use the same window the analysis uses (the 3 days before the symptom day; sleep = the 3 nights ending that morning)
  const wz = k => { const w = S[k].map((_, t) => mean(k === 'sleep_h' ? S[k].slice(Math.max(0, t - 2), t + 1) : S[k].slice(Math.max(0, t - 3), Math.max(1, t)))), m = mean(w), s = sdev(w, m) || 1; return w.map(v => (v - m) / s); };
  const EFF = { headache: [-1.1, { magnesium_mg: -1, sleep_h: -.6 }], fatigue: [-1.0, { iron_mg: -.9, sleep_h: -.5 }], muscle_cramps: [-1.6, { potassium_mg: -1, water_l: -.4 }], bone_joint_pain: [-1.9, { vit_d_ug: -.8 }] };
  if (space) { EFF.headache[1].co2_ppm = .9; EFF.back_pain = [-2.3, { resist_min: -.9 }]; EFF.muscle_weakness = [-2.4, { resist_min: -.8 }]; EFF.loneliness = [-2.3, { social_min: -.9 }]; EFF.insomnia = [-2.2, { noise_dba: .8 }]; }
  const Z = {}; new Set(Object.values(EFF).flatMap(([, c]) => Object.keys(c))).forEach(k => Z[k] = wz(k));
  const rx = mulberry(77), gx = () => Math.sqrt(-2 * Math.log(rx() + 1e-12)) * Math.cos(2 * Math.PI * rx()), oz = Array.from({ length: n }, gx);
  const OV = {}; if (space) { OV[n - 12] = { radiation_msv: 6 }; OV[n - 5] = { co2_ppm: 7400 }; }       // a radiation excursion, a CO2 excursion     // a solar particle event, a CO2 excursion
  tx(() => {
    ['entries', 'symptom_logs', 'days'].forEach(t => db.exec(`DELETE FROM ${t}`));
    const di = db.prepare('INSERT INTO days VALUES(?,?)'), ei = db.prepare('INSERT INTO entries VALUES(?,?,?)'), si = db.prepare('INSERT INTO symptom_logs VALUES(?,?,?)');
    for (let t = 0; t < n; t++) {
      if (rnd() < .05 && t > PRE + 4 && t < n - 14) continue;
      const date = addDays(today, -(n - 1 - t)), md = fl(t) ? t - PRE + 1 : null, sym = {}; di.run(date, '');
      vf.forEach(f => { const o = OV[t] && OV[t][f.k]; if (o != null || rnd() > .03) ei.run(date, f.k, r2(o != null ? o : S[f.k][t])); });
      if (rx() > .03) ei.run(date, 'omega3_g', r2(Math.max(0, R.omega3_g * .55 * (1 + .3 * oz[t]))));       // exercise type for the day (separate random stream)
      { const m = S.exercise_min[t], u2 = rx(); if (m < 5) ei.run(date, 'exercise_type', 0); else ei.run(date, 'exercise_type', u2 < .55 ? 1 : u2 < .8 ? 2 : 3); }
      if (S.water_l) ei.run(date, 'urine_color', clamp(Math.round(7.5 - 4.5 * S.water_l[t] / R.water_l), 1, 8));
      for (const [k, [b0, co]] of Object.entries(EFF)) { const pr = sig(b0 + Object.entries(co).reduce((a, [f, c]) => a + c * Z[f][t], 0)); if (rnd() < pr) sym[k] = clamp(1 + Math.floor(rnd() * 3) + (pr > .6 ? 1 : 0), 1, 5); }
      if (md != null && md <= 3) { if (rnd() < .8) sym.nausea = 3 + (rnd() < .4 ? 1 : 0); if (rnd() < .6) sym.dizziness = 3; if (rnd() < .5) sym.space_motion_sickness = 3; }
      if (md != null && md <= 10 && rnd() < .7) sym.facial_congestion = 2;
      if (space && (t === n - 9 || t === n - 2)) sym.vision_eye = 2;                                               // two vision days: the SANS watch rule fires
      for (const [k, v] of Object.entries(sym)) si.run(date, k, v);
    }
  });
  return db.prepare('SELECT COUNT(*) c FROM days').get().c;
}

/* ============================== image diagnosis (PatchCore): kidney + eye ============================== */
const DIAG_COMMON = {
  python: process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3'),
  timeoutMs: +process.env.DIAG_TIMEOUT_MS || +process.env.KIDNEY_TIMEOUT_MS || 240000,
  maxImages: 10,
  maxImageBytes: 6 * 1024 * 1024
};
DIAG_COMMON.maxBody = Math.ceil(DIAG_COMMON.maxImages * DIAG_COMMON.maxImageBytes * 4 / 3) + 1e6;   // base64 is ~4/3 of the file size
const DIAG = {
  kidney: {
    name: 'kidney', label: 'Kidney diagnosis', table: 'kidney_tests', trainScript: 'train_patchcore.py', modelEnv: 'KIDNEY_MODEL',
    script: process.env.KIDNEY_SCRIPT || path.join(__dirname, 'run_patchcore.py'),
    model: process.env.KIDNEY_MODEL || path.join(__dirname, 'models', 'patchcore_model.pt'),
    url: process.env.KIDNEY_URL || 'http://127.0.0.1:5001',        // patchcore_service.py (tried first; falls back to spawning Python)
    busy: false                                                    // one analysis at a time per model (memory)
  },
  eye: {
    name: 'eye', label: 'Eye diagnosis', table: 'eye_tests', trainScript: 'train_eye.py', modelEnv: 'EYE_MODEL',
    script: process.env.EYE_SCRIPT || path.join(__dirname, 'run_eye.py'),
    model: process.env.EYE_MODEL || path.join(__dirname, 'models', 'eye_model.pt'),
    url: process.env.EYE_URL || 'http://127.0.0.1:5002',           // eye_service.py
    busy: false
  }
};
/** Kidney and eye diagnosis are spaceflight-mode features (like the Health page): refuse the request on Earth mode. */
function requireSpace(cfg) {
  if (getProfile().mode !== 'space') throw httpErr(403, `${cfg.label} is only available in spaceflight mode. Switch the mode in Profile.`);
}

for (const c of Object.values(DIAG)) db.exec(`
CREATE TABLE IF NOT EXISTS ${c.table}(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  taken_date TEXT NOT NULL,
  taken_time TEXT NOT NULL,
  created_at TEXT NOT NULL,
  probability REAL NOT NULL,
  verdict TEXT NOT NULL,
  message TEXT,
  n_images INTEGER,
  n_flagged INTEGER,
  details TEXT
);
CREATE INDEX IF NOT EXISTS ${c.table}_date ON ${c.table}(taken_date);
`);

/** File type from the first bytes (never trust the browser-supplied name or MIME type). */
function imageExt(b) {
  if (b.length < 12) return null;
  if (b[0] === 0xFF && b[1] === 0xD8 && b[2] === 0xFF) return '.jpg';
  if (b.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))) return '.png';
  if (b[0] === 0x42 && b[1] === 0x4D) return '.bmp';
  if ((b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2A && b[3] === 0) || (b[0] === 0x4D && b[1] === 0x4D && b[2] === 0 && b[3] === 0x2A)) return '.tif';
  if (b.subarray(0, 4).toString('latin1') === 'RIFF' && b.subarray(8, 12).toString('latin1') === 'WEBP') return '.webp';
  return null;
}

/** Asks the running Python service (patchcore_service.py / eye_service.py) to analyse the saved files. Resolves null when the service is not running (caller falls back to spawning Python). */
async function runViaService(cfg, files) {
  let r;
  try {
    r = await fetch(cfg.url + '/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ files }), signal: AbortSignal.timeout(DIAG_COMMON.timeoutMs) });
  } catch (e) {
    if (e && e.name === 'TimeoutError') throw httpErr(504, 'The analysis took too long and was stopped. Try fewer or smaller images.');
    return null;                                                                       // service not reachable
  }
  let j = null; try { j = await r.json(); } catch (e) { }
  if (j && j.ok) return j;
  throw httpErr(500, 'The analysis failed: ' + String((j && j.error) || 'the service returned an unreadable answer').slice(0, 400));
}

/** Analyses the saved image files: through the running service if there is one, otherwise by starting the Python script. */
async function runModel(cfg, files, dir) {
  const viaService = await runViaService(cfg, files);
  if (viaService) return viaService;
  return runModelProcess(cfg, files, dir);
}

/** Runs the Python script on the saved image files and resolves with its parsed JSON result. */
function runModelProcess(cfg, files, dir) {
  return new Promise((resolve, reject) => {
    const out = path.join(dir, 'result.json');
    const args = [cfg.script, '--model', cfg.model, '--json_out', out, '--input', ...files];
    let err = '', settled = false, timedOut = false;
    const done = (fn, v) => { if (!settled) { settled = true; clearTimeout(timer); fn(v); } };
    const cp = spawn(DIAG_COMMON.python, args, { cwd: __dirname, windowsHide: true });      // no shell: arguments cannot be injected
    const timer = setTimeout(() => { timedOut = true; cp.kill(); }, DIAG_COMMON.timeoutMs);
    cp.stdout.on('data', () => { });
    cp.stderr.on('data', d => { err = (err + d).slice(-1500); });
    cp.on('error', e => done(reject, httpErr(500, e.code === 'ENOENT'
      ? `Python was not found ("${DIAG_COMMON.python}"). Install Python or set the PYTHON environment variable to its path.`
      : 'Could not start the analysis: ' + e.message)));
    cp.on('close', () => {
      if (timedOut) return done(reject, httpErr(504, 'The analysis took too long and was stopped. Try fewer or smaller images.'));
      let r = null;
      try { r = JSON.parse(fs.readFileSync(out, 'utf8')); } catch (e) { }
      if (r && r.ok) return done(resolve, r);
      const why = (r && r.error) || err.trim().split('\n').slice(-2).join(' ') || 'unknown error';
      done(reject, httpErr(500, 'The analysis failed: ' + why.slice(0, 400)));
    });
  });
}

async function diagnose(cfg, b) {
  const imgs = Array.isArray(b.images) ? b.images : [];
  if (!imgs.length) throw httpErr(400, 'Add at least one image first');
  if (imgs.length > DIAG_COMMON.maxImages) throw httpErr(400, `Send at most ${DIAG_COMMON.maxImages} images at a time`);
  const files = imgs.map((im, i) => {
    const raw = im && typeof im.data === 'string' ? im.data.replace(/^data:[^,]*,/, '') : '';
    const buf = Buffer.from(raw, 'base64');
    if (!buf.length) throw httpErr(400, `Image ${i + 1} is empty`);
    if (buf.length > DIAG_COMMON.maxImageBytes) throw httpErr(400, `Image ${i + 1} is larger than ${DIAG_COMMON.maxImageBytes / 1048576} MB`);
    const ext = imageExt(buf);
    if (!ext) throw httpErr(400, `Image ${i + 1} is not a supported picture (use JPG, PNG, BMP, TIFF or WebP)`);
    return { buf, ext, name: String(im.name || `image ${i + 1}`).slice(0, 80) };
  });
  if (!fs.existsSync(cfg.model)) throw httpErr(503, `No trained model found at ${cfg.model}. Train one with ${cfg.trainScript}, then put it there or set ${cfg.modelEnv}.`);
  if (cfg.busy) throw httpErr(429, 'Another analysis is still running. Please try again in a moment.');
  cfg.busy = true;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), cfg.name + '-'));
  try {
    const paths = files.map((f, i) => { const p = path.join(dir, `img${i + 1}${f.ext}`); fs.writeFileSync(p, f.buf); return p; });
    const r = await runModel(cfg, paths, dir);
    (r.images || []).forEach((it, i) => { if (files[i]) it.image = files[i].name; });
    // date/time come from the person's own clock (like the rest of the app), falling back to the server's
    const nowD = todayISO(), nowT = new Date().toTimeString().slice(0, 5);
    const okD = validDate(b.local_date) && Math.abs(dayDiff(nowD, b.local_date)) <= 1;
    const date = okD ? b.local_date : nowD, time = okD && /^([01]\d|2[0-3]):[0-5]\d$/.test(String(b.local_time)) ? b.local_time : nowT;
    const info = db.prepare(`INSERT INTO ${cfg.table}(taken_date,taken_time,created_at,probability,verdict,message,n_images,n_flagged,details) VALUES(?,?,?,?,?,?,?,?,?)`)
      .run(date, time, new Date().toISOString(), r.probability, r.verdict, r.message, r.n_images, r.n_flagged, JSON.stringify({ notes: r.notes, images: r.images, borderline: r.borderline }));
    return { id: Number(info.lastInsertRowid), taken_date: date, taken_time: time, probability: r.probability, verdict: r.verdict, message: r.message,
      notes: r.notes, borderline: r.borderline, n_images: r.n_images, n_flagged: r.n_flagged, images: r.images };
  } finally {
    cfg.busy = false;
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch (e) { }
  }
}

function diagTests(cfg, month) {
  const m = /^\d{4}-(0[1-9]|1[0-2])$/.test(String(month)) ? month : todayISO().slice(0, 7);
  return { month: m, tests: db.prepare(`SELECT id,taken_date,taken_time,probability,verdict,message,n_images,n_flagged FROM ${cfg.table} WHERE substr(taken_date,1,7)=? ORDER BY taken_date DESC, taken_time DESC, id DESC`).all(m) };
}

/* ============================== http ============================== */
const VENDOR = { '/vendor/react.js': 'react/umd/react.production.min.js', '/vendor/react-dom.js': 'react-dom/umd/react-dom.production.min.js', '/vendor/babel.js': '@babel/standalone/babel.min.js' };
const vendorPath = u => path.join(__dirname, 'node_modules', VENDOR[u]);
/** Offline-first: if react, react-dom and @babel/standalone are installed (npm install), the page uses them and needs no internet. Otherwise it falls back to the CDN. */
function shell() {
  const local = Object.keys(VENDOR).every(u => fs.existsSync(vendorPath(u))), src = (u, cdn) => local ? u : cdn;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Crew Health</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"></head>
<body><div id="root"></div>
<script src="${src('/vendor/react.js', 'https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js')}"></script>
<script src="${src('/vendor/react-dom.js', 'https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js')}"></script>
<script src="${src('/vendor/babel.js', 'https://cdn.jsdelivr.net/npm/@babel/standalone@7.26.4/babel.min.js')}"></script>
<script type="text/babel" data-presets="react" src="/App.jsx"></script></body></html>`;
}
const send = (res, code, body, type = 'application/json') => { res.writeHead(code, { 'Content-Type': type + '; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer' });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body)); };
const readJson = (req, max = 1e6) => new Promise((ok, no) => { let s = ''; req.on('data', c => { s += c; if (s.length > max) { no(httpErr(413, 'Body too large')); req.destroy(); } });
  req.on('end', () => { try { ok(s ? JSON.parse(s) : {}); } catch (e) { no(httpErr(400, 'Invalid JSON')); } }); req.on('error', no); });

const server = http.createServer(async (req, res) => {
  try {
    const u = new URL(req.url, 'http://x'), P = u.pathname, q = Object.fromEntries(u.searchParams), M = req.method;
    if (LOOPBACK && !HOSTS.has(String(req.headers.host || '').toLowerCase().replace(/:\d+$/, ''))) return send(res, 403, { error: 'Unexpected Host header' });
    if (req.headers.origin) { let o = ''; try { o = new URL(req.headers.origin).host; } catch (e) { } if (o !== req.headers.host) return send(res, 403, { error: 'Cross-site request blocked' }); }
    if ((M === 'POST' || M === 'PUT') && !/^application\/json/i.test(req.headers['content-type'] || '')) return send(res, 415, { error: 'Send JSON (Content-Type: application/json)' });
    if (M === 'GET' && (P === '/' || P === '/index.html')) return send(res, 200, shell(), 'text/html');
    if (M === 'GET' && VENDOR[P] && fs.existsSync(vendorPath(P))) return send(res, 200, fs.readFileSync(vendorPath(P)), 'application/javascript');
    if (M === 'GET' && P === '/App.jsx') return send(res, 200, fs.readFileSync(path.join(__dirname, 'App.jsx')), 'text/babel');
    if (!P.startsWith('/api/')) return send(res, 404, { error: 'Not found' });
    const route = P.slice(5), body = M === 'POST' || M === 'PUT' ? await readJson(req, /^(kidney|eye)\/diagnose$/.test(route) ? DIAG_COMMON.maxBody : undefined) : {};
    if (M === 'GET' && route === 'meta') return send(res, 200, meta());
    if (M === 'PUT' && route === 'profile') {
      const b = body, okd = v => v == null || v === '' || validDate(v), inr = (v, lo, hi) => v == null || v === '' || (+v >= lo && +v <= hi);
      const ok = b.age >= 10 && b.age <= 100 && ['male', 'female', 'other'].includes(b.sex) && b.weight_kg >= 30 && b.weight_kg <= 250 && b.height_cm >= 120 && b.height_cm <= 230 && ['earth', 'space'].includes(b.mode)
        && okd(b.mission_start) && okd(b.mission_end) && !(b.mission_start && b.mission_end && b.mission_end < b.mission_start) && inr(b.career_dose_msv, 0, 3000) && inr(b.career_limit_msv, 50, 3000);
      if (!ok) throw httpErr(400, 'Check age (10-100), sex, weight (30-250 kg), height (120-230 cm), mode, mission dates (valid, return after start) and radiation figures in mSv');
      db.prepare('UPDATE profile SET name=?,age=?,sex=?,weight_kg=?,height_cm=?,mode=?,mission_start=?,mission_end=?,career_dose_msv=?,career_limit_msv=? WHERE id=1')
        .run(String(b.name || '').slice(0, 60), +b.age, b.sex, +b.weight_kg, +b.height_cm, b.mode, b.mission_start || null, b.mission_end || null, +b.career_dose_msv || 0, +b.career_limit_msv || 600);
      return send(res, 200, meta());
    }
    if (M === 'GET' && route === 'days') return send(res, 200, load().map(d => ({ date: d.date, values: d.v, symptoms: d.s, note: d.note })));
    const dm = route.match(/^days\/(\d{4}-\d{2}-\d{2})$/);
    if (dm && M === 'PUT') { if (!validDate(dm[1])) throw httpErr(400, 'Not a real calendar date'); if (dm[1] > addDays(todayISO(), 1)) throw httpErr(400, 'That date is in the future'); saveDay(dm[1], body); return send(res, 200, { ok: true }); }
    if (dm && M === 'DELETE') { tx(() => ['entries', 'symptom_logs', 'days'].forEach(t => db.prepare(`DELETE FROM ${t} WHERE date=?`).run(dm[1]))); syncProfileWeight(); return send(res, 200, { ok: true }); }
    if (M === 'GET' && route === 'match') return send(res, 200, match(q.q || ''));
    if (M === 'GET' && route === 'alerts') return send(res, 200, alertsFor(clamp(Math.floor(+q.days) || 10, 1, 365)));
    if (M === 'GET' && route === 'analyze') return send(res, 200, analyze(q));
    if (M === 'GET' && route === 'organs') return send(res, 200, organsFor(q));
    if (M === 'POST' && route === 'demo') { if (db.prepare('SELECT 1 FROM days LIMIT 1').get() && body.replace !== true) throw httpErr(409, 'Demo data would replace your logged days. Send {"replace": true} to confirm.'); return send(res, 200, { days: demo() }); }
    if (M === 'DELETE' && route === 'data') { if (q.confirm !== 'yes') throw httpErr(400, 'Add ?confirm=yes to delete all data'); clearAll(); return send(res, 200, { ok: true }); }
    const dg = route.match(/^(kidney|eye)\/(status|tests|diagnose)$/);              // kidney + eye diagnosis (spaceflight mode only)
    if (dg) {
      const cfg = DIAG[dg[1]]; requireSpace(cfg);
      if (M === 'GET' && dg[2] === 'status') return send(res, 200, { model_ready: fs.existsSync(cfg.model), max_images: DIAG_COMMON.maxImages, max_mb: DIAG_COMMON.maxImageBytes / 1048576 });
      if (M === 'GET' && dg[2] === 'tests') return send(res, 200, diagTests(cfg, q.month));
      if (M === 'POST' && dg[2] === 'diagnose') return send(res, 200, await diagnose(cfg, body));
    }
    send(res, 404, { error: 'Not found' });
  } catch (e) { send(res, e.status || 500, { error: e.message }); }
});

if (require.main === module) {
  if (process.argv.includes('--demo') && !db.prepare('SELECT 1 FROM days LIMIT 1').get()) console.log(`Loaded ${demo()} demo days`);
  server.listen(PORT, HOST, () => console.log(`Crew Health running at http://${HOST === '127.0.0.1' ? 'localhost' : HOST}:${PORT}  (database: ${DB_FILE})`));
}
module.exports = {
  server,
  db,
  analyze,
  alertsFor,
  match,
  demo,
  refFor,
  organsFor,
  organDay,
  ORGANS,
  adequacy,
  CAPS,
  ENV,
  envPenalty,
  exerciseNormal,
  exerciseOf,
  FMAP,
  spaceStart,
  splitSpace
};
