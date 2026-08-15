// Read-only Qubic pool Stratum probe: log in, observe jobs, capture one live job. NO submits.
// Usage: node scripts/stratum_probe.mjs
// Token is read from deploy/appsettings.production.json (gitignored) and never printed.
import { readFileSync, writeFileSync } from 'node:fs';

const ENDPOINT = 'wss://wps.qubic.li/stratum';
const CFG = new URL('../deploy/appsettings.production.json', import.meta.url);
const FIXTURE = new URL('../scratchpad_stratum_job.json', import.meta.url);

function findToken(obj) {
  for (const k of Object.keys(obj)) {
    if (/accesstoken/i.test(k) && typeof obj[k] === 'string') return obj[k];
    if (obj[k] && typeof obj[k] === 'object') { const r = findToken(obj[k]); if (r) return r; }
  }
  return null;
}
const token = findToken(JSON.parse(readFileSync(CFG, 'utf8')));
if (!token) { console.error('no accessToken found'); process.exit(1); }

const redact = (s) => s.replace(token, '<TOKEN>');
console.log(`[probe] connecting to ${ENDPOINT} (worker m1-probe, observe-only)`);

const ws = new WebSocket(ENDPOINT);
let gotJob = 0, loginAt = Date.now();

ws.addEventListener('open', () => {
  console.log('[probe] socket open -> sending StratumLogin');
  ws.send(JSON.stringify({ Method: 'StratumLogin', AccessToken: token, Worker: 'm1-probe', Os: 'macOS', MessageVersion: 0 }));
});
ws.addEventListener('message', (ev) => {
  const raw = typeof ev.data === 'string' ? ev.data : ev.data.toString();
  let msg; try { msg = JSON.parse(raw); } catch { console.log('[recv non-json]', redact(raw).slice(0, 200)); return; }
  const type = msg.Method || `MessageType ${msg.MessageType}`;
  console.log(`[recv] ${type}: ${redact(JSON.stringify(msg))}`);
  const isJob = msg.Method === 'StratumJob' || msg.MessageType === 501;
  if (isJob && gotJob === 0) {
    writeFileSync(FIXTURE, JSON.stringify(msg, null, 2));
    console.log('[probe] captured first StratumJob -> scratchpad_stratum_job.json');
    // decode the interesting fields for immediate read
    const hexLen = (h) => (h ? h.length / 2 : 0) + 'B';
    console.log(`[job] Epoch=${msg.Epoch} RandomSeed=${hexLen(msg.RandomSeed)} PublicKey=${hexLen(msg.PublicKey)} Difficulty=${msg.Difficulty} DifficultyAddition=${msg.DifficultyAddition}`);
    const zero = (h) => h && /^0+$/.test(h);
    if (zero(msg.RandomSeed) && zero(msg.PublicKey)) console.log('[job] IDLE (all-zero seed/pubkey) — pool not serving work right now');
  }
  if (isJob) gotJob++;
  if (gotJob >= 2) { console.log('[probe] captured 2 jobs — done'); ws.close(); }
});
ws.addEventListener('close', (ev) => {
  const dt = ((Date.now() - loginAt) / 1000).toFixed(1);
  console.log(`[probe] socket closed after ${dt}s (code ${ev.code}). ${gotJob === 0 ? 'NO job received — if this was instant, login was likely rejected.' : `Received ${gotJob} job(s).`}`);
  process.exit(0);
});
ws.addEventListener('error', (ev) => console.log('[probe] ws error:', redact(String(ev.message || ev.type || ev))));

setTimeout(() => { console.log('[probe] 50s timeout — closing'); try { ws.close(); } catch {} setTimeout(() => process.exit(0), 500); }, 50000);
