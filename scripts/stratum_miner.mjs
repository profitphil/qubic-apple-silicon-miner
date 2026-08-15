// Qubic Stratum driver for the Metal miner.
//   node scripts/stratum_miner.mjs            # DRY-RUN: mine live jobs, log would-be shares, submit NOTHING
//   node scripts/stratum_miner.mjs --live     # submit shares — ONLY if the loaded task matches the live job
// Env: TASK=<path to real task .bin>  (default: example task — never matches live, so stays dry-run)
//      WORKER=<name>  COHORT=<n>
// Token is read from deploy/appsettings.production.json (gitignored) and never printed.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR   = path.dirname(fileURLToPath(import.meta.url));
const ROOT  = path.resolve(DIR, '..');
const METAL = path.join(ROOT, 'qiner-metal');
const HARNESS = path.join(METAL, 'build', 'harness');
const TASK  = process.env.TASK ? path.resolve(process.env.TASK)
                               : path.join(ROOT, 'qiner-macos', 'data', 'example_task_bpp9000.bin');
const ENDPOINT = 'wss://wps.qubic.li/stratum';
const WORKER = process.env.WORKER || 'm1-metal';
const LIVE = process.argv.includes('--live');

function findToken(o){for(const k of Object.keys(o)){if(/accesstoken/i.test(k)&&typeof o[k]==='string')return o[k];if(o[k]&&typeof o[k]==='object'){const r=findToken(o[k]);if(r)return r;}}return null;}
function loadToken(){
  if(process.env.QLI_TOKEN) return process.env.QLI_TOKEN.trim();
  for(const p of ['qli-config.json','deploy/appsettings.production.json']){
    try{ const t=findToken(JSON.parse(readFileSync(path.join(ROOT,p),'utf8'))); if(t) return t; }catch{}
  }
  console.error('No access token found. Get one at https://pool.qubic.li (control panel), then either:');
  console.error("  export QLI_TOKEN='<your JWT>'   (recommended)");
  console.error('  or create qli-config.json  {"accessToken":"<your JWT>"}');
  process.exit(1);
}
const token = loadToken();
const redact = s => (''+s).replace(token,'<TOKEN>');
const now = () => new Date().toISOString().slice(11,19);
const log = (...a) => console.log(now(), ...a);

const taskHash = createHash('sha256').update(readFileSync(TASK)).digest('hex').toLowerCase();
log(`task: ${path.basename(TASK)}  sha256=${taskHash.slice(0,16)}…`);
log(`mode: ${LIVE ? 'LIVE (will submit matching-task shares)' : 'DRY-RUN (no submits)'}  worker=${WORKER}`);

// --- spawn the Metal miner (stratum mode): we write jobs to its stdin, read shares from stdout ---
const miner = spawn(HARNESS, ['stratum'], { cwd: METAL, env: { ...process.env, TASK, COHORT: process.env.COHORT||'4' }, stdio: ['pipe','pipe','inherit'] });
miner.on('exit', c => { log(`[miner] exited (${c}) — shutting down`); process.exit(0); });

let ws = null, currentJob = null, taskMatches = false;
let lastJobKey = '', submitted = 0, accepted = 0, dryShares = 0, lastJobAt = Date.now();

// shares from the miner -> submit or dry-log
let outBuf = '';
miner.stdout.on('data', d => {
  outBuf += d; let i;
  while ((i = outBuf.indexOf('\n')) >= 0) {
    const line = outBuf.slice(0, i); outBuf = outBuf.slice(i + 1);
    if (!line.trim()) continue;
    let s; try { s = JSON.parse(line); } catch { continue; }
    if (s.type !== 'share') continue;
    onShare(s);
  }
});

function onShare(s) {
  const cur = currentJob;
  const stale = !cur || cur.PublicKey.toUpperCase() !== s.pubkey.toUpperCase();
  if (stale) { log(`[share] score=${s.score}<=${s.diff} but STALE (pubkey rotated) — skip`); return; }
  if (LIVE && taskMatches && ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ Method:'StratumSubmit', Epoch:cur.Epoch, RandomSeed:cur.RandomSeed, PublicKey:cur.PublicKey, Nonce:s.nonce, MessageVersion:0 }));
    submitted++;
    log(`[SUBMIT #${submitted}] score=${s.score}<=${s.diff} nonce=${s.nonce.slice(0,16)}…`);
  } else {
    dryShares++;
    const why = !taskMatches ? 'task≠livejob' : (!LIVE ? 'dry-run' : 'ws-down');
    log(`[would-submit #${dryShares}] score=${s.score}<=${s.diff} (${why}) nonce=${s.nonce.slice(0,16)}…`);
  }
}

function pushJob(j) {
  taskMatches = (j.Bpp9000TaskHash||'').toLowerCase() === taskHash;
  currentJob = j; lastJobAt = Date.now();
  const key = j.RandomSeed + j.PublicKey + j.DifficultyBpp9000;
  if (key === lastJobKey) return;   // dedupe identical pushes
  lastJobKey = key;
  const line = JSON.stringify({ seed:j.RandomSeed, pubkey:j.PublicKey, difficulty:j.DifficultyBpp9000, epoch:j.Epoch }) + '\n';
  try { miner.stdin.write(line); } catch (e) { log('[miner] stdin write failed', e.message); }
  log(`[job] epoch=${j.Epoch} diff=${j.DifficultyBpp9000} pubkey=${j.PublicKey.slice(0,12)}… taskMatch=${taskMatches}`);
}

function connect() {
  log(`[ws] connecting ${ENDPOINT}`);
  ws = new WebSocket(ENDPOINT);
  ws.addEventListener('open', () => { log('[ws] open -> login'); ws.send(JSON.stringify({ Method:'StratumLogin', AccessToken:token, Worker:WORKER, Os:'macOS', MessageVersion:0 })); });
  ws.addEventListener('message', ev => {
    let m; try { m = JSON.parse(typeof ev.data==='string'?ev.data:ev.data.toString()); } catch { return; }
    if (m.MessageType === 2) { log(`[ws] login ${m.Success?'OK':'FAIL'}: ${redact(m.Message||'')}`); return; }
    if (m.Method === 'StratumJob' || m.MessageType === 501) {
      const zero = h => h && /^0+$/.test(h);
      if (zero(m.RandomSeed) && zero(m.PublicKey)) { log('[job] IDLE (pool paused)'); return; }
      pushJob(m); return;
    }
    if (m.Method === 'StratumSubmit' || m.MessageType === 502) {
      if (m.Success) { accepted++; log(`[502] ACCEPTED (${accepted}/${submitted}) ${redact(m.Message||'')}`); }
      else log(`[502] rejected: ${redact(m.Message||'')}`);
      return;
    }
    log('[ws] other:', redact(JSON.stringify(m)).slice(0,200));
  });
  ws.addEventListener('close', e => { log(`[ws] closed (${e.code}) — reconnecting in 3s`); ws = null; setTimeout(connect, 3000); });
  ws.addEventListener('error', e => log('[ws] error', redact(String(e.message||e.type))));
}

// watchdog: if no job for 90s, force-reconnect (jobs push ~30s)
setInterval(() => { if (ws && Date.now() - lastJobAt > 90000) { log('[wd] no job 90s — reconnect'); try { ws.close(); } catch {} } }, 30000);

process.on('SIGINT', () => { log(`\n[done] submitted=${submitted} accepted=${accepted} dryShares=${dryShares}`); try{miner.kill('SIGINT');}catch{} setTimeout(()=>process.exit(0),500); });

connect();
