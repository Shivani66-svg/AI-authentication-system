/* ── Particle Canvas ──────────────────────────────────────── */
const canvas = document.getElementById('particles-canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

const particles = Array.from({ length: 60 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.5 + 0.3,
    dx: (Math.random() - 0.5) * 0.3,
    dy: (Math.random() - 0.5) * 0.3,
    color: ['#7c6ff7','#00d2ff','#00e676'][Math.floor(Math.random()*3)]
}));

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
        p.x += p.dx; p.y += p.dy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
    });
    // Draw connecting lines
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 120) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.strokeStyle = `rgba(124,111,247,${0.15 * (1 - dist/120)})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(animateParticles);
}
animateParticles();

/* ── Mobile Menu Toggle ───────────────────────────────────── */
const mobileToggle = document.getElementById('mobile-toggle');
const mobileMenu   = document.getElementById('mobile-menu');

function closeMobileMenu() {
    mobileMenu.classList.remove('open');
    mobileToggle.textContent = '☰';
}

mobileToggle.addEventListener('click', () => {
    const isOpen = mobileMenu.classList.toggle('open');
    mobileToggle.textContent = isOpen ? '✕' : '☰';
});

// Close menu when tapping outside
document.addEventListener('click', e => {
    if (!mobileMenu.contains(e.target) && !mobileToggle.contains(e.target)) {
        closeMobileMenu();
    }
});

/* ── Scroll Reveal ────────────────────────────────────────── */
const revealObserver = new IntersectionObserver(entries => {
    entries.forEach((e, i) => {
        if (e.isIntersecting) {
            const siblings = e.target.parentElement
                ? Array.from(e.target.parentElement.children).filter(c => c.classList.contains('reveal'))
                : [];
            const idx = siblings.indexOf(e.target);
            setTimeout(() => e.target.classList.add('in-view'), idx * 80);
            revealObserver.unobserve(e.target);
        }
    });
}, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ── Navbar scroll ────────────────────────────────────────── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    if (window.pageYOffset > 80) {
        navbar.style.padding = '8px 0';
        navbar.style.background = 'rgba(8,10,18,.97)';
    } else {
        navbar.style.padding = '14px 0';
        navbar.style.background = 'rgba(8,10,18,.85)';
    }
}, { passive: true });

/* ── Tab Switch ───────────────────────────────────────────── */
function switchTab(tab, btn) {
    document.querySelectorAll('.hiw-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.hiw-content').forEach(c => c.classList.add('hidden'));
    document.getElementById('tab-' + tab).classList.remove('hidden');
}

/* ── Copy code ────────────────────────────────────────────── */
function copyCode() {
    const code = document.getElementById('cta-code').textContent;
    navigator.clipboard.writeText(code).then(() => {
        const icon = document.getElementById('copy-icon');
        icon.textContent = '✓';
        setTimeout(() => icon.textContent = '⎘', 2000);
    });
}

/* ════════════════════════════════════════════════════════════
   INTERACTIVE DEMO ENGINE
   ════════════════════════════════════════════════════════════ */
let demoRunning = false;
let scanAnimId = null;
let scanOsc = 0;

// Demo data — simulates a successful biometric auth
const DEMO_STEPS = [
    { delay: 400,  action: 'log',     args: ['info',  '  Scanning enrolled users database...'] },
    { delay: 700,  action: 'log',     args: ['info',  '  Found 1 enrolled user: mayur'] },
    { delay: 1000, action: 'log',     args: ['dim',   '─────────────────────────────────────────'] },
    { delay: 1200, action: 'tier',    args: ['iris', 'active', 'Scanning...'] },
    { delay: 1200, action: 'event',   args: ['info',  'Iris scan initiated'] },
    { delay: 1400, action: 'log',     args: ['warn',  '  [LIVENESS] Blink detected — liveness confirmed  ✓'] },
    { delay: 1400, action: 'event',   args: ['pass',  'Liveness: PASSED'] },
    { delay: 2200, action: 'log',     args: ['info',  '  [TIER 1] Extracting iris geometry features...'] },
    { delay: 3200, action: 'log',     args: ['info',  '  [TIER 1] Matching against user: mayur...'] },
    { delay: 4000, action: 'score',   args: ['iris',  94.3, true] },
    { delay: 4000, action: 'log',     args: ['pass',  '  [TIER 1 ✓] Iris match: mayur (94.3%)'] },
    { delay: 4000, action: 'tier',    args: ['iris', 'passed', '94.3%'] },
    { delay: 4000, action: 'event',   args: ['pass',  'Tier 1 PASSED — Iris: 94.3%'] },
    { delay: 4400, action: 'log',     args: ['dim',   '─────────────────────────────────────────'] },
    { delay: 4600, action: 'tier',    args: ['voice', 'active', 'Recording...'] },
    { delay: 4600, action: 'event',   args: ['info',  'Voice recording started'] },
    { delay: 4800, action: 'log',     args: ['info',  '  [TIER 2] Recording passphrase...'] },
    { delay: 5800, action: 'log',     args: ['info',  '  [TIER 2] Extracting 13 MFCC coefficients...'] },
    { delay: 6600, action: 'log',     args: ['info',  '  [TIER 2] DTW distance from template...'] },
    { delay: 7400, action: 'score',   args: ['voice', 28.4, true] },
    { delay: 7400, action: 'log',     args: ['pass',  '  [TIER 2 ✓] Voice match: mayur (DTW: 28.4)'] },
    { delay: 7400, action: 'tier',    args: ['voice', 'passed', 'DTW 28.4'] },
    { delay: 7400, action: 'event',   args: ['pass',  'Tier 2 PASSED — Voice DTW: 28.4'] },
    { delay: 7800, action: 'log',     args: ['dim',   '─────────────────────────────────────────'] },
    { delay: 8000, action: 'tier',    args: ['gesture','active','Scanning...'] },
    { delay: 8000, action: 'event',   args: ['info',  'Gesture scan started'] },
    { delay: 8200, action: 'log',     args: ['info',  '  [TIER 3] Scanning hand gesture...'] },
    { delay: 9200, action: 'log',     args: ['info',  '  [TIER 3] Extracting 83 landmark features...'] },
    { delay: 10000,action: 'log',     args: ['info',  '  [TIER 3] Cosine similarity: 96.1%...'] },
    { delay: 10600,action: 'score',   args: ['gesture',96.1, true] },
    { delay: 10600,action: 'log',     args: ['pass',  '  [TIER 3 ✓] Gesture match: mayur (96.1%)'] },
    { delay: 10600,action: 'tier',    args: ['gesture','passed','96.1%'] },
    { delay: 10600,action: 'event',   args: ['pass',  'Tier 3 PASSED — Gesture: 96.1%'] },
    { delay: 11000,action: 'log',     args: ['dim',   '─────────────────────────────────────────'] },
    { delay: 11400,action: 'log',     args: ['info',  '  [FUSION] Verifying all tiers match same user...'] },
    { delay: 11900,action: 'log',     args: ['info',  '  iris=mayur  voice=mayur  gesture=mayur  ✓'] },
    { delay: 12400,action: 'log',     args: ['pass',  '██████████████████████████████████████████'] },
    { delay: 12500,action: 'log',     args: ['pass',  '  ACCESS GRANTED — Identity: mayur'] },
    { delay: 12600,action: 'log',     args: ['pass',  '██████████████████████████████████████████'] },
    { delay: 12600,action: 'verdict', args: ['success'] },
    { delay: 12600,action: 'event',   args: ['pass',  'ACCESS GRANTED ✓ — mayur authenticated'] },
    { delay: 12600,action: 'status',  args: ['success','Granted'] },
];

function addTermLine(cls, text) {
    const t = document.getElementById('terminal');
    const cursor = document.getElementById('cursor');
    const line = document.createElement('span');
    line.className = `term-line term-${cls}`;
    line.textContent = text;
    t.insertBefore(line, cursor);
    t.scrollTop = t.scrollHeight;
}

function addEvent(cls, text) {
    const list = document.getElementById('events-list');
    const item = document.createElement('div');
    item.className = `event-item event-${cls}`;
    item.textContent = text;
    list.appendChild(item);
    list.scrollTop = list.scrollHeight;
}

function setTierState(tier, state, label) {
    const el = document.getElementById('tier-' + tier);
    el.className = 'tier-item ' + state;
    document.getElementById(tier + '-state').textContent = label;
}

function animateScore(tier, value, pass) {
    const isVoice = tier === 'voice';
    // For voice, convert DTW distance to percentage (lower is better, max 45)
    const pct = isVoice ? Math.max(0, Math.min(100, (1 - value / 45) * 100)) : value;
    const bar = document.getElementById('bar-' + tier);
    const scoreEl = document.getElementById('score-' + tier);
    const block = document.getElementById('score-' + tier + '-block');

    requestAnimationFrame(() => {
        bar.style.width = pct + '%';
        bar.className = 'score-bar-fill' + (pass ? '' : ' low');
        scoreEl.textContent = isVoice ? `DTW: ${value}` : `${value}%`;
        scoreEl.className = 'score-value ' + (pass ? 'high' : 'low');
        block.className = 'score-tier ' + (pass ? 'passed-tier' : 'failed-tier');
    });
}

function setVerdict(result) {
    const box = document.getElementById('verdict-box');
    const icon = document.getElementById('verdict-icon');
    const text = document.getElementById('verdict-text');
    const user = document.getElementById('verdict-user');

    if (result === 'success') {
        box.className = 'verdict-box success';
        icon.textContent = '✅';
        text.textContent = 'ACCESS GRANTED';
        text.className = 'verdict-text success';
        user.textContent = 'User: mayur — Identity Confirmed';
    } else {
        box.className = 'verdict-box fail';
        icon.textContent = '❌';
        text.textContent = 'ACCESS DENIED';
        text.className = 'verdict-text fail';
        user.textContent = 'Authentication failed';
    }
}

function setStatus(state, label) {
    const el = document.getElementById('status-badge');
    el.textContent = label;
    el.className = 'sbar-status ' + state;
}

// Waveform animation on scan canvas
function startScanAnim(type) {
    const canvas = document.getElementById('scan-canvas');
    const c = canvas.getContext('2d');
    let frame = 0;

    cancelAnimationFrame(scanAnimId);
    function draw() {
        const w = canvas.width, h = canvas.height;
        c.clearRect(0, 0, w, h);

        if (type === 'iris') {
            // Scanning line animation
            c.strokeStyle = 'rgba(124,111,247,0.6)';
            c.lineWidth = 1.5;
            const cy = h / 2;
            const scanX = (frame * 3) % w;
            // Eye outline
            c.beginPath();
            c.ellipse(w/2, cy, 160, 50, 0, 0, Math.PI*2);
            c.strokeStyle = 'rgba(124,111,247,0.2)';
            c.stroke();
            // Iris
            c.beginPath();
            c.arc(w/2, cy, 28, 0, Math.PI*2);
            c.strokeStyle = 'rgba(124,111,247,0.8)';
            c.lineWidth = 2;
            c.stroke();
            // Scan line
            c.beginPath();
            c.moveTo(scanX, 10); c.lineTo(scanX, h-10);
            c.strokeStyle = `rgba(124,111,247,${0.6 + Math.sin(frame*0.1)*0.4})`;
            c.lineWidth = 1;
            c.stroke();
            // Grid lines
            for (let x = 0; x < w; x += 20) {
                c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h);
                c.strokeStyle = 'rgba(124,111,247,0.04)'; c.lineWidth = 1; c.stroke();
            }
        } else if (type === 'voice') {
            // Animated waveform
            c.beginPath();
            for (let x = 0; x < w; x++) {
                const amp = 30 + 20 * Math.sin(x * 0.05 + frame * 0.1);
                const y = h/2 + amp * Math.sin(x * 0.03 + frame * 0.08) * Math.sin(x * 0.07);
                x === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
            }
            c.strokeStyle = '#00d2ff';
            c.lineWidth = 2;
            c.shadowColor = '#00d2ff';
            c.shadowBlur = 8;
            c.stroke();
            c.shadowBlur = 0;
            // Frequency bars
            for (let i = 0; i < 32; i++) {
                const bh = 15 + 35 * Math.abs(Math.sin(i * 0.5 + frame * 0.15));
                const bx = i * 14 + 4;
                const grad = c.createLinearGradient(0, h, 0, h - bh);
                grad.addColorStop(0, 'rgba(0,210,255,0.6)');
                grad.addColorStop(1, 'rgba(0,230,118,0.3)');
                c.fillStyle = grad;
                c.fillRect(bx, h - bh, 10, bh);
            }
        } else {
            // Gesture — hand landmark dots
            const pts = [
                [230,100],[200,80],[180,60],[165,42],[150,30],
                [210,60],[195,35],[185,20],[175,12],
                [225,58],[215,30],[205,14],[198,5],
                [240,60],[235,30],[230,14],[226,5],
                [255,68],[260,44],[262,30],[263,20]
            ];
            const scale = 0.85 + Math.sin(frame*0.05)*0.05;
            const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
            connections.forEach(([a,b]) => {
                c.beginPath();
                c.moveTo(pts[a][0]*scale, pts[a][1]*scale);
                c.lineTo(pts[b][0]*scale, pts[b][1]*scale);
                c.strokeStyle = 'rgba(0,230,118,0.3)';
                c.lineWidth = 1.5; c.stroke();
            });
            pts.forEach(([px, py], i) => {
                c.beginPath();
                c.arc(px*scale, py*scale, i===0?6:3.5, 0, Math.PI*2);
                c.fillStyle = i===0 ? '#00e676' : 'rgba(0,230,118,0.8)';
                c.fill();
            });
        }
        frame++;
        scanAnimId = requestAnimationFrame(draw);
    }
    draw();
}

function stopScanAnim() {
    cancelAnimationFrame(scanAnimId);
    const canvas = document.getElementById('scan-canvas');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

async function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

async function startDemo() {
    if (demoRunning) return;
    demoRunning = true;

    const btn = document.getElementById('start-btn');
    const btnIcon = document.getElementById('btn-icon');
    const btnText = document.getElementById('btn-text');
    btn.disabled = true;
    btnIcon.textContent = '⟳';
    btnText.textContent = 'Authenticating...';

    // Clear terminal
    const t = document.getElementById('terminal');
    const cursor = document.getElementById('cursor');
    t.innerHTML = '';
    t.appendChild(cursor);

    // Clear events
    document.getElementById('events-list').innerHTML = '';

    // Reset scores
    ['iris','voice','gesture'].forEach(tier => {
        document.getElementById('bar-' + tier).style.width = '0%';
        document.getElementById('score-' + tier).textContent = '—';
        document.getElementById('score-' + tier).className = 'score-value';
        document.getElementById('score-' + tier + '-block').className = 'score-tier';
        setTierState(tier, '', '—');
    });
    document.getElementById('verdict-box').className = 'verdict-box';
    document.getElementById('verdict-icon').textContent = '🔒';
    document.getElementById('verdict-text').textContent = 'Awaiting scan...';
    document.getElementById('verdict-text').className = 'verdict-text';
    document.getElementById('verdict-user').textContent = '';
    setStatus('active', 'Scanning');

    addTermLine('info', 'SECURITIX v2.0 — Three-Tier Biometric Auth');
    addTermLine('dim',  '─────────────────────────────────────────');
    addTermLine('info', '  Auth started: ' + new Date().toLocaleTimeString());

    // Determine what scan anim to show based on active tier
    const scanTypes = { iris: 'iris', voice: 'voice', gesture: 'gesture' };
    let currentScan = null;

    const timers = [];
    DEMO_STEPS.forEach(step => {
        timers.push(setTimeout(() => {
            if (!demoRunning) return;
            const [a] = [step.action];
            if (a === 'log')     addTermLine(step.args[0], step.args[1]);
            if (a === 'event')   addEvent(step.args[0], step.args[1]);
            if (a === 'tier') {
                setTierState(step.args[0], step.args[1], step.args[2]);
                if (step.args[1] === 'active' && step.args[0] !== currentScan) {
                    currentScan = step.args[0];
                    startScanAnim(currentScan);
                }
                if (step.args[1] === 'passed' || step.args[1] === 'failed') {
                    stopScanAnim();
                    currentScan = null;
                }
            }
            if (a === 'score')   animateScore(step.args[0], step.args[1], step.args[2]);
            if (a === 'verdict') setVerdict(step.args[0]);
            if (a === 'status')  setStatus(step.args[0], step.args[1]);
        }, step.delay));
    });

    // After demo ends
    setTimeout(() => {
        demoRunning = false;
        btn.disabled = false;
        btnIcon.textContent = '↺';
        btnText.textContent = 'Run Again';
    }, 13500);
}

function resetDemo() {
    demoRunning = false;
    stopScanAnim();
    cancelAnimationFrame(scanAnimId);

    const t = document.getElementById('terminal');
    const cursor = document.getElementById('cursor');
    t.innerHTML = '';
    t.appendChild(cursor);
    addTermLine('info', 'SECURITIX v2.0 — Three-Tier Biometric Auth');
    addTermLine('dim',  '─────────────────────────────────────────');
    addTermLine('muted','Ready. Press "Start Demo Authentication" to begin.');

    ['iris','voice','gesture'].forEach(tier => {
        document.getElementById('bar-' + tier).style.width = '0%';
        document.getElementById('score-' + tier).textContent = '—';
        document.getElementById('score-' + tier).className = 'score-value';
        document.getElementById('score-' + tier + '-block').className = 'score-tier';
        setTierState(tier, '', '—');
    });

    document.getElementById('verdict-box').className = 'verdict-box';
    document.getElementById('verdict-icon').textContent = '🔒';
    document.getElementById('verdict-text').textContent = 'Awaiting scan...';
    document.getElementById('verdict-text').className = 'verdict-text';
    document.getElementById('verdict-user').textContent = '';
    document.getElementById('events-list').innerHTML = '<div class="event-item event-info">System initialized</div>';
    setStatus('', 'Idle');

    const btn = document.getElementById('start-btn');
    btn.disabled = false;
    document.getElementById('btn-icon').textContent = '▶';
    document.getElementById('btn-text').textContent = 'Start Demo Authentication';
}
