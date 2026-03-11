// ══════════════════════════════════════════════════════
//  THREE-TIER BIOMETRIC SECURITY SYSTEM — Frontend Logic
// ══════════════════════════════════════════════════════

let currentPage = 'dashboard';
let pollInterval = null;

// ── Navigation ──────────────────────────────────────
function showPage(page) {
    currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`[data-page="${page}"]`).classList.add('active');
    if (page === 'manage') loadUsers();
    if (page === 'dashboard') loadDashboardStats();
}

// ── Dashboard ───────────────────────────────────────
async function loadDashboardStats() {
    try {
        const res = await fetch('/api/users');
        const data = await res.json();
        document.getElementById('stat-users').textContent = data.count;
    } catch (e) { console.error(e); }
}

// ── Enrollment ──────────────────────────────────────
async function startEnrollment() {
    const username = document.getElementById('enroll-username').value.trim();
    if (!username) { alert('Please enter a username!'); return; }

    const res = await fetch('/api/enroll', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
    });
    const data = await res.json();
    if (!data.success) { alert(data.error); return; }

    document.getElementById('enroll-form').style.display = 'none';
    document.getElementById('enroll-operation').classList.add('active');
    document.getElementById('enroll-result').classList.remove('active', 'granted', 'denied');
    resetTierRows('enroll');
    startPolling('enroll');
}

// ── Authentication ──────────────────────────────────
async function startAuthentication() {
    const res = await fetch('/api/authenticate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json();
    if (!data.success) { alert(data.error); return; }

    document.getElementById('auth-start').style.display = 'none';
    document.getElementById('auth-operation').classList.add('active');
    document.getElementById('auth-result').classList.remove('active', 'granted', 'denied');
    resetTierRows('auth');
    startPolling('auth');
}

// ── Status Polling ──────────────────────────────────
function startPolling(mode) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => pollStatus(mode), 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

async function pollStatus(mode) {
    try {
        const res = await fetch('/api/status');
        const s = await res.json();

        // Update tier rows
        ['tier1', 'tier2', 'tier3'].forEach((tier, i) => {
            const row = document.getElementById(`${mode}-${tier}`);
            const detail = row.querySelector('.tier-row-detail');
            const badge = row.querySelector('.tier-row-status');

            row.className = 'tier-row';
            if (s.tier_status[tier]) {
                const ts = s.tier_status[tier];
                row.classList.add(ts.status);
                detail.textContent = ts.detail;
                badge.textContent = ts.status === 'running' ? 'Scanning...' :
                    ts.status === 'passed' ? 'Passed' : 'Failed';
            } else if (s.current_tier > i + 1) {
                row.classList.add('passed');
            } else if (s.current_tier === i + 1) {
                row.classList.add('running');
                badge.textContent = 'Scanning...';
            } else {
                row.classList.add('waiting');
                badge.textContent = 'Waiting';
            }
        });

        // Update message
        const msgEl = document.getElementById(`${mode}-message`);
        if (msgEl) msgEl.textContent = s.message;

        // Complete
        if (s.complete) {
            stopPolling();

            if (mode === 'enroll') {
                const result = document.getElementById('enroll-result');
                result.classList.add('active');
                if (s.result === 'success') {
                    result.classList.add('granted');
                    result.querySelector('.result-icon').textContent = '✅';
                    result.querySelector('.result-title').textContent = 'ENROLLMENT COMPLETE';
                    result.querySelector('.result-subtitle').textContent = `User "${s.username}" enrolled successfully!`;
                } else {
                    result.classList.add('denied');
                    result.querySelector('.result-icon').textContent = '❌';
                    result.querySelector('.result-title').textContent = 'ENROLLMENT FAILED';
                    result.querySelector('.result-subtitle').textContent = s.error || 'An error occurred.';
                }
            } else {
                const result = document.getElementById('auth-result');
                result.classList.add('active');
                if (s.result === 'granted') {
                    result.classList.add('granted');
                    result.querySelector('.result-icon').textContent = '🔓';
                    result.querySelector('.result-title').textContent = 'ACCESS GRANTED';
                    result.querySelector('.result-subtitle').textContent = `Identified as: ${s.username}`;
                } else {
                    result.classList.add('denied');
                    result.querySelector('.result-icon').textContent = '🔒';
                    result.querySelector('.result-title').textContent = 'ACCESS DENIED';
                    result.querySelector('.result-subtitle').textContent = s.error || 'Biometric verification failed.';
                }
            }

            // Reset server state after a delay
            setTimeout(async () => { await fetch('/api/reset', { method: 'POST' }); }, 2000);
        }
    } catch (e) { console.error(e); }
}

function resetTierRows(mode) {
    ['tier1', 'tier2', 'tier3'].forEach(tier => {
        const row = document.getElementById(`${mode}-${tier}`);
        row.className = 'tier-row waiting';
        row.querySelector('.tier-row-detail').textContent = 'Waiting...';
        row.querySelector('.tier-row-status').textContent = 'Waiting';
    });
}

function resetEnrollForm() {
    document.getElementById('enroll-form').style.display = 'block';
    document.getElementById('enroll-operation').classList.remove('active');
    document.getElementById('enroll-result').classList.remove('active', 'granted', 'denied');
    document.getElementById('enroll-username').value = '';
}

function resetAuthPanel() {
    document.getElementById('auth-start').style.display = 'block';
    document.getElementById('auth-operation').classList.remove('active');
    document.getElementById('auth-result').classList.remove('active', 'granted', 'denied');
}

// ── User Management ─────────────────────────────────
async function loadUsers() {
    const res = await fetch('/api/users');
    const data = await res.json();
    const list = document.getElementById('user-list');

    if (data.count === 0) {
        list.innerHTML = `<div class="empty-state"><div class="empty-icon">👤</div><p>No users enrolled yet.</p></div>`;
        return;
    }

    list.innerHTML = data.users.map(user => `
    <div class="user-item">
      <div class="user-info">
        <div class="user-avatar">👤</div>
        <span class="user-name">${user}</span>
      </div>
      <button class="btn btn-danger" onclick="deleteUser('${user}')">🗑 Delete</button>
    </div>
  `).join('');
}

async function deleteUser(username) {
    if (!confirm(`Delete user "${username}"? This cannot be undone.`)) return;
    const res = await fetch('/api/delete_user', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
    });
    const data = await res.json();
    if (data.success) loadUsers();
    else alert(data.error);
}

// ── Particle Background ─────────────────────────────
function initParticles() {
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];

    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            size: Math.random() * 2 + 0.5,
            opacity: Math.random() * 0.4 + 0.1,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 240, 255, ${p.opacity})`;
            ctx.fill();
        });

        // Connection lines
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${0.06 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
}

// ── Init ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    loadDashboardStats();
});
