// ── THEME TOGGLE ──────────────────────────────────────────────────────────
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
  document.querySelector('.theme-toggle').textContent = isDark ? '🌙' : '☀️';
}

// Apply saved theme on load
(function () {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  const btn = document.querySelector('.theme-toggle');
  if (btn) btn.textContent = saved === 'dark' ? '☀️' : '🌙';
})();

// ── NAV MENU TOGGLE ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  const toggle = document.getElementById('menuToggle');
  if (toggle) {
    toggle.onclick = function () {
      document.querySelector('.nav-links').classList.toggle('show');
    };
  }
});

// ── CHART COLOR GENERATOR ─────────────────────────────────────────────────
function generateColors(n) {
  const palette = [
    '#6c63ff','#a78bfa','#22c55e','#f59e0b','#ef4444',
    '#3b82f6','#ec4899','#14b8a6','#f97316','#8b5cf6',
    '#06b6d4','#84cc16','#e11d48','#0ea5e9','#d946ef',
  ];
  const colors = [];
  for (let i = 0; i < n; i++) {
    colors.push(palette[i % palette.length]);
  }
  return colors;
}

// ── AUTO-DISMISS ALERTS ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(alert => {
    setTimeout(() => {
      alert.style.transition = 'opacity 0.5s';
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 500);
    }, 4000);
  });
});