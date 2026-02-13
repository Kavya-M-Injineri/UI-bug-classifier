/**
 * UI Bug AI — Application Logic
 * SPA Router, File Upload, Chart Rendering, API Integration
 * JWT Authentication integrated
 */

// ═══════════════════════════════════════════════════════════════════════
// Auth Helpers
// ═══════════════════════════════════════════════════════════════════════
function getToken() {
    return localStorage.getItem('token');
}

function getUser() {
    try {
        return JSON.parse(localStorage.getItem('user'));
    } catch { return null; }
}

function authHeaders() {
    const token = getToken();
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function authFetch(url, options = {}) {
    const headers = { ...authHeaders(), ...(options.headers || {}) };
    const res = await fetch(url, { ...options, headers });

    if (res.status === 401) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login';
        return null;
    }
    return res;
}

function checkAuth() {
    if (!getToken()) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login';
}

function initUserDisplay() {
    const user = getUser();
    if (user) {
        const avatarEl = document.getElementById('userAvatar');
        if (avatarEl) {
            avatarEl.textContent = user.name ? user.name.charAt(0).toUpperCase() : 'U';
            avatarEl.title = `${user.name}\nClick to logout`;
            avatarEl.style.cursor = 'pointer';
            avatarEl.addEventListener('click', () => {
                if (confirm('Logout?')) logout();
            });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SPA Router
// ═══════════════════════════════════════════════════════════════════════
const pages = ['dashboard', 'upload', 'reports', 'metrics', 'settings'];

function navigateTo(page) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => {
        p.classList.remove('page--active');
    });

    // Show target page
    const target = document.getElementById(`page-${page}`);
    if (target) {
        target.classList.add('page--active');
    }

    // Update sidebar links
    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.page === page) {
            link.classList.add('active');
        }
    });

    // Load page data
    if (page === 'dashboard') loadDashboard();
    if (page === 'reports') loadReports();
    if (page === 'metrics') loadMetrics();
}

// Hash-based routing
function handleRoute() {
    const hash = window.location.hash.replace('#', '') || 'dashboard';
    navigateTo(hash);
}

window.addEventListener('hashchange', handleRoute);
window.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    initUserDisplay();
    handleRoute();
    initUpload();
    initSettings();
    initRetrainButton();
});

// ═══════════════════════════════════════════════════════════════════════
// Dashboard
// ═══════════════════════════════════════════════════════════════════════
let bugDistChart = null;
let accuracyChart = null;

async function loadDashboard() {
    try {
        const res = await authFetch('/api/dashboard');
        if (!res) return;
        const data = await res.json();

        // Update stats
        document.getElementById('statTotalProcessed').textContent =
            data.total_processed.toLocaleString();
        document.getElementById('statBestAccuracy').textContent =
            data.best_accuracy + '%';

        // Calculate time ago
        const lastTraining = new Date(data.last_training);
        const hoursAgo = Math.round((Date.now() - lastTraining) / 3600000);
        document.getElementById('statLastTraining').textContent =
            hoursAgo < 1 ? 'Just now' : `${hoursAgo}h ago`;

        renderBugDistChart(data.bug_distribution);
        renderAccuracyChart(data.epoch_accuracies);
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

function renderBugDistChart(distribution) {
    const ctx = document.getElementById('bugDistChart');
    if (!ctx) return;

    if (bugDistChart) bugDistChart.destroy();

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);
    const colors = ['#c4b5fd', '#86efac', '#93c5fd', '#fed7aa', '#fda4af'];

    bugDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 3,
                borderColor: '#ffffff',
                hoverBorderWidth: 0,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        font: { family: 'Inter', size: 12, weight: '500' },
                        color: '#525252'
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}

function renderAccuracyChart(accuracies) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;

    if (accuracyChart) accuracyChart.destroy();

    const labels = accuracies.map((_, i) => `Epoch ${i + 1}`);

    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: accuracies,
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.08)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#8b5cf6',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 7,
                borderWidth: 2.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 40,
                    max: 100,
                    grid: { color: 'rgba(0,0,0,0.04)' },
                    ticks: {
                        font: { family: 'Inter', size: 11 },
                        color: '#a3a3a3',
                        callback: val => val + '%'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { family: 'Inter', size: 10 },
                        color: '#a3a3a3',
                        maxRotation: 45
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e1b4b',
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'Inter' },
                    padding: 10,
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => `Accuracy: ${ctx.parsed.y}%`
                    }
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// File Upload
// ═══════════════════════════════════════════════════════════════════════
function initUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');

    if (!uploadArea || !fileInput) return;

    // Browse button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // Click on upload area
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    // Upload another button
    if (uploadAnotherBtn) {
        uploadAnotherBtn.addEventListener('click', resetUpload);
    }
}

async function handleFileUpload(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, WebP)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be under 10MB');
        return;
    }

    // Show loader
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('classifyLoader').style.display = 'block';
    document.getElementById('resultsView').style.display = 'none';

    // Send to API with JWT
    const formData = new FormData();
    formData.append('image', file);

    try {
        const res = await authFetch('/api/classify', {
            method: 'POST',
            body: formData
        });

        if (!res) return;
        const result = await res.json();

        if (result.error) {
            alert('Error: ' + result.error);
            resetUpload();
            return;
        }

        displayResults(result, file);
    } catch (err) {
        console.error('Upload error:', err);
        alert('Classification failed. Please check if the server is running.');
        resetUpload();
    }
}

function displayResults(result, file) {
    // Hide loader, show results
    document.getElementById('classifyLoader').style.display = 'none';
    document.getElementById('resultsView').style.display = 'block';

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Filename and timestamp
    document.getElementById('previewFilename').textContent = result.filename;
    const ts = new Date(result.timestamp);
    document.getElementById('previewTimestamp').textContent = ts.toLocaleString();

    // Bug type
    document.getElementById('resultBugType').textContent = result.predicted_class;

    // Priority badge
    const priorityEl = document.getElementById('resultPriority');
    priorityEl.textContent = result.priority;
    priorityEl.className = 'priority-badge';
    priorityEl.classList.add(`priority-badge--${result.priority.toLowerCase()}`);

    // Confidence Bar
    document.getElementById('resultConfidence').textContent = result.confidence + '%';
    const bar = document.getElementById('resultConfidenceBar');
    bar.style.width = '0%';

    // Animate confidence bar
    requestAnimationFrame(() => {
        setTimeout(() => {
            bar.style.width = result.confidence + '%';

            // Color based on confidence
            if (result.confidence >= 85) {
                bar.style.background = 'linear-gradient(135deg, #86efac 0%, #22c55e 100%)';
            } else if (result.confidence >= 60) {
                bar.style.background = 'linear-gradient(135deg, #c4b5fd 0%, #8b5cf6 100%)';
            } else {
                bar.style.background = 'linear-gradient(135deg, #fed7aa 0%, #f97316 100%)';
            }
        }, 100);
    });

    // Low confidence warning
    const warningEl = document.getElementById('lowConfidenceWarning');
    warningEl.style.display = result.low_confidence ? 'flex' : 'none';

    // Root causes
    const causesEl = document.getElementById('resultCauses');
    causesEl.innerHTML = result.root_causes.map(c =>
        `<li>${c}</li>`
    ).join('');

    // Recommended fixes
    const fixesEl = document.getElementById('resultFixes');
    fixesEl.innerHTML = result.recommended_fixes.map(f =>
        `<li>${f}</li>`
    ).join('');
}

function resetUpload() {
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('classifyLoader').style.display = 'none';
    document.getElementById('resultsView').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

// ═══════════════════════════════════════════════════════════════════════
// Reports
// ═══════════════════════════════════════════════════════════════════════
async function loadReports() {
    try {
        const res = await authFetch('/api/reports');
        if (!res) return;
        const data = await res.json();
        renderReports(data.reports);
    } catch (err) {
        console.error('Reports load error:', err);
    }
}

function renderReports(reports) {
    const tbody = document.getElementById('reportsBody');
    if (!tbody) return;

    tbody.innerHTML = reports.map(r => {
        const ts = new Date(r.timestamp);
        const timeStr = ts.toLocaleDateString() + ' ' + ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const priorityClass = `badge--${r.priority === 'High' ? 'pink' : r.priority === 'Medium' ? 'peach' : 'mint'}`;

        return `
            <tr>
                <td><code style="font-size:12px;color:#8b5cf6">${r.id}</code></td>
                <td>${r.filename}</td>
                <td><strong>${r.predicted_class}</strong></td>
                <td>${r.confidence}%</td>
                <td><span class="badge ${priorityClass}">${r.priority}</span></td>
                <td style="color:#9ca3af;font-size:13px">${timeStr}</td>
            </tr>
        `;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════════════
// Model Metrics
// ═══════════════════════════════════════════════════════════════════════
let metricsAccChart = null;
let metricsLossChart = null;

async function loadMetrics() {
    try {
        const res = await authFetch('/api/metrics');
        if (!res) return;
        const data = await res.json();
        renderMetricsCharts(data);
        renderArchitecture(data.model_architecture);
        renderConfig(data.training_config);
        renderEpochTrack(data);
    } catch (err) {
        console.error('Metrics load error:', err);
    }
}

function renderMetricsCharts(data) {
    // Accuracy chart
    const accCtx = document.getElementById('metricsAccuracyChart');
    if (accCtx) {
        if (metricsAccChart) metricsAccChart.destroy();
        const labels = data.epoch_accuracies.map((_, i) => `E${i + 1}`);
        metricsAccChart = new Chart(accCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Accuracy',
                    data: data.epoch_accuracies,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.08)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#22c55e',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 3,
                    borderWidth: 2.5
                }]
            },
            options: chartOptions('Accuracy (%)', val => val + '%')
        });
    }

    // Loss chart
    const lossCtx = document.getElementById('metricsLossChart');
    if (lossCtx) {
        if (metricsLossChart) metricsLossChart.destroy();
        const labels = data.epoch_losses.map((_, i) => `E${i + 1}`);
        metricsLossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Loss',
                    data: data.epoch_losses,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.08)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#f97316',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 3,
                    borderWidth: 2.5
                }]
            },
            options: chartOptions('Loss', val => val.toFixed(2))
        });
    }
}

function chartOptions(label, tickCallback) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                grid: { color: 'rgba(0,0,0,0.04)' },
                ticks: {
                    font: { family: 'Inter', size: 11 },
                    color: '#a3a3a3',
                    callback: tickCallback
                }
            },
            x: {
                grid: { display: false },
                ticks: {
                    font: { family: 'Inter', size: 10 },
                    color: '#a3a3a3'
                }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1e1b4b',
                cornerRadius: 8,
                padding: 10,
                titleFont: { family: 'Inter', weight: '600' },
                bodyFont: { family: 'Inter' }
            }
        },
        animation: { duration: 1200, easing: 'easeOutQuart' }
    };
}

function renderArchitecture(arch) {
    const container = document.getElementById('archLayers');
    if (!container || !arch) return;

    container.innerHTML = arch.layers.map(layer =>
        `<div class="arch-layer">${layer}</div>`
    ).join('');
}

function renderConfig(config) {
    const container = document.getElementById('configGrid');
    if (!container || !config) return;

    const items = [
        { label: 'Total Epochs', value: config.total_epochs },
        { label: 'Batch Size', value: config.batch_size },
        { label: 'Validation Split', value: (config.validation_split * 100) + '%' },
        { label: 'ES Patience', value: config.early_stopping_patience },
        { label: 'Best Weights', value: config.best_weights_restored ? 'Restored' : 'No' }
    ];

    container.innerHTML = items.map(item => `
        <div class="config-item">
            <span class="config-label">${item.label}</span>
            <span class="config-value">${item.value}</span>
        </div>
    `).join('');
}

function renderEpochTrack(data) {
    const container = document.getElementById('epochTrack');
    if (!container) return;

    const bestEpoch = data.epoch_accuracies.indexOf(Math.max(...data.epoch_accuracies)) + 1;
    const stopEpoch = data.early_stopping_epoch || 20;

    let html = '';
    for (let i = 1; i <= 20; i++) {
        let cls = 'epoch-dot epoch-dot--complete';
        if (i === bestEpoch) cls = 'epoch-dot epoch-dot--best';
        else if (i > stopEpoch) cls = 'epoch-dot epoch-dot--stopped';

        const acc = data.epoch_accuracies[i - 1];
        html += `<div class="${cls}" title="Epoch ${i}: ${acc}%">${i}</div>`;
    }
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════
// Settings
// ═══════════════════════════════════════════════════════════════════════
function initSettings() {
    const thresholdRange = document.getElementById('thresholdRange');
    const thresholdValue = document.getElementById('thresholdValue');

    if (thresholdRange && thresholdValue) {
        thresholdRange.addEventListener('input', () => {
            thresholdValue.textContent = thresholdRange.value + '%';
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Retrain Button
// ═══════════════════════════════════════════════════════════════════════
function initRetrainButton() {
    const btn = document.getElementById('retrainBtn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        const textEl = btn.querySelector('.btn-text');
        const loaderEl = btn.querySelector('.btn-loader');

        textEl.textContent = 'Retraining...';
        loaderEl.style.display = 'inline-block';
        btn.disabled = true;
        btn.style.opacity = '0.7';

        try {
            const res = await authFetch('/api/retrain', { method: 'POST' });
            if (!res) return;
            const data = await res.json();

            textEl.textContent = '✓ Retrain Triggered';
            loaderEl.style.display = 'none';

            setTimeout(() => {
                textEl.textContent = 'Start Retrain';
                btn.disabled = false;
                btn.style.opacity = '1';
            }, 3000);
        } catch (err) {
            textEl.textContent = 'Failed — Retry';
            loaderEl.style.display = 'none';
            btn.disabled = false;
            btn.style.opacity = '1';
        }
    });
}
