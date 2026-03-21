// Global variables
let searchFilter = '';
let propFilter = ''; // empty = all props
let csvData = [];
let prizepicksData = [];
let underdogData = [];
let prizepicksHeaders = [];
let underdogHeaders = [];
let activeTab = 'prizepicks'; // 'prizepicks' or 'underdog'

// CSV parsing function with proper handling of quoted fields
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return { headers: [], data: [] };

    function parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        return result;
    }

    const headers = parseCSVLine(lines[0]);
    const data = [];

    // Columns that should be parsed as floats
    const numericCols = new Set([
        'Line',
        'Odds Over', 'Odds Under',
        'Implied Over', 'Implied Under',
        'EV Over', 'EV Under',
        // Backward compatible: older exports used Pts naming
        'Avg Pts L5', 'Std Pts L5',
        // New all-props exports use generic stat naming
        'Avg Stat L5', 'Std Stat L5',
        'Z Score',
        'Prob Over', 'Prob Under',
        'OVER L5', 'OVER L10', 'OVER L15',
        'Avg Min L5', 'Std Min L5',
        'Avg USG% L5', 'Std USG% L5',
        'Avg Stat vs Matchup', 'Matchup Games',
        'Spread', 'Total',
        'Opp Def Rating', 'Opp Def Rank',
        'Opp Pace', 'Opp Pace Rank'
    ]);

    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length !== headers.length) continue;

        const row = {};
        headers.forEach((header, idx) => {
            const key = header.trim();
            let value = values[idx].trim();

            // Strip surrounding quotes
            if (value.startsWith('"') && value.endsWith('"')) {
                value = value.slice(1, -1);
            }

            if (numericCols.has(key)) {
                const num = parseFloat(value);
                row[key] = value === '' || !isFinite(num) ? null : num;
            } else {
                row[key] = value;
            }
        });
        data.push(row);
    }

    return { headers, data };
}

// Load CSV data for a specific platform
function loadCSVData(platform = 'prizepicks') {
    const paths = {
        'prizepicks': [
            'data/props/ev_analysis/prizepicks.csv',
            '../data/props/ev_analysis/prizepicks.csv',
            './data/props/ev_analysis/prizepicks.csv'
        ],
        'underdog': [
            'data/props/ev_analysis/underdog.csv',
            '../data/props/ev_analysis/underdog.csv',
            './data/props/ev_analysis/underdog.csv'
        ]
    };

    const platformPaths = paths[platform] || paths['prizepicks'];
    let pathIndex = 0;

    function tryLoad(path) {
        fetch(path, { cache: 'no-store' })
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.text();
            })
            .then(csvText => {
                const parsed = parseCSV(csvText);
                const data = parsed.data;
                const headers = parsed.headers;
                // Sort by best EV (max of EV Over / EV Under) descending
                data.sort((a, b) => {
                    const bestA = Math.max(a['EV Over'] || 0, a['EV Under'] || 0);
                    const bestB = Math.max(b['EV Over'] || 0, b['EV Under'] || 0);
                    return bestB - bestA;
                });

                if (platform === 'prizepicks') {
                    prizepicksData = data;
                    prizepicksHeaders = headers;
                } else {
                    underdogData = data;
                    underdogHeaders = headers;
                }

                updateActiveData();
                syncPropFilterOptions();
                render();
            })
            .catch(error => {
                console.error(`Error loading CSV from ${path}:`, error);
                pathIndex++;
                if (pathIndex < platformPaths.length) {
                    tryLoad(platformPaths[pathIndex]);
                } else {
                    console.error(`Failed to load CSV from all paths for ${platform}`);
                    if (platform === 'prizepicks') {
                        prizepicksData = [];
                        prizepicksHeaders = [];
                    } else {
                        underdogData = [];
                        underdogHeaders = [];
                    }
                    updateActiveData();
                    syncPropFilterOptions();
                    render();
                }
            });
    }

    tryLoad(platformPaths[0]);
}

// Update active data based on current tab
function updateActiveData() {
    csvData = activeTab === 'prizepicks' ? prizepicksData : underdogData;
}

// Switch between tabs
function switchTab(platform) {
    activeTab = platform;

    const prizepicksTab = document.getElementById('prizepicksTab');
    const underdogTab   = document.getElementById('underdogTab');

    if (prizepicksTab && underdogTab) {
        prizepicksTab.classList.toggle('active', platform === 'prizepicks');
        underdogTab.classList.toggle('active',   platform === 'underdog');
    }

    // Re-render immediately so the UI always matches the selected tab,
    // even if the CSV is still loading (it may be empty temporarily).
    updateActiveData();
    syncPropFilterOptions();
    render();

    if (platform === 'prizepicks' && prizepicksData.length === 0) {
        loadCSVData('prizepicks');
    } else if (platform === 'underdog' && underdogData.length === 0) {
        loadCSVData('underdog');
    }
}

// ─── Prop filter (dropdown options from active platform data) ───────────────

function uniquePropsFromRows(rows) {
    const seen = new Set();
    for (const row of rows) {
        const p = row['Prop'];
        if (p == null) continue;
        const s = String(p).trim();
        if (s) seen.add(s);
    }
    return Array.from(seen).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
}

function syncPropFilterOptions() {
    const sel = document.getElementById('propFilter');
    if (!sel) return;

    const rows = activeTab === 'prizepicks' ? prizepicksData : underdogData;
    const props = uniquePropsFromRows(rows);
    const prev = propFilter;

    sel.replaceChildren();
    const allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All props';
    sel.appendChild(allOpt);
    for (const p of props) {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        sel.appendChild(opt);
    }

    if (prev && props.includes(prev)) {
        sel.value = prev;
    } else {
        sel.value = '';
        if (prev && !props.includes(prev)) propFilter = '';
    }
}

// ─── Utility helpers ────────────────────────────────────────────────────────

function getEVClass(ev) {
    if (ev >= 10) return 'ev-high';
    if (ev >= 7)  return 'ev-medium';
    return 'ev-low';
}

function formatOdds(odds) {
    if (!odds && odds !== 0) return '—';
    return odds > 0 ? `+${odds}` : `${odds}`;
}

// Color a hit-rate percentage (0-100)
function getHitRateColor(val) {
    const pct = normalizeHitRate(val);
    if (pct === null) return '#a0a0a0';
    if (pct >= 70) return '#34d399';
    if (pct >= 50) return '#fbbf24';
    return '#f87171';
}

function formatHitRate(val) {
    const pct = normalizeHitRate(val);
    if (pct === null) return 'N/A';
    return pct.toFixed(0) + '%';
}

// ─── Table renderer ──────────────────────────────────────────────────────────

// The CSV switched to `OVER L{5,10,15}` being stored as 0-1 probabilities.
// For backward compatibility (or any future export), accept either 0-1 or 0-100.
function normalizeHitRate(val) {
    if (val === null || val === undefined) return null;
    if (val === '') return null;
    const num = Number(val);
    if (!isFinite(num)) return null;

    // Treat 0-1 values as probabilities (e.g. 0.8 => 80%).
    if (num >= 0 && num <= 1.5) return num * 100;

    // Treat larger values as already-percent (e.g. 80 => 80%).
    return num;
}

function escapeHtml(str) {
    return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function formatCell(header, value) {
    if (value === null || value === undefined || value === '') return '—';

    // EV/prob formatting for readability (rest stays "as-is")
    if (header === 'EV Over' || header === 'EV Under') {
        return `${value.toFixed(2)}%`;
    }
    if (header === 'Prob Over' || header === 'Prob Under') {
        // CSV stores Prob Over/Under as 0-1
        return `${(value * 100).toFixed(1)}%`;
    }
    if (header === 'Avg USG% L5' || header === 'Std USG% L5') {
        // Stored as fraction (e.g. 0.33 => 33%)
        return `${(value * 100).toFixed(1)}%`;
    }
    if (header.startsWith('OVER L')) {
        return `${normalizeHitRate(value).toFixed(0)}%`;
    }
    if (header === 'Odds Over' || header === 'Odds Under' || header === 'Implied Over' || header === 'Implied Under') {
        // Odds are +/-; implied are floats but formatting as "raw" is fine.
        if (header === 'Odds Over' || header === 'Odds Under') return formatOdds(value);
        return value.toFixed(3);
    }

    // Keep numeric columns compact
    if (typeof value === 'number') {
        // Show integers without decimals
        if (Math.abs(value - Math.round(value)) < 1e-9) return String(value);
        return value.toFixed(2);
    }

    return escapeHtml(value);
}

function renderSinglesTable(data) {
    const headers = (activeTab === 'prizepicks' ? prizepicksHeaders : underdogHeaders) || [];

    // If headers didn't parse for some reason, fall back to keys from first row.
    const finalHeaders = headers.length ? headers : Object.keys(data[0] || {});

    const thead = `
        <tr>
            ${finalHeaders.map(h => `<th>${escapeHtml(h)}</th>`).join('')}
        </tr>
    `;

    const tbody = data.map(row => {
        const evOver = row['EV Over'] ?? null;
        const evUnder = row['EV Under'] ?? null;

        return `
        <tr>
            ${finalHeaders.map(header => {
                const raw = row[header];
                const formatted = formatCell(header, raw);

                // Preserve coloring for key columns.
                if (header === 'EV Over') {
                    const ev = evOver ?? 0;
                    return `<td class="ev-cell ${getEVClass(ev)}">${formatted}</td>`;
                }
                if (header === 'EV Under') {
                    const ev = evUnder ?? 0;
                    return `<td class="ev-cell ${getEVClass(ev)}">${formatted}</td>`;
                }
                if (header.startsWith('OVER L')) {
                    return `<td style="color:${getHitRateColor(raw)}; font-weight:600;">${formatted}</td>`;
                }

                if (header === 'Player') return `<td class="player-name">${formatted}</td>`;
                if (header === 'Opponent') return `<td style="color:#9ca3af;">${formatted}</td>`;
                if (header === 'Line') return `<td class="line-value">${formatted}</td>`;

                return `<td>${formatted}</td>`;
            }).join('')}
        </tr>
        `;
    }).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

// ─── Stats banner ────────────────────────────────────────────────────────────

function updateStats(data) {
    const statsHTML = `
        <div class="stat-card">
            <div class="stat-label">Total Props</div>
            <div class="stat-value">${data.length}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">EV Over / EV Under</div>
            <div class="stat-value" style="font-size:0.85rem; line-height:1.3;">
                Expected value (%) for each side of the line
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Prob O/U</div>
            <div class="stat-value" style="font-size:0.85rem; line-height:1.3;">
                Model's probability for Over (top) and Under (bottom)
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">L-5 / L-10 / L-15</div>
            <div class="stat-value" style="font-size:0.85rem; line-height:1.3;">
                Hit-rate going <strong>over</strong> the line in last 5 / 10 / 15 games
            </div>
        </div>
    `;

    const statsContainer = document.getElementById('statsContainer');
    if (statsContainer) statsContainer.innerHTML = statsHTML;
}

// ─── Core render ─────────────────────────────────────────────────────────────

function render() {
    let data = csvData.slice();

    if (searchFilter) {
        const q = searchFilter.toLowerCase();
        data = data.filter(row =>
            (row['Player']   || '').toLowerCase().includes(q) ||
            (row['Opponent'] || '').toLowerCase().includes(q)
        );
    }

    if (propFilter) {
        data = data.filter(row => String(row['Prop'] || '').trim() === propFilter);
    }

    renderSinglesTable(data);
    updateStats(data);

    const picksCountEl = document.getElementById('picksCount');
    if (picksCountEl) {
        const platformName = activeTab === 'prizepicks' ? 'PrizePicks' : 'Underdog';
        picksCountEl.textContent = `Showing ${data.length} single props from ${platformName} EV Analysis`;
    }
}

// ─── Timestamp ───────────────────────────────────────────────────────────────

function updateLastUpdated() {
    const metaTag = document.querySelector('meta[name="last-updated"]');
    const timestamp = (metaTag && metaTag.content && metaTag.content !== 'BUILD_TIMESTAMP')
        ? new Date(metaTag.content)
        : new Date();

    const options = {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
        hour: 'numeric', minute: '2-digit', hour12: true
    };

    const timeElement = document.getElementById('lastUpdatedTime');
    if (timeElement) timeElement.textContent = timestamp.toLocaleString('en-US', options);
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    const playerSearch = document.getElementById('playerSearch');
    if (playerSearch) {
        playerSearch.addEventListener('input', function () {
            searchFilter = this.value;
            render();
        });
    }

    const propFilterEl = document.getElementById('propFilter');
    if (propFilterEl) {
        propFilterEl.addEventListener('change', function () {
            propFilter = this.value;
            render();
        });
    }

    updateLastUpdated();
    loadCSVData('prizepicks');
    loadCSVData('underdog');
    render();
});