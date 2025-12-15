// Global variables
let searchFilter = '';
let csvData = [];
let prizepicksData = [];
let underdogData = [];
let activeTab = 'prizepicks'; // 'prizepicks' or 'underdog'

// CSV parsing function with proper handling of quoted fields
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];
    
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
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length !== headers.length) continue;
        
        const row = {};
        headers.forEach((header, idx) => {
            const key = header.trim();
            let value = values[idx].trim();
            
            // Remove quotes if present
            if (value.startsWith('"') && value.endsWith('"')) {
                value = value.slice(1, -1);
            }
            
            // Parse numeric values
            if (key === 'LINE' || key === 'PREDICTION' || key === 'STD') {
                value = parseFloat(value) || 0;
            } else if (key === 'MODEL_PROB' || key === 'FAIR_PROB' || key === 'PRIZEPICKS_IMPLIED' || 
                       key === 'EDGE_VS_FAIR' || key === 'EDGE_VS_PP' || key === 'EV' || 
                       key === 'EV_PERCENT' || key === 'KELLY_QUARTER' || key === 'USAGE_ADJ' ||
                       key === 'L-5' || key === 'L-10' || key === 'L-15' ||
                       key === 'L5' || key === 'L10' || key === 'L15') {
                value = parseFloat(value) || 0;
            } else if (key === 'PRIZEPICKS_ODDS' || key === 'BEST_US_ODDS') {
                value = parseFloat(value) || -137;
            }
            
            row[key] = value;
        });
        data.push(row);
    }
    
    return data;
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
        fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text();
            })
            .then(csvText => {
                const data = parseCSV(csvText);
                // Sort by EV_PERCENT descending
                data.sort((a, b) => (b.EV_PERCENT || 0) - (a.EV_PERCENT || 0));
                
                // Store in appropriate variable
                if (platform === 'prizepicks') {
                    prizepicksData = data;
                } else {
                    underdogData = data;
                }
                
                // Update active data and render
                updateActiveData();
                render();
            })
            .catch(error => {
                console.error(`Error loading CSV from ${path}:`, error);
                pathIndex++;
                if (pathIndex < platformPaths.length) {
                    tryLoad(platformPaths[pathIndex]);
                } else {
                    console.error(`Failed to load CSV from all paths for ${platform}`);
                    // Set empty data for this platform
                    if (platform === 'prizepicks') {
                        prizepicksData = [];
                    } else {
                        underdogData = [];
                    }
                    updateActiveData();
                    render();
                }
            });
    }
    
    tryLoad(platformPaths[0]);
}

// Update active data based on current tab
function updateActiveData() {
    if (activeTab === 'prizepicks') {
        csvData = prizepicksData;
    } else {
        csvData = underdogData;
    }
}

// Switch between tabs
function switchTab(platform) {
    activeTab = platform;
    
    // Update tab button styles
    const prizepicksTab = document.getElementById('prizepicksTab');
    const underdogTab = document.getElementById('underdogTab');
    
    if (prizepicksTab && underdogTab) {
        if (platform === 'prizepicks') {
            prizepicksTab.classList.add('active');
            underdogTab.classList.remove('active');
        } else {
            underdogTab.classList.add('active');
            prizepicksTab.classList.remove('active');
        }
    }
    
    // Load data if not already loaded
    if (platform === 'prizepicks' && prizepicksData.length === 0) {
        loadCSVData('prizepicks');
    } else if (platform === 'underdog' && underdogData.length === 0) {
        loadCSVData('underdog');
    } else {
        // Data already loaded, just switch
        updateActiveData();
        render();
    }
}

// Utility functions
function getEVClass(ev) {
    if (ev >= 10) return 'ev-high';
    if (ev >= 7) return 'ev-medium';
    return 'ev-low';
}

function formatOdds(odds) {
    return odds > 0 ? `+${odds}` : odds;
}

// Render singles table from CSV data
function renderSinglesTable(data) {
    const thead = `
        <tr>
            <th style="width: 10%">Player</th>
            <th style="width: 8%">Side</th>
            <th style="width: 8%">Line</th>
            <th style="width: 8%">Projection</th>
            <th style="width: 8%">Model Prob</th>
            <th style="width: 8%">EV%</th>
            <th style="width: 8%">Kelly</th>
            <th style="width: 8%">L-5</th>
            <th style="width: 8%">L-10</th>
            <th style="width: 8%">L-15</th>
        </tr>
    `;

    const tbody = data.map((row, index) => {
        const evPercent = row.EV_PERCENT || 0;
        const kellyQuarter = row.KELLY_QUARTER || 0;
        // L5, L10, L15 are already in percentage format (60.0 = 60%)
        const l5 = row.L5 !== undefined && row.L5 !== null ? parseFloat(row.L5) : null;
        const l10 = row.L10 !== undefined && row.L10 !== null ? parseFloat(row.L10) : null;
        const l15 = row.L15 !== undefined && row.L15 !== null ? parseFloat(row.L15) : null;
        
        // Format L values - they're already percentages, just add % sign
        const formatLValue = (val) => {
            if (val === null || val === undefined || isNaN(val)) return 'N/A';
            return val.toFixed(0) + '%';
        };
        
        // Color coding based on percentage (already in 0-100 format)
        const getLColor = (val) => {
            if (val === null || val === undefined || isNaN(val)) return '#a0a0a0';
            if (val >= 70) return '#34d399';
            if (val >= 50) return '#fbbf24';
            return '#f87171';
        };
        
        return `
        <tr>
            <td class="player-name">${row.NAME || ''}</td>
            <td>
                <span class="side-badge side-${(row.SIDE || '').toLowerCase()}">${row.SIDE || ''}</span>
            </td>
            <td class="line-value">${row.LINE || ''}</td>
            <td style="color: #9ca3af; font-weight: 600;">${(row.PREDICTION || 0).toFixed(1)}</td>
            <td style="font-weight: 600; color: #667eea; font-size: 0.9rem;">
                ${((row.MODEL_PROB || 0) * 100).toFixed(1)}%
            </td>
            <td class="ev-cell ${getEVClass(evPercent)}">${evPercent.toFixed(2)}%</td>
            <td class="kelly-cell">${(kellyQuarter * 100).toFixed(1)}%</td>
            <td style="color: ${getLColor(l5)}; font-weight: 600;">
                ${formatLValue(l5)}
            </td>
            <td style="color: ${getLColor(l10)}; font-weight: 600;">
                ${formatLValue(l10)}
            </td>
            <td style="color: ${getLColor(l15)}; font-weight: 600;">
                ${formatLValue(l15)}
            </td>
        </tr>
    `}).join('');

    document.getElementById('tableHead').innerHTML = thead;
    document.getElementById('tableBody').innerHTML = tbody;
}

// Get data
function getData() {
    return csvData;
}

// Main render function
function render() {
    let data = getData();
    
    // Apply search filter
    if (searchFilter) {
        data = data.filter(row => 
            (row.NAME || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
            (row.TEAM || '').toLowerCase().includes(searchFilter.toLowerCase())
        );
    }
    
    // Render table
    renderSinglesTable(data);
    
    // Update stats
    updateStats(data);
    
    // Update picks count
    const picksCountEl = document.getElementById('picksCount');
    if (picksCountEl) {
        const platformName = activeTab === 'prizepicks' ? 'PrizePicks' : 'Underdog';
        picksCountEl.textContent = `Showing ${data.length} single props from ${platformName} EV Analysis`;
    }
}

// Update stats display
function updateStats(data) {
    const statsHTML = `
        <div class="stat-card">
            <div class="stat-label">Total Props</div>
            <div class="stat-value">${data.length}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Model Prob</div>
            <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Probability the model assigns to the recommended side</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">EV%</div>
            <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Expected value as a percentage of your stake</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Kelly</div>
            <div class="stat-value" style="font-size: 0.85rem; line-height: 1.3;">Optimal bet sizing percentage (quarter Kelly)</div>
        </div>
    `;
    
    const statsContainer = document.getElementById('statsContainer');
    if (statsContainer) {
        statsContainer.innerHTML = statsHTML;
    }
}

// Update last updated timestamp
function updateLastUpdated() {
    const metaTag = document.querySelector('meta[name="last-updated"]');
    let timestamp;
    
    if (metaTag && metaTag.content && metaTag.content !== 'BUILD_TIMESTAMP') {
        timestamp = new Date(metaTag.content);
    } else {
        timestamp = new Date();
    }
    
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    };
    const formattedDate = timestamp.toLocaleString('en-US', options);
    const timeElement = document.getElementById('lastUpdatedTime');
    if (timeElement) {
        timeElement.textContent = formattedDate;
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Player search
    const playerSearch = document.getElementById('playerSearch');
    if (playerSearch) {
        playerSearch.addEventListener('input', function() {
            searchFilter = this.value;
            render();
        });
    }
    
    // Initial load
    updateLastUpdated();
    loadCSVData('prizepicks');
    loadCSVData('underdog'); // Load both datasets
    render();
});
