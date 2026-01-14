const REFRESH_INTERVAL = 5000;
let charts = {};
let activeGameId = null;

async function updateDashboard() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) throw new Error('Failed to fetch state');
        const state = await response.json();

        updatePortfolio(state.portfolio);
        updateGames(state.games);

        document.getElementById('status-badge').textContent = 'Live';
        document.getElementById('status-badge').className = 'badge';
    } catch (err) {
        console.error(err);
        document.getElementById('status-badge').textContent = 'Disconnected';
        document.getElementById('status-badge').className = 'badge disconnected';
    }
}

function updatePortfolio(portfolio) {
    if (!portfolio) return;

    document.getElementById('stat-exposure').textContent = `$${portfolio.exposure.toFixed(2)}`;
    document.getElementById('stat-cash').textContent = `$${portfolio.cash.toFixed(2)}`;
}

function updateGames(games) {
    const container = document.getElementById('games-container');

    if (!games || games.length === 0) {
        if (container.querySelector('.loading-state')) return;
        container.innerHTML = '<div class="loading-state"><p>No active games being tracked...</p></div>';
        return;
    }

    // Remove loading state if it exists
    if (container.querySelector('.loading-state')) {
        container.innerHTML = '';
    }

    // Set initial active game if none
    if (!activeGameId && games.length > 0) {
        activeGameId = games[0].game_id;
    }

    updateTabs(games);

    games.forEach(game => {
        let gameEl = document.getElementById(`game-${game.game_id}`);
        if (!gameEl) {
            gameEl = createGameCard(game);
            container.appendChild(gameEl);
            initChart(game);
        }

        // Handle visibility
        if (game.game_id === activeGameId) {
            gameEl.classList.remove('hidden');
        } else {
            gameEl.classList.add('hidden');
        }

        updateGameCard(gameEl, game);
        if (game.game_id === activeGameId) {
            updateChart(game);
        }
    });
}

function updateTabs(games) {
    const tabsContainer = document.getElementById('game-tabs');
    tabsContainer.innerHTML = '';

    games.forEach(game => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${game.game_id === activeGameId ? 'active' : ''}`;
        btn.onclick = () => switchTab(game.game_id);

        btn.innerHTML = `
            <span>${game.away_team} @ ${game.home_team}</span>
            <span class="tab-score">${game.away_score}-${game.home_score}</span>
        `;
        tabsContainer.appendChild(btn);
    });
}

function switchTab(gameId) {
    activeGameId = gameId;
    updateDashboard(); // Fast refresh to update visibility
}

function createGameCard(game) {
    const card = document.createElement('div');
    card.id = `game-${game.game_id}`;
    card.className = 'game-card';

    card.innerHTML = `
        <div class="game-header">
            <div class="teams-info">
                <div class="team">
                    <span class="tricode">${game.away_team}</span>
                </div>
                <div class="score-container">
                    <span class="score" id="score-${game.game_id}">${game.away_score} - ${game.home_score}</span>
                </div>
                <div class="team">
                    <span class="tricode">${game.home_team}</span>
                </div>
            </div>
            <div class="game-meta">
                <span class="clock" id="clock-${game.game_id}">Q${game.period} ${formatTime(game.seconds_remaining)}</span>
            </div>
        </div>
        <div class="card-body">
            <div class="chart-container">
                <canvas id="chart-${game.game_id}"></canvas>
            </div>
            <div class="features-and-markets">
                <div class="feature-grid" id="features-${game.game_id}">
                    <!-- Features injected here -->
                </div>
                <div class="markets-title">Spread Markets</div>
                <div class="markets-list" id="markets-${game.game_id}">
                    <!-- Markets injected here -->
                </div>
            </div>
        </div>
    `;
    return card;
}

function updateGameCard(card, game) {
    document.getElementById(`score-${game.game_id}`).textContent = `${game.away_score} - ${game.home_score}`;

    const clockText = game.seconds_remaining === 0 ? 'Final' : `Q${game.period} ${formatTime(game.seconds_remaining)}`;
    document.getElementById(`clock-${game.game_id}`).textContent = clockText;

    // Update Features
    const featuresContainer = document.getElementById(`features-${game.game_id}`);
    if (game.driving_features && featuresContainer) {
        featuresContainer.innerHTML = '';
        const allItems = {
            ...game.driving_features.live,
            ...game.driving_features.team_recent,
            ...game.driving_features.volatility
        };

        Object.entries(allItems).forEach(([label, value]) => {
            const item = document.createElement('div');
            item.className = 'feature-item';
            item.innerHTML = `
                <span class="feature-label">${label}</span>
                <span class="feature-value">${value}</span>
            `;
            featuresContainer.appendChild(item);
        });
    }

    const marketsContainer = document.getElementById(`markets-${game.game_id}`);
    marketsContainer.innerHTML = '';

    game.markets.sort((a, b) => b.spread - a.spread).forEach(m => {
        const row = document.createElement('div');
        row.className = 'market-row';

        const posClass = m.position > 0 ? 'pos-long' : (m.position < 0 ? 'pos-short' : '');
        const posText = m.position !== 0 ? `${m.position > 0 ? 'Long' : 'Short'} $${Math.abs(m.position_exp || 0).toFixed(2)}` : '';

        const pendingBuyText = m.pending_buy > 0 ? `Buy $${(m.pending_buy_exp || 0).toFixed(2)}` : '';
        const pendingSellText = m.pending_sell > 0 ? `Sell $${(m.pending_sell_exp || 0).toFixed(2)}` : '';

        row.innerHTML = `
            <div class="market-info">
                <span class="market-spread">${m.team} ${m.spread > 0 ? '+' : ''}${m.spread}</span>
                <span class="market-fair">Model: ${(m.fair_value || 0).toFixed(1)}¢</span>
                <div class="status-tags">
                    ${posText ? `<span class="position-tag ${posClass}">${posText}</span>` : ''}
                    ${pendingBuyText ? `<span class="position-tag pending pos-long">${pendingBuyText}</span>` : ''}
                    ${pendingSellText ? `<span class="position-tag pending pos-short">${pendingSellText}</span>` : ''}
                </div>
            </div>
            <div class="market-prices">
                <div class="price-box">
                    <span class="p-label">Bid</span>
                    <span class="p-value">${m.bid || '-'}</span>
                </div>
                <div class="price-box">
                    <span class="p-label">Ask</span>
                    <span class="p-value">${m.ask || '-'}</span>
                </div>
            </div>
        `;
        marketsContainer.appendChild(row);
    });
}

function formatTime(seconds) {
    const s = seconds % 60;
    const m = Math.floor(seconds / 60) % 12;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function initChart(game) {
    const ctx = document.getElementById(`chart-${game.game_id}`).getContext('2d');

    charts[game.game_id] = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Predicted Final Score Distribution',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Home Team Margin', color: '#9ca3af' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    display: false,
                    grid: { display: false }
                }
            }
        }
    });
}

function updateChart(game) {
    const chart = charts[game.game_id];
    if (!chart) return;

    const { loc, scale, df } = game.distribution;

    // Generate data points for Student-T
    const min = loc - 4 * scale;
    const max = loc + 4 * scale;
    const points = [];
    const step = (max - min) / 100;

    for (let x = min; x <= max; x += step) {
        // jStat.studentt.pdf(x, df, loc, scale)
        // Note: jStat's studentt.pdf takes (x, df) and is centered at 0, scale 1.
        // We need to shift and scale ourselves.
        const z = (x - loc) / scale;
        const pdf = jStat.studentt.pdf(z, df) / scale;
        points.push({ x, y: pdf });
    }

    chart.data.datasets[0].data = points;

    // Update annotations (vertical lines for market spreads and current score)
    const annotations = [];

    // Current Score line
    const currentMargin = game.home_score - game.away_score;
    const maxPdf = Math.max(...points.map(p => p.y));

    annotations.push({
        type: 'line',
        xMin: currentMargin,
        xMax: currentMargin,
        borderColor: 'rgba(255, 255, 255, 0.5)',
        borderWidth: 1,
        borderDash: [5, 5],
        label: { content: 'Live', display: true, position: 'start' }
    });

    // Market spreads
    game.markets.forEach((m, idx) => {
        // Threshold is spread for home team, -spread for away team
        const threshold = m.team === game.home_team ? m.spread : -m.spread;

        // Scale height based on exposure ($3.00 = max height usually)
        const scaleHeight = (exp, isPending = false) => {
            const ratio = Math.min(Math.abs(exp) / 3.0, 1.2); // Cap at 120% height
            return ratio * maxPdf;
        };

        // Position line (Solid)
        if (m.position !== 0) {
            const h = scaleHeight(m.position_exp);
            annotations.push({
                type: 'line',
                xMin: threshold,
                xMax: threshold,
                yMin: 0,
                yMax: h,
                borderColor: m.position > 0 ? '#10b981' : '#ef4444',
                borderWidth: 4, // Thicker for positions
                label: {
                    content: `$${Math.abs(m.position_exp).toFixed(2)}`,
                    display: true,
                    position: 'end',
                    backgroundColor: m.position > 0 ? '#10b981' : '#ef4444',
                    font: { size: 10, weight: 'bold' }
                }
            });
        }

        // Order line (Dashed)
        if (m.pending_buy > 0 || m.pending_sell > 0) {
            const side = m.pending_buy > 0 ? 'buy' : 'sell';
            const exp = side === 'buy' ? m.pending_buy_exp : m.pending_sell_exp;
            const color = side === 'buy' ? '#10b981' : '#ef4444';
            const h = scaleHeight(exp, true);

            annotations.push({
                type: 'line',
                xMin: threshold,
                xMax: threshold,
                yMin: 0,
                yMax: h,
                borderColor: color,
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                    content: `$${Math.abs(exp).toFixed(2)}`,
                    display: true,
                    position: 'center',
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    color: color,
                    font: { size: 9, weight: 'bold' }
                }
            });
        }

        // Default gray line if neither
        if (m.position === 0 && m.pending_buy === 0 && m.pending_sell === 0) {
            annotations.push({
                type: 'line',
                xMin: threshold,
                xMax: threshold,
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
            });
        }
    });

    // Chart.js annotations plugin requirement:
    chart.options.plugins.annotation = { annotations };

    chart.update('none');
}

// Start polling
updateDashboard();
setInterval(updateDashboard, REFRESH_INTERVAL);
