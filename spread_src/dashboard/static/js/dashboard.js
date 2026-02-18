const REFRESH_INTERVAL = 5000;
const MAX_GAME_EXPOSURE = 15; // matches trader config
let charts = {};
let activeGameId = null;

// ─── Data Fetching ───────────────────────────────────────────────────────────

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

// ─── Games ───────────────────────────────────────────────────────────────────

function updateGames(games) {
    const container = document.getElementById('games-container');

    if (!games || games.length === 0) {
        if (container.querySelector('.loading-state')) return;
        container.innerHTML = '<div class="loading-state"><p>No active games being tracked...</p></div>';
        return;
    }

    if (container.querySelector('.loading-state')) {
        container.innerHTML = '';
    }

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

// ─── Tabs (with exposure badges) ─────────────────────────────────────────────

function updateTabs(games) {
    const tabsContainer = document.getElementById('game-tabs');
    tabsContainer.innerHTML = '';

    games.forEach(game => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${game.game_id === activeGameId ? 'active' : ''}`;
        btn.onclick = () => switchTab(game.game_id);

        const gameExp = game.markets.reduce((sum, m) => sum + Math.abs(m.position_exp || 0), 0);
        const expBadge = gameExp > 0
            ? `<span class="tab-exposure">$${gameExp.toFixed(0)}</span>`
            : '';

        btn.innerHTML = `
            <span>${game.away_team} @ ${game.home_team}</span>
            <span class="tab-score">${game.away_score}-${game.home_score}</span>
            ${expBadge}
        `;
        tabsContainer.appendChild(btn);
    });
}

function switchTab(gameId) {
    activeGameId = gameId;
    updateDashboard();
}

// ─── Game Card Layout ────────────────────────────────────────────────────────

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
                <div class="game-money" id="money-${game.game_id}">
                    <div class="money-stat">
                        <span class="money-label">Game $</span>
                        <span class="money-value" id="game-exp-${game.game_id}">$0</span>
                    </div>
                    <div class="exposure-meter">
                        <div class="exposure-fill" id="exp-fill-${game.game_id}"></div>
                    </div>
                    <div class="money-stat">
                        <span class="money-count" id="pos-count-${game.game_id}">0</span>
                        <span class="money-label">pos</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="card-body">
            <div class="chart-section">
                <div class="chart-container">
                    <canvas id="chart-${game.game_id}"></canvas>
                </div>
                <div class="chart-legend">
                    <span class="legend-item"><span class="legend-swatch bid-swatch"></span>Bid</span>
                    <span class="legend-item"><span class="legend-swatch spread-swatch"></span>Spread</span>
                    <span class="legend-item"><span class="legend-swatch model-swatch"></span>Model Price</span>
                </div>
            </div>
            <div class="side-panel">
                <div class="feature-grid" id="features-${game.game_id}"></div>
                <div class="markets-title">Positions & Orders</div>
                <div class="market-strip" id="markets-${game.game_id}"></div>
            </div>
        </div>
    `;
    return card;
}

// ─── Update Game Card (scores, money, features, market pills) ────────────────

function updateGameCard(card, game) {
    document.getElementById(`score-${game.game_id}`).textContent =
        `${game.away_score} - ${game.home_score}`;

    const clockText = game.seconds_remaining === 0
        ? 'Final'
        : `Q${game.period} ${formatTime(game.seconds_remaining)}`;
    document.getElementById(`clock-${game.game_id}`).textContent = clockText;

    // ── Game money stats ──
    const posExp = game.markets.reduce((s, m) => s + Math.abs(m.position_exp || 0), 0);
    const pendExp = game.markets.reduce((s, m) =>
        s + (m.pending_buy_exp || 0) + (m.pending_sell_exp || 0), 0);
    const totalExp = posExp + pendExp;
    const posCount = game.markets.filter(m => m.position !== 0).length;
    const pendCount = game.markets.filter(m => m.pending_buy > 0 || m.pending_sell > 0).length;

    document.getElementById(`game-exp-${game.game_id}`).textContent = `$${totalExp.toFixed(2)}`;
    const countStr = pendCount > 0 ? `${posCount}+${pendCount}` : `${posCount}`;
    document.getElementById(`pos-count-${game.game_id}`).textContent = countStr;

    const pct = Math.min((totalExp / MAX_GAME_EXPOSURE) * 100, 100);
    const fillEl = document.getElementById(`exp-fill-${game.game_id}`);
    fillEl.style.width = `${pct}%`;
    fillEl.className = pct >= 80 ? 'exposure-fill hot' : pct >= 50 ? 'exposure-fill warm' : 'exposure-fill';

    // ── Features ──
    const fc = document.getElementById(`features-${game.game_id}`);
    if (game.driving_features && fc) {
        fc.innerHTML = '';
        const all = {
            ...(game.driving_features.live_state || game.driving_features.live || {}),
            ...(game.driving_features.team_context || game.driving_features.team_recent || {}),
            ...(game.driving_features.variance_drivers || game.driving_features.volatility || {}),
        };
        Object.entries(all).forEach(([label, value]) => {
            const d = document.createElement('div');
            d.className = 'feature-item';
            d.innerHTML = `<span class="feature-label">${label}</span><span class="feature-value">${value}</span>`;
            fc.appendChild(d);
        });
    }

    // ── Compact market pills ──
    const mc = document.getElementById(`markets-${game.game_id}`);
    mc.innerHTML = '';

    game.markets
        .filter(m => m.position !== 0 || m.pending_buy > 0 || m.pending_sell > 0)
        .sort((a, b) => b.spread - a.spread)
        .forEach(m => {
            const pill = document.createElement('div');
            pill.className = 'market-pill';

            const label = `${m.team} ${m.spread > 0 ? '+' : ''}${m.spread}`;
            let tags = '';

            if (m.position > 0)
                tags += `<span class="pill-tag long">LONG $${Math.abs(m.position_exp || 0).toFixed(2)}</span>`;
            else if (m.position < 0)
                tags += `<span class="pill-tag short">SHORT $${Math.abs(m.position_exp || 0).toFixed(2)}</span>`;

            if (m.pending_buy > 0)
                tags += `<span class="pill-tag pending-buy">BUY $${(m.pending_buy_exp || 0).toFixed(2)}</span>`;
            if (m.pending_sell > 0)
                tags += `<span class="pill-tag pending-sell">SELL $${(m.pending_sell_exp || 0).toFixed(2)}</span>`;

            pill.innerHTML = `<span class="pill-label">${label}</span>${tags}`;
            mc.appendChild(pill);
        });

    // If no positions, show a subtle message
    if (mc.children.length === 0) {
        mc.innerHTML = '<span class="no-positions">No active positions</span>';
    }
}

function formatTime(seconds) {
    const s = seconds % 60;
    const m = Math.floor(seconds / 60) % 12;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// ─── Chart: Mixed Bar + Line ─────────────────────────────────────────────────

function initChart(game) {
    const ctx = document.getElementById(`chart-${game.game_id}`).getContext('2d');

    charts[game.game_id] = new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [
                // 0: Bid bars (bottom of stack)
                {
                    label: 'Bid',
                    data: [],
                    backgroundColor: 'rgba(16, 185, 129, 0.30)',
                    borderColor: 'rgba(16, 185, 129, 0.55)',
                    borderWidth: 1,
                    borderRadius: 3,
                    stack: 'market',
                    barThickness: 18,
                    order: 3,
                    yAxisID: 'y',
                },
                // 1: Spread bars (stacked on bid)
                {
                    label: 'Spread',
                    data: [],
                    backgroundColor: 'rgba(245, 158, 11, 0.20)',
                    borderColor: 'rgba(245, 158, 11, 0.45)',
                    borderWidth: 1,
                    borderRadius: { topLeft: 3, topRight: 3 },
                    stack: 'market',
                    barThickness: 18,
                    order: 3,
                    yAxisID: 'y',
                },
                // 2: Model fair value line (connects the dots)
                {
                    type: 'line',
                    label: 'Model Price',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.06)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#3b82f6',
                    pointBorderWidth: 2.5,
                    borderWidth: 2.5,
                    order: 1,
                    yAxisID: 'y',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(3, 7, 18, 0.9)',
                    titleColor: '#f9fafb',
                    bodyColor: '#d1d5db',
                    padding: 10,
                    cornerRadius: 8,
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label(ctx) {
                            const ds = ctx.dataset.label;
                            if (ds === 'Bid') return `Bid: ${ctx.parsed.y}¢`;
                            if (ds === 'Spread') return `Spread: ${ctx.parsed.y}¢`;
                            if (ds === 'Model Price') return `Model: ${ctx.parsed.y.toFixed(1)}¢`;
                            return null;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Home Team Margin', color: '#6b7280', font: { size: 11 } },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#6b7280', font: { size: 10 } },
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Price (¢)', color: '#6b7280', font: { size: 11 } },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#6b7280', font: { size: 10 }, callback: v => v + '¢' },
                },

            },
        },
    });
}

// ─── Update Chart Data ───────────────────────────────────────────────────────

function updateChart(game) {
    const chart = charts[game.game_id];
    if (!chart) return;

    // — Build bar data with per-bar colors based on position status —
    const bidData = [];
    const spreadData = [];
    const fairData = [];
    const bidBg = [];
    const bidBorder = [];
    const spreadBg = [];
    const spreadBorder = [];
    const barBorderWidth = [];
    const barBorderDash = []; // not natively per-bar, but we handle via borderWidth

    game.markets.forEach(m => {
        const th = m.team === game.home_team ? m.spread : -m.spread;
        const bid = m.bid || 0;
        const ask = m.ask || 0;
        bidData.push({ x: th, y: bid });
        spreadData.push({ x: th, y: Math.max(ask - bid, 0) });
        if (m.fair_value != null) {
            fairData.push({ x: th, y: m.fair_value });
        }

        // Color based on position / pending status
        if (m.position > 0) {
            // Long position — bright green
            bidBg.push('rgba(16, 185, 129, 0.55)');
            bidBorder.push('rgba(16, 185, 129, 0.9)');
            spreadBg.push('rgba(16, 185, 129, 0.25)');
            spreadBorder.push('rgba(16, 185, 129, 0.7)');
            barBorderWidth.push(2.5);
        } else if (m.position < 0) {
            // Short position — red
            bidBg.push('rgba(239, 68, 68, 0.45)');
            bidBorder.push('rgba(239, 68, 68, 0.85)');
            spreadBg.push('rgba(239, 68, 68, 0.20)');
            spreadBorder.push('rgba(239, 68, 68, 0.6)');
            barBorderWidth.push(2.5);
        } else if (m.pending_buy > 0) {
            // Pending buy — lighter green, thinner
            bidBg.push('rgba(16, 185, 129, 0.15)');
            bidBorder.push('rgba(16, 185, 129, 0.6)');
            spreadBg.push('rgba(16, 185, 129, 0.08)');
            spreadBorder.push('rgba(16, 185, 129, 0.4)');
            barBorderWidth.push(1.5);
        } else if (m.pending_sell > 0) {
            // Pending sell — lighter red, thinner
            bidBg.push('rgba(239, 68, 68, 0.15)');
            bidBorder.push('rgba(239, 68, 68, 0.6)');
            spreadBg.push('rgba(239, 68, 68, 0.08)');
            spreadBorder.push('rgba(239, 68, 68, 0.4)');
            barBorderWidth.push(1.5);
        } else {
            // No position — neutral
            bidBg.push('rgba(255, 255, 255, 0.08)');
            bidBorder.push('rgba(255, 255, 255, 0.15)');
            spreadBg.push('rgba(245, 158, 11, 0.12)');
            spreadBorder.push('rgba(245, 158, 11, 0.3)');
            barBorderWidth.push(1);
        }
    });

    // Sort everything together by threshold
    const indices = bidData.map((_, i) => i);
    indices.sort((a, b) => bidData[a].x - bidData[b].x);

    const sorted = (arr) => indices.map(i => arr[i]);
    chart.data.datasets[0].data = sorted(bidData);
    chart.data.datasets[0].backgroundColor = sorted(bidBg);
    chart.data.datasets[0].borderColor = sorted(bidBorder);
    chart.data.datasets[0].borderWidth = sorted(barBorderWidth);
    chart.data.datasets[1].data = sorted(spreadData);
    chart.data.datasets[1].backgroundColor = sorted(spreadBg);
    chart.data.datasets[1].borderColor = sorted(spreadBorder);
    chart.data.datasets[1].borderWidth = sorted(barBorderWidth);

    fairData.sort((a, b) => a.x - b.x);
    chart.data.datasets[2].data = fairData;

    // — Only annotation: Live score margin line —
    const currentMargin = game.home_score - game.away_score;
    chart.options.plugins.annotation = {
        annotations: [{
            type: 'line',
            xMin: currentMargin,
            xMax: currentMargin,
            borderColor: 'rgba(255, 255, 255, 0.35)',
            borderWidth: 1.5,
            borderDash: [6, 4],
            label: {
                content: 'Live',
                display: true,
                position: 'start',
                color: '#9ca3af',
                backgroundColor: 'rgba(0,0,0,0.5)',
                font: { size: 9 },
                padding: 3,
            },
        }],
    };
    chart.update('none');
}

// ─── Start Polling ───────────────────────────────────────────────────────────
updateDashboard();
setInterval(updateDashboard, REFRESH_INTERVAL);
