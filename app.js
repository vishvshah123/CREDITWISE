const API = 'https://creditwise-api.onrender.com';

// ---- State ----
let lastFormData = null;
let baseProb = null;
let shapChartInstance = null;
let wiShapChartInstance = null;
let wiDebounce = null;

// ---- Tab Navigation ----
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
        e.preventDefault();
        const tab = item.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active', 'hidden') || t.classList.add('hidden'));
        item.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.remove('hidden');
        document.getElementById(`tab-${tab}`).classList.add('active');

        const titles = {
            assess: ['Loan Risk Assessment', 'Fill in the applicant profile to generate a full credit risk analysis.'],
            whatif: ['What-If Simulator', 'Adjust sliders to see real-time impact on the approval score.']
        };
        document.getElementById('page-title').textContent = titles[tab][0];
        document.getElementById('page-subtitle').textContent = titles[tab][1];
    });
});

// ---- Main Form Submit ----
document.getElementById('loanForm').addEventListener('submit', async e => {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');

    btn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnSpinner.classList.remove('hidden');

    lastFormData = collectFormData();

    try {
        const predRes = await fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(lastFormData)
        });

        if (!predRes.ok) {
            const err = await predRes.json().catch(() => ({}));
            throw new Error(err.detail || `API Error ${predRes.status}`);
        }

        const predData = await predRes.json();

        // Fetch counterfactual separately so its failure doesn't block the main result
        let cfData = null;
        try {
            const cfRes = await fetch(`${API}/counterfactual`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastFormData)
            });
            if (cfRes.ok) cfData = await cfRes.json();
        } catch (cfErr) { console.warn('Counterfactual skipped:', cfErr); }

        baseProb = predData.probability;
        renderResults(predData, cfData);
        loadWhatIfSimulator(lastFormData);

        // Update live badge
        document.getElementById('live-score-badge').classList.remove('hidden');
        document.getElementById('live-score-value').textContent = `${(predData.probability * 100).toFixed(0)}%`;
        document.getElementById('live-score-value').style.color = predData.prediction === 'Approved' ? 'var(--primary)' : 'var(--danger)';

    } catch (err) {
        alert(`Connection failed: ${err.message}\n\nThe Render server may be waking up. Please wait 30 seconds and try again.`);
        console.error(err);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Run Assessment';
        btnSpinner.classList.add('hidden');
    }
});

function collectFormData() {
    return {
        Married: document.getElementById('Married').value,
        Dependents: document.getElementById('Dependents').value,
        Education: document.getElementById('Education').value,
        Self_Employed: document.getElementById('Self_Employed').value,
        ApplicantIncome: parseFloat(document.getElementById('ApplicantIncome').value),
        CoapplicantIncome: parseFloat(document.getElementById('CoapplicantIncome').value),
        LoanAmount: parseFloat(document.getElementById('LoanAmount').value),
        Loan_Amount_Term: parseFloat(document.getElementById('Loan_Amount_Term').value),
        Credit_History: parseFloat(document.getElementById('Credit_History').value),
        Property_Area: document.getElementById('Property_Area').value,
        Existing_Debt: parseFloat(document.getElementById('Existing_Debt').value),
        Credit_Utilization: parseFloat(document.getElementById('Credit_Utilization').value),
        Employment_Stability: parseFloat(document.getElementById('Employment_Stability').value)
    };
}

// ---- Render Results ----
function renderResults(data, cfData) {
    document.getElementById('initialState').classList.add('hidden');
    const content = document.getElementById('resultsContent');
    content.classList.remove('hidden');
    content.classList.add('fade-in');

    const approved = data.prediction === 'Approved';

    // Decision Banner
    const banner = document.getElementById('decisionBanner');
    banner.className = `decision-banner ${approved ? '' : 'rejected'}`;
    document.getElementById('decisionIcon').textContent = approved ? '✓' : '✗';
    document.getElementById('decisionLabel').textContent = data.prediction;
    document.getElementById('decisionSub').textContent = `Approval Probability: ${(data.probability * 100).toFixed(1)}%`;

    // Score Ring animation
    const pct = data.probability;
    const circumference = 163.4;
    const offset = circumference - (pct * circumference);
    const arc = document.getElementById('scoreArc');
    arc.style.transition = 'stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)';
    arc.style.strokeDashoffset = offset;
    document.getElementById('scorePercent').textContent = `${(pct * 100).toFixed(0)}%`;

    // Stats
    document.getElementById('statRisk').textContent = data.risk_score.toFixed(3);
    document.getElementById('statThreshold').textContent = data.threshold_used.toFixed(3);
    const gap = data.threshold_used - data.probability;
    const gapEl = document.getElementById('statGap');
    gapEl.textContent = gap > 0 ? `+${(gap * 100).toFixed(1)}%` : 'Within Range';
    gapEl.style.color = gap > 0 ? 'var(--danger)' : 'var(--primary)';

    // SHAP Waterfall Chart — fall back to top_factors if backend is old version
    const factors = (data.explanation && data.explanation.all_factors) ||
                    (data.explanation && data.explanation.top_factors) || [];
    renderSHAPChart(factors, 'shapChart');

    // Counterfactual
    if (cfData && !approved && cfData.recommendations && cfData.recommendations.length > 0) {
        const cfCard = document.getElementById('cfCard');
        cfCard.classList.remove('hidden');
        const cfList = document.getElementById('cfList');
        cfList.innerHTML = '';
        cfData.recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'cf-item';
            item.innerHTML = `
                <div>
                    <div class="cf-action">${rec.action}</div>
                    <div class="cf-detail">${rec.detail}</div>
                </div>
                <div class="cf-target">→ ${rec.target}</div>
            `;
            cfList.appendChild(item);
        });
    }
}

// ---- SHAP Waterfall Chart ----
function renderSHAPChart(factors, canvasId) {
    const top = factors.slice(0, 8);
    const labels = top.map(f => formatFeature(f.feature));
    const values = top.map(f => parseFloat(f.impact.toFixed(4)));
    const colors = values.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.85)' : 'rgba(248, 81, 73, 0.85)');
    const borderColors = values.map(v => v >= 0 ? '#3fb950' : '#f85149');

    const ctx = document.getElementById(canvasId).getContext('2d');

    if (canvasId === 'shapChart' && shapChartInstance) shapChartInstance.destroy();
    if (canvasId === 'wiShapChart' && wiShapChartInstance) wiShapChartInstance.destroy();

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `SHAP Impact: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(4)}`
                    },
                    backgroundColor: '#1c2128',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    titleColor: '#e6edf3',
                    bodyColor: '#7d8590'
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#7d8590', font: { size: 11 } },
                    border: { color: 'rgba(255,255,255,0.08)' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e6edf3', font: { size: 11 } },
                    border: { color: 'rgba(255,255,255,0.08)' }
                }
            }
        }
    });

    if (canvasId === 'shapChart') shapChartInstance = chart;
    else wiShapChartInstance = chart;
}

// ---- What-If Simulator ----
function loadWhatIfSimulator(formData) {
    document.getElementById('whatif-not-ready').classList.add('hidden');
    document.getElementById('whatif-not-ready-2').classList.add('hidden');
    document.getElementById('whatif-controls').classList.remove('hidden');
    document.getElementById('whatif-score-panel').classList.remove('hidden');

    const sliders = [
        { id: 'wi-income', valId: 'wi-income-val', field: 'ApplicantIncome', fmt: v => `$${v.toLocaleString()}` },
        { id: 'wi-coincome', valId: 'wi-coincome-val', field: 'CoapplicantIncome', fmt: v => `$${v.toLocaleString()}` },
        { id: 'wi-loan', valId: 'wi-loan-val', field: 'LoanAmount', fmt: v => `$${v}K` },
        { id: 'wi-debt', valId: 'wi-debt-val', field: 'Existing_Debt', fmt: v => `$${v.toLocaleString()}` },
        { id: 'wi-util', valId: 'wi-util-val', field: 'Credit_Utilization', fmt: v => `${v}%` },
        { id: 'wi-emp', valId: 'wi-emp-val', field: 'Employment_Stability', fmt: v => `${v} yrs` }
    ];

    sliders.forEach(({ id, valId, field, fmt }) => {
        const slider = document.getElementById(id);
        slider.value = formData[field];
        document.getElementById(valId).textContent = fmt(formData[field]);
        slider.oninput = () => {
            document.getElementById(valId).textContent = fmt(Number(slider.value));
            clearTimeout(wiDebounce);
            wiDebounce = setTimeout(runWhatIf, 400);
        };
    });

    runWhatIf();
}

async function runWhatIf() {
    if (!lastFormData) return;
    const modData = { ...lastFormData };
    modData.ApplicantIncome = parseFloat(document.getElementById('wi-income').value);
    modData.CoapplicantIncome = parseFloat(document.getElementById('wi-coincome').value);
    modData.LoanAmount = parseFloat(document.getElementById('wi-loan').value);
    modData.Existing_Debt = parseFloat(document.getElementById('wi-debt').value);
    modData.Credit_Utilization = parseFloat(document.getElementById('wi-util').value);
    modData.Employment_Stability = parseFloat(document.getElementById('wi-emp').value);

    try {
        const res = await fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modData)
        });
        if (!res.ok) return;
        const data = await res.json();
        if (!data || !data.explanation) return;

        const approved = data.prediction === 'Approved';
        const wiBanner = document.getElementById('wi-decision-banner');
        wiBanner.className = `decision-banner wi-banner ${approved ? '' : 'rejected'}`;
        document.getElementById('wi-decision-label').textContent = data.prediction;
        document.getElementById('wi-prob-text').textContent = `Probability: ${(data.probability * 100).toFixed(1)}%`;
        document.getElementById('wi-score-num').textContent = `${(data.probability * 100).toFixed(0)}%`;

        const delta = data.probability - baseProb;
        const deltaEl = document.getElementById('wi-delta');
        deltaEl.textContent = `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}%`;
        deltaEl.className = `delta-value ${delta > 0.01 ? 'positive' : delta < -0.01 ? 'negative' : 'neutral'}`;

        const wiFactors = (data.explanation && data.explanation.all_factors) ||
                          (data.explanation && data.explanation.top_factors) || [];
        renderSHAPChart(wiFactors, 'wiShapChart');
    } catch (e) {
        console.error('What-If error:', e);
    }
}

function formatFeature(name) {
    const map = {
        'Credit_Utilization': 'Credit Utilization',
        'EMI_to_Income': 'EMI-to-Income',
        'Balance_Income': 'Balance Income',
        'Employment_Stability': 'Employment Stability',
        'DTI': 'Debt-to-Income (DTI)',
        'LoanAmount_Log': 'Loan Amount',
        'EMI': 'Monthly EMI',
        'Total_Income_Log': 'Total Income',
        'Property_Semiurban': 'Semiurban Area',
        'Property_Urban': 'Urban Area',
        'Loan_Amount_Term': 'Loan Term',
    };
    return map[name] || name.replace(/_/g, ' ');
}
