document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI State Management
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('resultsContent');
    const initialState = document.getElementById('initialState');

    submitBtn.disabled = true;
    initialState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loading.classList.remove('hidden');

    // Gather Form Data
    const formData = {
        Loan_Purpose: document.getElementById('Loan_Purpose').value,
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

    try {
        const response = await fetch('https://creditwise-api.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            let errorDetail = `API Error: ${response.status}`;
            try {
                const errorJson = await response.json();
                errorDetail = errorJson.detail || JSON.stringify(errorJson);
            } catch (e) {
                // Ignore JSON parse error if response is not JSON
            }
            throw new Error(errorDetail);
        }

        const data = await response.json();
        
        // Update UI
        updateResults(data);

    } catch (error) {
        alert("Failed to connect to the backend API. The Render server might be waking up or still deploying. Please wait 30 seconds and try again. Error: " + error.message);
        console.error(error);
        initialState.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
        submitBtn.disabled = false;
    }
});

function updateResults(data) {
    const resultsContent = document.getElementById('resultsContent');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText');
    const riskBar = document.getElementById('riskBar');
    const riskScoreText = document.getElementById('riskScoreText');
    const shapList = document.getElementById('shapList');

    // Update Decision
    predictionText.textContent = data.prediction;
    predictionText.className = ''; // reset
    if (data.prediction === 'Approved') {
        predictionText.classList.add('approved');
    } else {
        predictionText.classList.add('rejected');
    }

    probabilityText.textContent = `Approval Probability: ${(data.probability * 100).toFixed(1)}%`;

    // Update Risk Score (0 to 1)
    const riskPct = data.risk_score * 100;
    riskBar.style.width = `${riskPct}%`;
    riskScoreText.textContent = `${data.risk_score.toFixed(2)} / 1.00`;
    
    if (data.risk_score > 0.7) {
        riskBar.style.backgroundColor = 'var(--danger)';
    } else if (data.risk_score > 0.4) {
        riskBar.style.backgroundColor = '#f59e0b'; // warning orange
    } else {
        riskBar.style.backgroundColor = 'var(--success)';
    }

    // Update Explanations
    shapList.innerHTML = '';
    data.explanation.top_factors.forEach(factor => {
        const li = document.createElement('li');
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = formatFeatureName(factor.feature);
        
        const impactSpan = document.createElement('span');
        const isPositive = factor.impact > 0;
        
        impactSpan.textContent = isPositive ? '+ Positive' : '- Negative';
        impactSpan.className = isPositive ? 'impact-pos' : 'impact-neg';
        
        li.appendChild(nameSpan);
        li.appendChild(impactSpan);
        shapList.appendChild(li);
    });

    resultsContent.classList.remove('hidden');
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ');
}
