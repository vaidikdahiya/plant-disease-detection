/* Plant Disease Detection — Frontend */

const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImg = document.getElementById('previewImg');
const clearBtn   = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText    = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');
const resultsSection = document.getElementById('resultsSection');

let selectedFile = null;

// ── Drag & Drop ─────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File Handling ────────────────────────────────────────────────────────────
function handleFile(file) {
    const allowed = ['image/png','image/jpeg','image/bmp','image/webp'];
    if (!allowed.includes(file.type)) {
        alert('Please upload a PNG, JPG, BMP, or WEBP image.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewSection.classList.remove('hidden');
        uploadZone.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    analyzeBtn.classList.add('hidden');
    resultsSection.classList.add('hidden');
});

// ── Analysis ─────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Loading state
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnSpinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const resp = await fetch('/predict', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        renderResults(data);
    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        analyzeBtn.disabled = false;
        btnText.textContent = '🔍 Analyze Leaf';
        btnSpinner.classList.add('hidden');
    }
});

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
    const preds = data.predictions;
    const top = preds[0];
    const isHealthy = top.disease.toLowerCase().includes('healthy');
    const severity = top.severity;

    // Primary card class
    const primaryEl = document.getElementById('primaryResult');
    primaryEl.className = 'result-primary';
    if (!isHealthy) {
        if (severity === 'Critical') primaryEl.classList.add('critical');
        else if (severity === 'High') primaryEl.classList.add('disease');
        else primaryEl.classList.add('warning');
    }

    primaryEl.innerHTML = `
        <div class="result-plant">${top.plant}</div>
        <div class="result-disease">${isHealthy ? '✅ ' : '⚠️ '}${top.disease}</div>
        <span class="severity-badge severity-${severity}">Severity: ${severity}</span>
        <div class="result-confidence">
            <div class="confidence-label">Confidence</div>
            <div class="confidence-bar-outer">
                <div class="confidence-bar-inner ${isHealthy ? '' : 'disease'}" style="width:${top.confidence}%"></div>
            </div>
            <div class="confidence-value">${top.confidence}%</div>
        </div>
    `;

    // Predictions list
    const listEl = document.getElementById('predictionsList');
    listEl.innerHTML = preds.map((p, i) => `
        <div class="pred-item">
            <div class="pred-top">
                <span class="pred-name">${p.plant} — ${p.disease}</span>
                <span class="pred-pct">${p.confidence}%</span>
            </div>
            <div class="pred-bar-outer">
                <div class="pred-bar-inner ${i === 0 ? 'first' : ''}" style="width:${p.confidence}%"></div>
            </div>
        </div>
    `).join('');

    // Treatment
    const treatEl = document.getElementById('treatmentBox');
    treatEl.innerHTML = `
        <h4>💊 Recommended Treatment</h4>
        <p>${top.treatment}</p>
    `;

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
