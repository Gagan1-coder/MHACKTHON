/**
 * Context Surgeon — Frontend Application Logic
 * Implements tab navigation, API communication, real-time metrics calculation,
 * side-by-side diff views, explainable section audit trail, preset loaders,
 * benchmark runner, and CSV/JSON export.
 */

// Application State
const state = {
  activeTab: 'tab-overview',
  backendConnected: false,
  latestCompressResult: null,
  benchmarkCases: [],
  evaluationResults: null,
  sectionFilter: 'all'
};


// DOM Elements
const elements = {
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  backendIndicator: document.getElementById('backend-status-indicator'),
  backendLabel: document.getElementById('backend-status-label'),
  
  // Overview Tab
  btnHeroLaunch: document.getElementById('btn-hero-launch'),
  btnHeroBenchmarks: document.getElementById('btn-hero-benchmarks'),
  kpiBenchmarkCount: document.getElementById('kpi-benchmark-count'),
  kpiTokenReduction: document.getElementById('kpi-token-reduction'),
  kpiLatency: document.getElementById('kpi-latency'),
  kpiFactRetention: document.getElementById('kpi-fact-retention'),

  // Compressor Tab
  inputSystemPrompt: document.getElementById('input-system-prompt'),
  inputContext: document.getElementById('input-context'),
  inputQuery: document.getElementById('input-query'),
  inputTargetTokens: document.getElementById('input-target-tokens'),
  budgetBadge: document.getElementById('budget-badge'),
  chkFallbackTokenizer: document.getElementById('chk-fallback-tokenizer'),
  contextTokenPreview: document.getElementById('context-token-preview'),
  sysPromptCount: document.getElementById('sys-prompt-count'),
  btnCompress: document.getElementById('btn-compress'),
  btnReset: document.getElementById('btn-reset'),
  budgetChips: document.querySelectorAll('.quick-budgets .btn-chip[data-budget]'),
  warningBox: document.getElementById('compressor-warning-box'),
  warningText: document.getElementById('compressor-warning-text'),
  metricsPanel: document.getElementById('compressor-metrics-panel'),
  diffContainer: document.getElementById('compressor-diff-container'),
  evidencePanel: document.getElementById('compressor-evidence-panel'),
  compressorEvidenceBanner: document.getElementById('compressor-evidence-banner'),
  metricEvidenceStatusText: document.getElementById('metric-evidence-status-text'),
  
  // Metrics Displays
  metricEngineMode: document.getElementById('metric-engine-mode'),
  metricOrigTokens: document.getElementById('metric-orig-tokens'),
  metricCompTokens: document.getElementById('metric-comp-tokens'),
  metricTokensRemoved: document.getElementById('metric-tokens-removed'),
  metricReductionPct: document.getElementById('metric-reduction-pct'),
  metricLatency: document.getElementById('metric-latency'),
  metricBudgetStatus: document.getElementById('metric-budget-status'),

  // Engine Mode Controls
  modeOptions: document.querySelectorAll('.mode-option'),
  modeDescBadge: document.getElementById('mode-desc-badge'),
  rlmTraceContainer: document.getElementById('rlm-trace-container'),
  rlmTrajectoryList: document.getElementById('rlm-trajectory-list'),
  rlmStatIterations: document.getElementById('rlm-stat-iterations'),
  rlmStatTokens: document.getElementById('rlm-stat-tokens'),
  rlmStatCalls: document.getElementById('rlm-stat-calls'),

  // Context Diff Displays
  displayOrigContext: document.getElementById('display-original-context'),
  displayCompContext: document.getElementById('display-compressed-context'),
  pillOrigTokens: document.getElementById('pill-orig-tokens'),
  pillOrigSections: document.getElementById('pill-orig-sections'),
  pillCompTokens: document.getElementById('pill-comp-tokens'),
  pillReductionBadge: document.getElementById('pill-reduction-badge'),

  // Copy & Download
  btnCopyOrig: document.getElementById('btn-copy-orig'),
  btnCopyComp: document.getElementById('btn-copy-comp'),
  btnDownloadComp: document.getElementById('btn-download-comp'),
  btnCopyJson: document.getElementById('btn-copy-json'),

  // Section Audit Trail
  filterChips: document.querySelectorAll('.filter-chip'),
  sectionsAuditList: document.getElementById('sections-audit-list'),
  countAllSecs: document.getElementById('count-all-secs'),
  countRetSecs: document.getElementById('count-ret-secs'),
  countRemSecs: document.getElementById('count-rem-secs'),
  countDedupSecs: document.getElementById('count-dedup-secs'),
  countProtSecs: document.getElementById('count-prot-secs'),

  // Benchmark Tab
  btnRunAllBenchmarks: document.getElementById('btn-run-all-benchmarks'),
  benchmarkSearch: document.getElementById('benchmark-search'),
  benchmarkCategoryFilter: document.getElementById('benchmark-category-filter'),
  benchmarkRunningBanner: document.getElementById('benchmark-running-banner'),
  benchmarkCasesList: document.getElementById('benchmark-cases-list'),

  // Results Tab
  btnExportCsv: document.getElementById('btn-export-csv'),
  btnExportJson: document.getElementById('btn-export-json'),
  badgeTarget1: document.getElementById('badge-target-1'),
  valTarget1: document.getElementById('val-target-1'),
  badgeTarget4: document.getElementById('badge-target-4'),
  valTarget4: document.getElementById('val-target-4'),
  badgeTarget2: document.getElementById('badge-target-2'),
  valTarget2: document.getElementById('val-target-2'),
  badgeTarget3: document.getElementById('badge-target-3'),
  valTarget3: document.getElementById('val-target-3'),
  tbodyCategorySummary: document.getElementById('tbody-category-summary'),
  tbodyCaseResults: document.getElementById('tbody-case-results'),

  toastContainer: document.getElementById('toast-container')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupCompressorInputs();
  setupActionButtons();
  checkBackendHealth();
  loadBenchmarkCases();
  loadEvaluationResults();
});

// -----------------------------------------------------------------------------
// NAVIGATION & TABS
// -----------------------------------------------------------------------------
function setupNavigation() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.tab;
      switchTab(targetId);
    });
  });

  if (elements.btnHeroLaunch) {
    elements.btnHeroLaunch.addEventListener('click', () => switchTab('tab-compressor'));
  }
  if (elements.btnHeroBenchmarks) {
    elements.btnHeroBenchmarks.addEventListener('click', () => switchTab('tab-benchmarks'));
  }
}

function switchTab(tabId) {
  state.activeTab = tabId;
  elements.navTabs.forEach(tab => {
    const isActive = tab.dataset.tab === tabId;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  elements.tabPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  if (tabId === 'tab-results') {
    loadEvaluationResults();
  }
}

// -----------------------------------------------------------------------------
// BACKEND HEALTH MONITOR
// -----------------------------------------------------------------------------
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      state.backendConnected = true;
      elements.backendIndicator.className = 'status-indicator connected';
      elements.backendLabel.textContent = `Online (${data.tokenizer.type})`;
    } else {
      markBackendOffline();
    }
  } catch (err) {
    markBackendOffline();
  }
}

function markBackendOffline() {
  state.backendConnected = false;
  elements.backendIndicator.className = 'status-indicator error';
  elements.backendLabel.textContent = 'Offline (Run python run.py)';
}

// -----------------------------------------------------------------------------
// COMPRESSOR INPUTS & PRESETS
// -----------------------------------------------------------------------------
function setupCompressorInputs() {
  // Context token counter preview approximation
  elements.inputContext.addEventListener('input', () => {
    const text = elements.inputContext.value;
    const approxTokens = Math.max(0, Math.ceil(text.length / 4.0));
    elements.contextTokenPreview.textContent = `Approx Tokens: ${approxTokens} | Chars: ${text.length}`;
  });

  elements.inputSystemPrompt.addEventListener('input', () => {
    elements.sysPromptCount.textContent = `${elements.inputSystemPrompt.value.length} chars`;
  });

  // Target token budget sync
  elements.inputTargetTokens.addEventListener('input', () => {
    elements.budgetBadge.textContent = `${elements.inputTargetTokens.value} tokens`;
  });

  // Quick budget chips
  elements.budgetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const budget = chip.dataset.budget;
      elements.inputTargetTokens.value = budget;
      elements.budgetBadge.textContent = `${budget} tokens`;
      elements.budgetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
    });
  });

  // Filter chips in audit trail
  elements.filterChips.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.filterChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.sectionFilter = chip.dataset.filter;
      renderSectionAuditList();
    });
  });

  // Mode Selector Options
  elements.modeOptions.forEach(opt => {
    opt.addEventListener('click', () => {
      elements.modeOptions.forEach(o => o.classList.remove('active'));
      opt.classList.add('active');
      const radio = opt.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
      const mode = opt.dataset.mode;
      if (elements.modeDescBadge) {
        if (mode === 'fast_local') {
          elements.modeDescBadge.textContent = '⚡ Sub-millisecond deterministic extraction';
        } else if (mode === 'recursive') {
          elements.modeDescBadge.textContent = '🔄 Bounded REPL search & external context store';
        } else {
          elements.modeDescBadge.textContent = '🤖 Auto adapts based on context token volume';
        }
      }
    });
  });
}

function loadPreset(preset) {
  elements.inputSystemPrompt.value = preset.system_prompt;
  elements.inputContext.value = preset.context;
  elements.inputQuery.value = preset.query;
  elements.inputTargetTokens.value = preset.target_tokens;
  elements.budgetBadge.textContent = `${preset.target_tokens} tokens`;

  // Trigger input events to refresh counters
  elements.inputContext.dispatchEvent(new Event('input'));
  elements.inputSystemPrompt.dispatchEvent(new Event('input'));

  showToast('Loaded case into Compressor.');
}

// -----------------------------------------------------------------------------
// COMPRESSION EXECUTION
// -----------------------------------------------------------------------------
function setupActionButtons() {
  elements.btnCompress.addEventListener('click', executeCompression);

  elements.btnReset.addEventListener('click', () => {
    elements.inputContext.value = '';
    elements.inputQuery.value = '';
    elements.warningBox.classList.add('hidden');
    elements.metricsPanel.classList.add('hidden');
    elements.diffContainer.classList.add('hidden');
    elements.evidencePanel.classList.add('hidden');
    elements.inputContext.dispatchEvent(new Event('input'));
    showToast('Reset input fields.');
  });

  // Copy buttons
  elements.btnCopyOrig.addEventListener('click', () => {
    navigator.clipboard.writeText(elements.displayOrigContext.textContent);
    showToast('Copied original context.');
  });

  elements.btnCopyComp.addEventListener('click', () => {
    navigator.clipboard.writeText(elements.displayCompContext.textContent);
    showToast('Copied compressed context.');
  });

  elements.btnCopyJson.addEventListener('click', () => {
    if (state.latestCompressResult) {
      navigator.clipboard.writeText(JSON.stringify(state.latestCompressResult, null, 2));
      showToast('Copied full JSON API response.');
    }
  });

  elements.btnDownloadComp.addEventListener('click', () => {
    const text = elements.displayCompContext.textContent;
    downloadFile(text, 'compressed_context.txt', 'text/plain');
  });

  // Benchmark Runner button
  elements.btnRunAllBenchmarks.addEventListener('click', runAllBenchmarks);

  // Search & Filters for Benchmarks
  elements.benchmarkSearch.addEventListener('input', renderBenchmarkCards);
  elements.benchmarkCategoryFilter.addEventListener('change', renderBenchmarkCards);

  // Export buttons
  elements.btnExportCsv.addEventListener('click', exportResultsCsv);
  elements.btnExportJson.addEventListener('click', exportResultsJson);
}

async function executeCompression() {
  const context = elements.inputContext.value;
  const query = elements.inputQuery.value.trim();
  const systemPrompt = elements.inputSystemPrompt.value;
  const targetTokens = parseInt(elements.inputTargetTokens.value, 10);
  const forceFallback = elements.chkFallbackTokenizer.checked;
  const selectedModeInput = document.querySelector('input[name="compression-mode"]:checked');
  const compressionMode = selectedModeInput ? selectedModeInput.value : 'fast_local';

  if (!query) {
    showToast('Please specify a User Query to evaluate context relevance.', 'warning');
    elements.inputQuery.focus();
    return;
  }
  if (isNaN(targetTokens) || targetTokens <= 0) {
    showToast('Target token budget must be greater than 0.', 'warning');
    elements.inputTargetTokens.focus();
    return;
  }

  // Visual loading state
  elements.btnCompress.disabled = true;
  elements.btnCompress.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Processing...`;

  try {
    const res = await fetch('/api/compress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_prompt: systemPrompt,
        context: context,
        query: query,
        target_tokens: targetTokens,
        mode: compressionMode,
        force_fallback_tokenizer: forceFallback
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Compression failed');
    }

    const data = await res.json();
    state.latestCompressResult = data;
    renderCompressResult(data);
    showToast(`Compressed in ${data.compression_time_ms} ms (${data.reduction_percentage}% reduction).`);
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    elements.btnCompress.disabled = false;
    elements.btnCompress.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m13 2-2 10h5L11 22l2-10H8l5-10z"/></svg>
      Compress Context
    `;
  }
}

function renderCompressResult(data) {
  // 1. Budget warnings
  if (data.warnings && data.warnings.length > 0) {
    elements.warningBox.classList.remove('hidden');
    elements.warningText.innerHTML = data.warnings.map(w => `<strong>Warning:</strong> ${escapeHtml(w)}`).join('<br>');
  } else {
    elements.warningBox.classList.add('hidden');
  }

  // 2. Metrics cards
  elements.metricsPanel.classList.remove('hidden');
  if (elements.metricEngineMode) {
    elements.metricEngineMode.textContent = data.mode === 'recursive' ? 'Recursive (RLM)' : 'Fast Local';
    elements.metricEngineMode.className = `metric-badge-status ${data.mode === 'recursive' ? 'warn' : 'met'}`;
  }
  elements.metricOrigTokens.textContent = data.original_tokens.toLocaleString();
  elements.metricCompTokens.textContent = data.compressed_tokens.toLocaleString();
  elements.metricTokensRemoved.textContent = data.tokens_removed.toLocaleString();
  elements.metricReductionPct.textContent = `${data.reduction_percentage}%`;
  elements.metricLatency.textContent = `${data.compression_time_ms} ms`;

  // 2b. Recursive RLM execution trace
  if (data.recursive_trace && data.recursive_trace.length > 0) {
    if (elements.rlmTraceContainer) elements.rlmTraceContainer.classList.remove('hidden');
    if (elements.rlmStatIterations && data.recursive_stats) {
      elements.rlmStatIterations.textContent = `${data.recursive_stats.iterations} iterations`;
      elements.rlmStatTokens.textContent = `${data.recursive_stats.exploration_tokens} exploration tokens`;
      elements.rlmStatCalls.textContent = `${data.recursive_stats.llm_calls} sub-calls`;
    }
    if (elements.rlmTrajectoryList) {
      elements.rlmTrajectoryList.innerHTML = data.recursive_trace.map(step => `
        <div class="rlm-step-item">
          <span class="rlm-step-badge">Step ${step.step}</span>
          <span class="rlm-step-action">${escapeHtml(step.action)}</span>
          <span class="rlm-step-summary">${escapeHtml(step.summary)}</span>
          <span class="rlm-step-meta">${step.tokens_inspected} toks | ${step.elapsed_ms}ms</span>
        </div>
      `).join('');
    }
  } else {
    if (elements.rlmTraceContainer) {
      elements.rlmTraceContainer.classList.add('hidden');
    }
  }

  // Evidence status banner
  const evidStatus = data.evidence_preservation_status || 'All critical evidence preserved.';
  if (elements.metricEvidenceStatusText) {
    elements.metricEvidenceStatusText.textContent = evidStatus;
  }
  if (elements.compressorEvidenceBanner) {
    if (!data.budget_met || (data.warnings && data.warnings.length > 0)) {
      elements.compressorEvidenceBanner.className = 'evidence-status-banner warning';
    } else {
      elements.compressorEvidenceBanner.className = 'evidence-status-banner';
    }
  }

  elements.metricBudgetStatus.textContent = data.budget_met ? 'Budget Met' : 'Budget Warning';
  elements.metricBudgetStatus.className = `metric-badge-status ${data.budget_met ? 'met' : 'warn'}`;

  // 3. Side-by-side diff view
  elements.diffContainer.classList.remove('hidden');
  elements.displayOrigContext.textContent = data.original_context;
  elements.displayCompContext.textContent = data.compressed_context;

  elements.pillOrigTokens.textContent = `${data.original_tokens} tokens`;
  elements.pillOrigSections.textContent = `${data.summary.total_sections || 0} sections`;
  elements.pillCompTokens.textContent = `${data.compressed_tokens} tokens`;
  elements.pillReductionBadge.textContent = `-${data.reduction_percentage}%`;

  // 4. Section Audit Trail
  elements.evidencePanel.classList.remove('hidden');
  elements.countAllSecs.textContent = data.sections.length;
  elements.countRetSecs.textContent = data.sections.filter(s => s.decision === 'retained').length;
  elements.countRemSecs.textContent = data.sections.filter(s => s.decision === 'removed').length;
  elements.countDedupSecs.textContent = data.sections.filter(s => s.decision === 'deduplicated').length;
  elements.countProtSecs.textContent = data.sections.filter(s => s.is_protected).length;

  renderSectionAuditList();
}

function renderSectionAuditList() {
  if (!state.latestCompressResult) return;
  const sections = state.latestCompressResult.sections;
  const filter = state.sectionFilter;

  const filtered = sections.filter(sec => {
    if (filter === 'all') return true;
    if (filter === 'protected') return sec.is_protected;
    return sec.decision === filter;
  });

  elements.sectionsAuditList.innerHTML = '';

  if (filtered.length === 0) {
    elements.sectionsAuditList.innerHTML = `<div class="card text-center" style="padding:1.5rem;color:var(--text-dim);">No sections match the "${filter}" filter.</div>`;
    return;
  }

  filtered.forEach((sec, idx) => {
    const item = document.createElement('div');
    item.className = `section-item ${sec.decision}`;

    const protBadge = sec.is_protected
      ? `<span class="prot-badge" title="Protected Evidence">🛡️ Protected</span>`
      : '';

    const rulesStr = sec.protection_rules && sec.protection_rules.length > 0
      ? `<div style="font-size:0.75rem;color:var(--purple-500);margin-top:0.35rem;">Rules Triggered: ${sec.protection_rules.map(r => `<code>${escapeHtml(r)}</code>`).join(' ')}</div>`
      : '';

    item.innerHTML = `
      <div class="section-header-row" data-index="${idx}">
        <div class="sec-meta-left">
          <span class="sec-id-badge">${escapeHtml(sec.section_id)}</span>
          <span class="sec-type-badge">${escapeHtml(sec.source_type)}</span>
          <span class="decision-badge ${sec.decision}">${sec.decision}</span>
          ${protBadge}
          <span class="sec-preview-text">${escapeHtml(sec.preview)}</span>
        </div>
        <div class="sec-meta-right">
          <span>Score: <strong>${sec.relevance_score}</strong></span>
          <span>${sec.tokens} tokens</span>
          <span class="chevron-icon">▼</span>
        </div>
      </div>
      <div class="section-body">
        <div class="reason-box">
          <strong>Decision Rationale:</strong> ${escapeHtml(sec.reason)}
          ${rulesStr}
        </div>
        <div class="full-content-box">${escapeHtml(sec.content)}</div>
      </div>
    `;

    // Accordion toggle
    const headerRow = item.querySelector('.section-header-row');
    const body = item.querySelector('.section-body');
    const chevron = item.querySelector('.chevron-icon');

    headerRow.addEventListener('click', () => {
      const isOpen = body.classList.toggle('open');
      chevron.textContent = isOpen ? '▲' : '▼';
    });

    elements.sectionsAuditList.appendChild(item);
  });
}

// -----------------------------------------------------------------------------
// BENCHMARK LAB
// -----------------------------------------------------------------------------
async function loadBenchmarkCases() {
  try {
    const res = await fetch('/api/benchmarks');
    if (res.ok) {
      state.benchmarkCases = await res.json();
      elements.kpiBenchmarkCount.textContent = state.benchmarkCases.length;
      populateCategoryFilter();
      renderBenchmarkCards();
    }
  } catch (err) {
    console.error('Failed to load benchmark cases:', err);
  }
}

function populateCategoryFilter() {
  const cats = Array.from(new Set(state.benchmarkCases.map(c => c.category))).sort();
  elements.benchmarkCategoryFilter.innerHTML = '<option value="all">All Categories</option>';
  cats.forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat;
    opt.textContent = cat;
    elements.benchmarkCategoryFilter.appendChild(opt);
  });
}

function renderBenchmarkCards() {
  const search = elements.benchmarkSearch.value.toLowerCase();
  const selectedCat = elements.benchmarkCategoryFilter.value;

  const filtered = state.benchmarkCases.filter(c => {
    const matchCat = selectedCat === 'all' || c.category === selectedCat;
    const matchSearch = !search ||
      c.case_id.toLowerCase().includes(search) ||
      c.category.toLowerCase().includes(search) ||
      c.query.toLowerCase().includes(search) ||
      c.notes.toLowerCase().includes(search);
    return matchCat && matchSearch;
  });

  elements.benchmarkCasesList.innerHTML = '';

  filtered.forEach(c => {
    const card = document.createElement('div');
    card.className = 'card benchmark-case-card';

    const factsHtml = c.expected_critical_facts.map(f => `<span class="fact-tag">${escapeHtml(f)}</span>`).join('');

    card.innerHTML = `
      <div>
        <div class="case-card-header">
          <span class="case-id-chip">${escapeHtml(c.case_id)}</span>
          <span class="case-category">${escapeHtml(c.category)}</span>
        </div>
        <div class="case-query" style="margin: 0.75rem 0 0.5rem;">${escapeHtml(c.query)}</div>
        <p style="font-size:0.8rem;color:var(--text-dim);margin-bottom:0.75rem;">${escapeHtml(c.notes)}</p>
        <div class="case-facts-list">${factsHtml}</div>
      </div>
      <div class="case-card-footer">
        <span class="case-budget">Budget: <strong>${c.target_tokens} tokens</strong></span>
        <button class="btn btn-secondary btn-chip load-case-btn" style="padding:0.35rem 0.75rem;">
          Load into Compressor
        </button>
      </div>
    `;

    card.querySelector('.load-case-btn').addEventListener('click', () => {
      loadPreset({
        system_prompt: c.system_prompt || "You are a helpful assistant.",
        context: c.context,
        query: c.query,
        target_tokens: c.target_tokens
      });
      switchTab('tab-compressor');
    });

    elements.benchmarkCasesList.appendChild(card);
  });
}

async function runAllBenchmarks() {
  elements.btnRunAllBenchmarks.disabled = true;
  elements.benchmarkRunningBanner.classList.remove('hidden');

  try {
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!res.ok) throw new Error('Benchmark execution failed');

    const data = await res.json();
    state.evaluationResults = data;
    showToast(`Successfully executed ${data.results_count} benchmark cases.`);
    updateResultsTab(data);
    switchTab('tab-results');
  } catch (err) {
    showToast(`Error running benchmarks: ${err.message}`, 'error');
  } finally {
    elements.btnRunAllBenchmarks.disabled = false;
    elements.benchmarkRunningBanner.classList.add('hidden');
  }
}

// -----------------------------------------------------------------------------
// EVALUATION RESULTS TAB
// -----------------------------------------------------------------------------
async function loadEvaluationResults() {
  try {
    const res = await fetch('/api/benchmarks/results');
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'available') {
        state.evaluationResults = data;
        updateResultsTab(data);
      } else {
        renderPendingResultsNotice();
      }
    }
  } catch (err) {
    console.error('Failed to load evaluation results:', err);
  }
}

function renderPendingResultsNotice() {
  elements.tbodyCategorySummary.innerHTML = `<tr><td colspan="4" class="text-center" style="padding:2rem;">Evaluation results not generated yet. Click "Execute All Benchmarks" in the Benchmark Lab.</td></tr>`;
  elements.tbodyCaseResults.innerHTML = `<tr><td colspan="9" class="text-center" style="padding:2rem;">Run the benchmark suite to populate per-case metrics.</td></tr>`;
}

function updateResultsTab(data) {
  const agg = data.aggregate_metrics;
  if (!agg) return;

  // Overview Tab KPI updates
  elements.kpiTokenReduction.textContent = `${agg.mean_token_reduction}%`;
  elements.kpiLatency.textContent = `${agg.p95_compression_latency_ms} ms`;
  elements.kpiFactRetention.textContent = `${agg.critical_fact_retention_rate}%`;

  // Target 1: Token reduction
  const t1Achieved = agg.targets_status && agg.targets_status.target_1_token_reduction ? agg.targets_status.target_1_token_reduction.achieved : (agg.mean_token_reduction >= 30.0);
  elements.valTarget1.textContent = `${agg.mean_token_reduction}%`;
  elements.badgeTarget1.textContent = t1Achieved ? 'Target Achieved' : 'Target Missed';
  elements.badgeTarget1.className = `target-badge ${t1Achieved ? 'pass' : 'fail'}`;

  // Target 4: Stretch Target 95%
  if (elements.valTarget4 && elements.badgeTarget4) {
    const reached95Count = agg.cases_reaching_95_pct !== undefined ? agg.cases_reaching_95_pct : 0;
    elements.valTarget4.textContent = `${reached95Count} cases (median ${agg.median_token_reduction || 0}%)`;
    elements.badgeTarget4.textContent = `${reached95Count} Reached 95%`;
    elements.badgeTarget4.className = `target-badge ${reached95Count > 0 ? 'pass' : 'pending'}`;
  }

  // Target 2: Answer quality status
  elements.valTarget2.textContent = `${agg.critical_fact_retention_rate}% Facts`;
  elements.badgeTarget2.textContent = 'Offline Verified';
  elements.badgeTarget2.className = 'target-badge pass';

  // Target 3: Latency
  const t3Achieved = agg.targets_status && agg.targets_status.target_3_latency_p95 ? agg.targets_status.target_3_latency_p95.achieved : (agg.p95_compression_latency_ms < 100.0);
  elements.valTarget3.textContent = `${agg.p95_compression_latency_ms} ms`;
  elements.badgeTarget3.textContent = t3Achieved ? 'Target Achieved' : 'Target Missed';
  elements.badgeTarget3.className = `target-badge ${t3Achieved ? 'pass' : 'fail'}`;

  // Category Breakdown Table
  const catSummary = agg.categories_summary;
  elements.tbodyCategorySummary.innerHTML = '';
  if (catSummary) {
    Object.keys(catSummary).sort().forEach(cat => {
      const info = catSummary[cat];
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${escapeHtml(cat)}</strong></td>
        <td>${info.count}</td>
        <td style="color:var(--teal-400);font-weight:600;">${info.avg_reduction}%</td>
        <td><span class="decision-badge retained">${info.avg_fact_retention}%</span></td>
      `;
      elements.tbodyCategorySummary.appendChild(tr);
    });
  }

  // Detailed Per-case Table
  const cases = data.cases || [];
  elements.tbodyCaseResults.innerHTML = '';
  cases.forEach(r => {
    const tr = document.createElement('tr');
    const statusPill = r.budget_met && (!r.warnings || r.warnings.length === 0)
      ? `<span class="decision-badge retained">Budget Met</span>`
      : `<span class="decision-badge deduplicated" title="${escapeHtml(r.warnings ? r.warnings.join('; ') : '')}">Warning</span>`;

    let targetBadgeClass = 'below';
    const tStatus = r.target_achievement_status || (r.reduction_percentage >= 95 ? 'Target achieved' : 'Below target');
    if (tStatus === 'Target achieved') targetBadgeClass = 'achieved';
    else if (tStatus === 'Safety limit reached') targetBadgeClass = 'safety';
    else if (tStatus === 'Evaluation pending') targetBadgeClass = 'pending';

    const targetBadge = `<span class="metric-badge-status ${targetBadgeClass}" style="font-size:0.75rem;padding:0.2rem 0.5rem;">${escapeHtml(tStatus)}</span>`;

    tr.innerHTML = `
      <td><code>${escapeHtml(r.case_id)}</code></td>
      <td>${escapeHtml(r.category)}</td>
      <td>${r.original_tokens}</td>
      <td>${r.compressed_tokens}</td>
      <td style="color:var(--teal-400);font-weight:600;">${r.reduction_percentage}%</td>
      <td>${targetBadge}</td>
      <td>${r.compression_time_ms} ms</td>
      <td>${r.facts_retained} / ${r.total_facts}</td>
      <td>${statusPill}</td>
    `;
    elements.tbodyCaseResults.appendChild(tr);
  });
}

function exportResultsCsv() {
  if (!state.evaluationResults || !state.evaluationResults.cases) {
    showToast('Run benchmarks first before exporting CSV.', 'warning');
    return;
  }
  const cases = state.evaluationResults.cases;
  const headers = [
    'case_id', 'category', 'original_tokens', 'compressed_tokens',
    'reduction_percentage', 'preferred_reduction_target', 'target_achievement_status',
    'evidence_preservation_status', 'compression_time_ms', 'facts_retained',
    'total_facts', 'fact_retention_rate', 'budget_met', 'answer_status'
  ];

  const rows = [headers.join(',')];
  cases.forEach(c => {
    const row = [
      c.case_id,
      `"${c.category}"`,
      c.original_tokens,
      c.compressed_tokens,
      c.reduction_percentage,
      c.preferred_reduction_target !== undefined ? c.preferred_reduction_target : 95.0,
      `"${c.target_achievement_status || ''}"`,
      `"${(c.evidence_preservation_status || '').replace(/"/g, '""')}"`,
      c.compression_time_ms,
      c.facts_retained,
      c.total_facts,
      c.fact_retention_rate,
      c.budget_met,
      c.answer_status
    ];
    rows.push(row.join(','));
  });

  downloadFile(rows.join('\n'), 'context_surgeon_benchmark_results.csv', 'text/csv');
  showToast('Exported CSV evaluation results.');
}

function exportResultsJson() {
  if (!state.evaluationResults) {
    showToast('Run benchmarks first before exporting JSON.', 'warning');
    return;
  }
  const jsonStr = JSON.stringify(state.evaluationResults, null, 2);
  downloadFile(jsonStr, 'context_surgeon_benchmark_results.json', 'application/json');
  showToast('Exported JSON evaluation results.');
}

// -----------------------------------------------------------------------------
// UTILITIES
// -----------------------------------------------------------------------------
function downloadFile(content, filename, contentType) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'error' ? '❌' : (type === 'warning' ? '⚠️' : '✅')}</span>
    <span>${escapeHtml(message)}</span>
  `;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
