// static/js/script.js
// Enhanced Fee Reminder App - Hardened drop-in
class FeeReminderApp {
  constructor() {
    this.selectedFile = null;
    this.isProcessing = false;
    this.currentStep = 1;
    this.sessionId = null;

    // >>> If your Flask runs elsewhere (e.g., http://127.0.0.1:5000), set it here:
    this.BASE_URL = ''; // '' uses same origin. Example: 'http://127.0.0.1:5000'

    this.initializeElements();
    this.setupFileValidation();
    this.bindEvents();
    this.initializeAnalytics();

    // Initial UI state
    this.elements.loadingOverlay?.classList.add('hidden');
    if (this.elements.exportButton) this.elements.exportButton.disabled = true;
    if (this.elements.emailButton) this.elements.emailButton.disabled = true;
    if (this.elements.submitButton) this.elements.submitButton.disabled = true;

    console.log('FeeReminderApp initialized', this.elements);
  }

  // -------- DOM ----------
  initializeElements() {
    this.elements = {
      form: document.getElementById('upload-form'),
      fileInput: document.getElementById('file-upload'),
      dropZone: document.getElementById('drop-zone'),
      dropZonePrompt: document.querySelector('.drop-zone-prompt'),
      fileDisplay: document.getElementById('file-display'),
      fileNameSpan: document.getElementById('file-name'),
      removeFileButton: document.getElementById('remove-file'),
      submitButton: document.getElementById('submit-btn'),
      loadingOverlay: document.getElementById('loading-overlay'),
      progressBar: document.getElementById('progress-bar'),
      progressText: document.getElementById('progress-text'),
      resultsContent: document.getElementById('results'),
      resultsPrompt: document.querySelector('.results-prompt'),
      actionButtons: document.getElementById('action-buttons'),
      exportButton: document.getElementById('export-btn'),
      emailButton: document.getElementById('email-btn'),
      stepper: document.getElementById('stepper'),
      errorContainer: document.getElementById('error-container'),
      successContainer: document.getElementById('success-container'),
      fileSizeLabel: document.getElementById('file-size'),
      healthPill: document.getElementById('health'),
    };

    // Log missing elements for debugging
    Object.keys(this.elements).forEach((key) => {
      if (!this.elements[key]) console.warn(`Element not found: ${key}`);
    });
  }

  // -------- Config --------
  setupFileValidation() {
    this.fileConfig = {
      maxSize: 16 * 1024 * 1024, // 16MB
      allowedExtensions: ['.xlsx', '.xls'],
    };
  }

  initializeAnalytics() {
    this.analytics = { sessionStart: Date.now(), events: [] };
  }

  // -------- Events --------
  bindEvents() {
    const { dropZone, fileInput, form, submitButton } = this.elements;

    if (dropZone) {
      dropZone.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput?.click();
      });
    }

    if (fileInput) {
      fileInput.addEventListener('change', this.handleFileInputChange.bind(this));
    }

    this.bindDragDropEvents();

    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleFormSubmit(e);
      });
    }

    if (submitButton) {
      submitButton.addEventListener('click', (e) => {
        e.preventDefault();
        if (submitButton.disabled || !this.selectedFile || this.isProcessing) return false;
        this.handleFormSubmit(e);
      });
    }

    this.elements.removeFileButton?.addEventListener('click', this.resetFileInput.bind(this));
    this.elements.exportButton?.addEventListener('click', this.handleExport.bind(this));
    this.elements.emailButton?.addEventListener('click', this.handleEmailPreparation.bind(this));

    this.bindKeyboardEvents();
  }

  bindDragDropEvents() {
    const { dropZone } = this.elements;
    if (!dropZone) return;

    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((evt) => {
      dropZone.addEventListener(evt, stop, false);
      document.body.addEventListener(evt, stop, false);
    });

    dropZone.addEventListener('dragover', () => dropZone.classList.add('drag-over'));
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
      dropZone.classList.remove('drag-over');
      this.handleDrop(e);
    });
  }

  bindKeyboardEvents() {
    const { dropZone } = this.elements;
    if (!dropZone) return;
    dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.elements.fileInput?.click();
      }
    });
  }

  // -------- File Handling --------
  validateFile(file) {
    if (!file) return { valid: false, error: 'No file selected' };
    if (file.size > this.fileConfig.maxSize) {
      return { valid: false, error: `File too large. Max ${this.formatFileSize(this.fileConfig.maxSize)}` };
    }
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!this.fileConfig.allowedExtensions.includes(ext)) {
      return { valid: false, error: 'Invalid file format. Upload .xlsx or .xls' };
    }
    return { valid: true };
  }

  handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files && files.length) this.handleFile(files[0]);
  }

  handleFileInputChange() {
    const files = this.elements.fileInput?.files;
    if (files && files.length) this.handleFile(files[0]);
  }

  handleFile(file) {
    const validation = this.validateFile(file);
    if (!validation.valid) {
      this.showError(validation.error);
      this.resetFileInput();
      return;
    }
    this.selectedFile = file;
    this.updateFileDisplay(file);
    this.enableSubmission(true);
    this.clearMessages();
    this.trackEvent('file_selected', { file_size: file.size, file_name: file.name });
  }

  updateFileDisplay(file) {
    const { fileNameSpan, dropZonePrompt, fileDisplay, fileSizeLabel } = this.elements;
    if (fileNameSpan) fileNameSpan.textContent = `${file.name} (${this.formatFileSize(file.size)})`;
    if (fileSizeLabel) fileSizeLabel.textContent = `Selected: ${file.name} • ${this.formatFileSize(file.size)}`;
    dropZonePrompt?.classList.add('hidden');
    fileDisplay?.classList.remove('hidden');
  }

  enableSubmission() {
    const { submitButton } = this.elements;
    if (!submitButton) return;
    submitButton.disabled = false;
    submitButton.removeAttribute('disabled');
    submitButton.classList.remove('disabled', 'cursor-not-allowed', 'opacity-50');
    submitButton.style.pointerEvents = 'auto';
    submitButton.style.cursor = 'pointer';
  }

  resetFileInput() {
    const { fileInput, fileNameSpan, fileDisplay, dropZonePrompt, submitButton, fileSizeLabel } = this.elements;
    this.selectedFile = null;
    if (fileInput) fileInput.value = '';
    if (fileNameSpan) fileNameSpan.textContent = '';
    if (fileSizeLabel) fileSizeLabel.textContent = '';
    fileDisplay?.classList.add('hidden');
    dropZonePrompt?.classList.remove('hidden');
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.classList.add('disabled');
      submitButton.textContent = 'Process File';
    }
    this.clearMessages();
  }

  // -------- Submit / Fetch --------
  async handleFormSubmit(e) {
    e.preventDefault();
    if (!this.selectedFile) {
      this.showError('Please select a file first');
      return;
    }
    if (this.isProcessing) return;
    await this.processFile();
  }

  async processFile(retryCount = 0) {
    const maxRetries = 2;
    try {
      this.setProcessingState(true);
      this.showProgress('Uploading and analyzing file...', 10);

      const formData = new FormData();
      formData.append('file', this.selectedFile);

      const response = await this.fetchWithTimeout(`${this.BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      }, 300000);

      // Handle common HTTP failures explicitly
      if (!response.ok) {
        // Try to extract error text/json for clarity
        let serverMsg = '';
        try {
          const ct = response.headers.get('content-type') || '';
          serverMsg = ct.includes('application/json') ? (await response.json()).error || '' : (await response.text());
        } catch (_) {}
        if (response.status === 413) throw new Error('File too large for server (HTTP 413).');
        if (response.status === 415) throw new Error('Unsupported media type (HTTP 415).');
        if (response.status >= 500) throw new Error(serverMsg || `Server error (HTTP ${response.status}).`);
        throw new Error(serverMsg || `Upload failed (HTTP ${response.status}).`);
      }

      this.showProgress('Processing data...', 60);

      // Defensive parse: JSON or Text
      const contentType = response.headers.get('content-type') || '';
      let result;
      if (contentType.includes('application/json')) {
        result = await response.json();
      } else {
        const text = await response.text();
        try {
          result = JSON.parse(text);
        } catch {
          throw new Error(text || 'Server returned a non-JSON response.');
        }
      }

      // Expecting { data: [...] }
      const data = Array.isArray(result?.data) ? result.data : Array.isArray(result) ? result : null;
      if (!data) throw new Error('Unexpected server response format.');

      this.showProgress('Analysis complete!', 100);
      this.handleUploadSuccess(data);
      this.showSuccess(`File processed successfully! ${data.length} record(s).`);
      setTimeout(() => this.hideProgress(), 800);
    } catch (error) {
      if (retryCount < maxRetries && this.shouldRetry(error)) {
        this.showProgress(`Retrying... (${retryCount + 1}/${maxRetries})`, 30, 'warning');
        await this.delay(1200);
        return this.processFile(retryCount + 1);
      }
      this.handleUploadError(error);
    } finally {
      this.setProcessingState(false);
    }
  }

  async fetchWithTimeout(url, options, timeout = 30000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
      const resp = await fetch(url, { ...options, signal: controller.signal, credentials: 'same-origin' });
      clearTimeout(id);
      return resp;
    } catch (err) {
      clearTimeout(id);
      if (err.name === 'AbortError') throw new Error('Request timeout. Please try again.');
      throw err;
    }
  }

  shouldRetry(error) {
    const msg = (error?.message || '').toLowerCase();
    return (
      msg.includes('timeout') ||
      msg.includes('network') ||
      msg.includes('failed to fetch') ||
      msg.includes('server error') ||
      msg.includes('(http 5')
    );
  }

  delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  setProcessingState(processing) {
    this.isProcessing = processing;
    const { loadingOverlay, submitButton } = this.elements;
    if (processing) {
      loadingOverlay?.classList.remove('hidden');
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';
        submitButton.classList.add('disabled');
      }
    } else {
      loadingOverlay?.classList.add('hidden');
      if (submitButton) {
        if (this.selectedFile) {
          submitButton.disabled = false;
          submitButton.removeAttribute('disabled');
          submitButton.classList.remove('disabled');
        } else {
          submitButton.disabled = true;
        }
        submitButton.textContent = 'Process File';
      }
    }
  }

  showProgress(message, percentage = 0, type = 'info') {
    if (this.elements.progressText) this.elements.progressText.textContent = message;
    if (this.elements.progressBar) {
      this.elements.progressBar.style.width = `${percentage}%`;
      this.elements.progressBar.className = `progress-bar progress-bar-${type}`;
    }
  }

  hideProgress() {
    this.elements.loadingOverlay?.classList.add('hidden');
  }

  handleUploadSuccess(data) {
    this.buildResults(data);
    this.updateStepper(2);
    if (this.elements.exportButton) this.elements.exportButton.disabled = false;
    if (this.elements.emailButton) this.elements.emailButton.disabled = false;

    const btn = this.elements.submitButton;
    if (btn && this.selectedFile) {
      btn.disabled = false;
      btn.removeAttribute('disabled');
      btn.classList.remove('disabled', 'cursor-not-allowed', 'opacity-50');
      btn.textContent = 'Reprocess File';
    }

    this.trackEvent('file_processed', {
      file_size: this.selectedFile.size,
      records_count: data.length,
    });
  }

  handleUploadError(error) {
    this.showError(`Upload failed: ${error.message}`);
    this.hideProgress();
  }

  // -------- Results UI --------
  buildResults(data) {
    const { resultsContent, resultsPrompt, actionButtons } = this.elements;
    if (!resultsContent) return;

    if (resultsPrompt) resultsPrompt.style.display = 'none';

    const card = document.createElement('div');
    card.className = 'results-card bg-white rounded-lg shadow-lg p-6';
    card.appendChild(this.createSummary(data));
    card.appendChild(this.createResultsTable(data));

    resultsContent.innerHTML = '';
    resultsContent.appendChild(card);
    actionButtons?.classList.remove('hidden');
  }

  createSummary(data) {
    const total = data.length;
    const high = data.filter((s) => (s['Risk Tier'] || s.risk_tier) === 'High Risk').length;
    const medium = data.filter((s) => (s['Risk Tier'] || s.risk_tier) === 'Medium Risk').length;
    const low = data.filter((s) => (s['Risk Tier'] || s.risk_tier) === 'Low Risk').length;
    const overdue = data.filter((s) => (s.status || '').toLowerCase() === 'overdue').length;
    const dueSoon = data.filter((s) => (s.status || '').toLowerCase() === 'due soon').length;

    const wrap = document.createElement('div');
    wrap.className = 'summary mb-6';
    wrap.innerHTML = `
      <h3 class="text-xl font-bold text-gray-800 mb-4">Analysis Summary</h3>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div class="stat-card bg-blue-50 p-4 rounded-lg text-center"><div class="text-2xl font-bold text-blue-600">${total}</div><div class="text-sm text-gray-600">Total</div></div>
        <div class="stat-card bg-red-50 p-4 rounded-lg text-center"><div class="text-2xl font-bold text-red-600">${high}</div><div class="text-sm text-gray-600">High Risk</div></div>
        <div class="stat-card bg-amber-50 p-4 rounded-lg text-center"><div class="text-2xl font-bold text-amber-600">${medium}</div><div class="text-sm text-gray-600">Medium Risk</div></div>
        <div class="stat-card bg-green-50 p-4 rounded-lg text-center"><div class="text-2xl font-bold text-green-600">${low}</div><div class="text-sm text-gray-600">Low Risk</div></div>
        <div class="stat-card bg-yellow-50 p-4 rounded-lg text-center"><div class="text-2xl font-bold text-yellow-600">${overdue}/${dueSoon}</div><div class="text-sm text-gray-600">Overdue / Due Soon</div></div>
      </div>`;
    return wrap;
  }

  createResultsTable(data) {
    const wrap = document.createElement('div');
    wrap.className = 'table-container overflow-x-auto';

    const table = document.createElement('table');
    table.className = 'results-table w-full border-collapse bg-white';

    table.innerHTML = `
      <thead>
        <tr class="bg-gray-50">
          <th class="border border-gray-200 px-4 py-3 text-left font-semibold text-gray-700">Student Name</th>
          <th class="border border-gray-200 px-4 py-3 text-left font-semibold text-gray-700">Status</th>
          <th class="border border-gray-200 px-4 py-3 text-left font-semibold text-gray-700">Risk Tier</th>
        </tr>
      </thead>
      <tbody></tbody>`;

    const tbody = table.querySelector('tbody');

    if (!Array.isArray(data) || data.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-gray-600 text-sm mt-2';
      empty.textContent = 'No results to display.';
      wrap.appendChild(empty);
      wrap.appendChild(table);
      return wrap;
    }

    data.forEach((s, i) => {
      const tr = document.createElement('tr');
      tr.className = i % 2 === 0 ? 'bg-white' : 'bg-gray-50';
      const status = s.status || 'N/A';
      const risk = s['Risk Tier'] || s.risk_tier || '—';
      const name = s.student_name || s.name || '';
      const statusClass = this.getStatusClass(status);
      const riskClass = this.getRiskClass(risk);
      tr.innerHTML = `
        <td class="border border-gray-200 px-4 py-3">${name}</td>
        <td class="border border-gray-200 px-4 py-3"><span class="status-badge ${statusClass} px-2 py-1 rounded-full text-xs font-medium">${status}</span></td>
        <td class="border border-gray-200 px-4 py-3"><span class="risk-badge ${riskClass} px-2 py-1 rounded-full text-xs font-medium">${risk}</span></td>`;
      tbody.appendChild(tr);
    });

    wrap.appendChild(table);
    return wrap;
  }

  getStatusClass(s) {
    const m = {
      Overdue: 'bg-red-100 text-red-800',
      'Due Soon': 'bg-yellow-100 text-yellow-800',
      Paid: 'bg-green-100 text-green-800',
      'N/A': 'bg-gray-100 text-gray-800',
    };
    return m[s] || 'bg-gray-100 text-gray-800';
  }

  getRiskClass(r) {
    const m = {
      'High Risk': 'bg-red-100 text-red-800',
      'Medium Risk': 'bg-orange-100 text-orange-800',
      'Low Risk': 'bg-green-100 text-green-800',
    };
    return m[r] || 'bg-gray-100 text-gray-800';
  }

  updateStepper(step) {
    this.currentStep = step;
    const steps = this.elements.stepper?.querySelectorAll('.step') || [];
    steps.forEach((el, idx) => {
      if (idx < step) { el.classList.add('completed'); el.classList.remove('active'); }
      else if (idx === step - 1) { el.classList.add('active'); el.classList.remove('completed'); }
      else { el.classList.remove('active', 'completed'); }
    });
  }

  // -------- Actions --------
  async handleExport() {
    try {
      this.showProgress('Preparing export...', 50);
      const resp = await fetch(`${this.BASE_URL}/export`);
      if (!resp.ok) {
        let msg = 'Export failed';
        try {
          const ct = resp.headers.get('content-type') || '';
          msg = ct.includes('application/json') ? (await resp.json()).error || msg : (await resp.text()) || msg;
        } catch {}
        throw new Error(msg);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'fee_risk_predictions.xlsx';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      this.showSuccess('File exported successfully!');
    } catch (e) {
      this.showError(`Export failed: ${e.message}`);
    } finally {
      this.hideProgress();
    }
  }

  async handleEmailPreparation() {
    try {
      this.setProcessingState(true);
      this.showProgress('Preparing emails...', 30);
      const resp = await fetch(`${this.BASE_URL}/prepare_emails`, { method: 'POST' });
      const ct = resp.headers.get('content-type') || '';
      const result = ct.includes('application/json') ? await resp.json() : { message: await resp.text() };
      if (!resp.ok) throw new Error(result.error || 'Email preparation failed');
      this.showProgress('Emails prepared successfully!', 100);
      this.showSuccess(result.message || 'Emails prepared.');
      if (result.log_url) setTimeout(() => window.open(result.log_url, '_blank'), 1200);
    } catch (e) {
      this.showError(`Email preparation failed: ${e.message}`);
    } finally {
      this.setProcessingState(false);
      setTimeout(() => this.hideProgress(), 1200);
    }
  }

  // -------- Messaging / Analytics --------
  showError(msg) {
    this.clearMessages();
    const c = this.elements.errorContainer;
    if (!c) return;
    c.innerHTML = `<div class="bg-red-50 border-l-4 border-red-400 p-4 mb-4"><p class="text-sm text-red-700">${msg}</p></div>`;
    c.classList.remove('hidden');
  }

  showSuccess(msg) {
    this.clearMessages();
    const c = this.elements.successContainer;
    if (!c) return;
    c.innerHTML = `<div class="bg-green-50 border-l-4 border-green-400 p-4 mb-4"><p class="text-sm text-green-700">${msg}</p></div>`;
    c.classList.remove('hidden');
    setTimeout(() => c.classList.add('hidden'), 5000);
  }

  clearMessages() {
    this.elements.errorContainer?.classList.add('hidden');
    this.elements.successContainer?.classList.add('hidden');
  }

  trackEvent(name, props = {}) {
    this.analytics.events.push({
      name,
      timestamp: Date.now(),
      properties: { ...props, sessionDuration: Date.now() - this.analytics.sessionStart, currentStep: this.currentStep },
    });
  }

  async checkHealth() {
    try {
      const r = await fetch(`${this.BASE_URL}/health`);
      const ct = r.headers.get('content-type') || '';
      return ct.includes('application/json') ? await r.json() : { status: 'unknown' };
    } catch (e) {
      return { status: 'unhealthy', error: e.message };
    }
  }

  // -------- Utils --------
  formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i];
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.feeReminderApp = new FeeReminderApp();
  window.feeReminderApp.checkHealth().then((h) => {
    const pill = window.feeReminderApp.elements.healthPill;
    if (!pill) return;
    if (h && (h.model_available || h.status === 'ok')) pill.textContent = 'Server OK • Model loaded';
    else pill.textContent = 'Server reachable • Model missing';
  });
});

if (typeof module !== 'undefined' && module.exports) {
  module.exports = FeeReminderApp;
}
