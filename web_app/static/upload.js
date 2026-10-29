const form = document.getElementById('upload-form');
const titleInput = document.getElementById('document-title');
const aliasInput = document.getElementById('document-alias');
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('upload-dropzone');
const fileMeta = document.getElementById('file-meta');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const submitBtn = document.getElementById('submit-btn');
const submitBtnLabel = document.getElementById('submit-btn-label');
const modePdfBtn = document.getElementById('mode-pdf-btn');
const modeUrlBtn = document.getElementById('mode-url-btn');
const modeCombineBtn = document.getElementById('mode-combine-btn');
const pdfModeFields = document.getElementById('pdf-mode-fields');
const urlModeFields = document.getElementById('url-mode-fields');
const combineModeFields = document.getElementById('combine-mode-fields');
const combineList = document.getElementById('combine-list');
const urlInput = document.getElementById('document-url');

const MODE_BUTTONS = { pdf: modePdfBtn, url: modeUrlBtn, combine: modeCombineBtn };
const MODE_SUBMIT_LABELS = {
  pdf: 'Extract document text',
  url: 'Parse GOV.UK page',
  combine: 'Combine documents'
};

let mode = 'pdf';
// Sources are combined in the order they were ticked, not list order.
let combineSelection = [];
let combineDocumentsLoaded = false;

function setMode(nextMode) {
  mode = nextMode;

  Object.entries(MODE_BUTTONS).forEach(([name, button]) => {
    const isActive = name === mode;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-selected', String(isActive));
  });

  pdfModeFields.hidden = mode !== 'pdf';
  urlModeFields.hidden = mode !== 'url';
  combineModeFields.hidden = mode !== 'combine';

  fileInput.required = mode === 'pdf';
  urlInput.required = mode === 'url';

  submitBtnLabel.textContent = MODE_SUBMIT_LABELS[mode];

  statusEl.textContent = '';
  statusEl.className = 'upload-status';
  resultEl.innerHTML = '';

  if (mode === 'combine' && !combineDocumentsLoaded) {
    loadCombinableDocuments();
  }
}

modePdfBtn.addEventListener('click', () => setMode('pdf'));
modeUrlBtn.addEventListener('click', () => setMode('url'));
modeCombineBtn.addEventListener('click', () => setMode('combine'));

async function loadCombinableDocuments() {
  try {
    const response = await fetch('/upload/documents');
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    const data = await response.json();
    renderCombinableDocuments(data.documents || []);
    combineDocumentsLoaded = true;
  } catch (err) {
    combineList.innerHTML = `<p class="upload-saved-path">Could not load documents: ${escapeHtml(err.message)}</p>`;
  }
}

function renderCombinableDocuments(documents) {
  if (documents.length < 2) {
    combineList.innerHTML = '<p class="upload-saved-path">At least two ingested documents are needed before they can be combined.</p>';
    return;
  }

  combineList.innerHTML = documents.map(doc => `
    <label class="upload-combine-option">
      <input type="checkbox" value="${escapeHtml(doc.alias)}">
      <span class="upload-combine-option-title">${escapeHtml(doc.title)}</span>
      <span class="upload-combine-option-meta">${doc.sentence_count} sentences &middot; ${doc.reference_count} references &middot; <code>${escapeHtml(doc.alias)}</code></span>
      <span class="upload-combine-order" aria-hidden="true"></span>
    </label>
  `).join('');

  combineList.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        combineSelection.push(checkbox.value);
      } else {
        combineSelection = combineSelection.filter(alias => alias !== checkbox.value);
      }
      updateCombineOrderLabels();
    });
  });
}

function updateCombineOrderLabels() {
  combineList.querySelectorAll('.upload-combine-option').forEach(option => {
    const checkbox = option.querySelector('input[type="checkbox"]');
    const position = combineSelection.indexOf(checkbox.value);
    option.classList.toggle('is-selected', position !== -1);
    option.querySelector('.upload-combine-order').textContent = position === -1 ? '' : String(position + 1);
  });
}

function isPdf(file) {
  return file && (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'));
}

function formatFileSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 10 * 1024 * 1024 ? 0 : 1)} MB`;
}

function showSelectedFile(file) {
  if (!file) {
    fileMeta.hidden = true;
    return;
  }
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  fileMeta.hidden = false;
  statusEl.textContent = '';
  statusEl.className = 'upload-status';
}

function setSelectedFile(file) {
  if (!isPdf(file)) {
    statusEl.textContent = 'Please select a PDF file.';
    statusEl.className = 'upload-status is-error';
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showSelectedFile(file);
}

let aliasEditedManually = false;

function slugify(value) {
  return value.trim().toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_-]/g, '');
}

aliasInput.addEventListener('input', () => {
  aliasEditedManually = true;
  aliasInput.value = aliasInput.value.replace(/\s+/g, '_');
  aliasInput.setCustomValidity(/^[A-Za-z0-9_-]+$/.test(aliasInput.value) ? '' : 'Use only letters, numbers, hyphens, and underscores.');
});

titleInput.addEventListener('input', () => {
  if (aliasEditedManually) return;
  aliasInput.value = slugify(titleInput.value);
  aliasInput.setCustomValidity(/^[A-Za-z0-9_-]+$/.test(aliasInput.value) ? '' : 'Use only letters, numbers, hyphens, and underscores.');
});

fileInput.addEventListener('change', () => showSelectedFile(fileInput.files[0]));

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove('is-dragging');
  });
});

dropzone.addEventListener('drop', event => setSelectedFile(event.dataTransfer.files[0]));

form.addEventListener('submit', async event => {
  event.preventDefault();

  const title = titleInput.value.trim();
  const alias = aliasInput.value.trim();
  if (!title || !/^[A-Za-z0-9_-]+$/.test(alias)) {
    statusEl.textContent = 'Enter a document title and an alias without spaces.';
    statusEl.className = 'upload-status is-error';
    return;
  }

  if (mode === 'pdf') {
    await submitPdf(title, alias);
  } else if (mode === 'url') {
    await submitUrl(title, alias);
  } else {
    await submitCombine(title, alias);
  }
});

async function submitPdf(title, alias) {
  const file = fileInput.files[0];
  if (!isPdf(file)) {
    statusEl.textContent = 'Please select a PDF file.';
    statusEl.className = 'upload-status is-error';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('title', title);
  formData.append('alias', alias);

  submitBtn.disabled = true;
  submitBtn.classList.add('is-loading');
  statusEl.textContent = 'Reading your document…';
  statusEl.className = 'upload-status is-processing';
  resultEl.innerHTML = '';

  try {
    const response = await fetch('/upload/analyze', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${response.status})`);
    }

    const data = await response.json();

    if (data.flagged_pages.length === 0) {
      await confirmPdf({ upload_token: data.upload_token, filename: data.filename, title, alias, use_ocr_fallback: false });
      return;
    }

    submitBtn.disabled = false;
    submitBtn.classList.remove('is-loading');
    statusEl.textContent = 'Some pages need a decision before continuing.';
    statusEl.className = 'upload-status is-processing';
    renderOcrConfirmation(data, title, alias);
  } catch (err) {
    statusEl.textContent = `Unable to read document: ${err.message}`;
    statusEl.className = 'upload-status is-error';
    submitBtn.disabled = false;
    submitBtn.classList.remove('is-loading');
  }
}

function renderOcrConfirmation(analyzeData, title, alias) {
  const pageList = analyzeData.flagged_pages.map(p => `page ${p.page}`).join(', ');
  resultEl.innerHTML = `
    <div class="upload-result-heading"><span>Needs a decision</span><strong>${escapeHtml(analyzeData.filename)}</strong></div>
    <p class="upload-saved-path">
      ${analyzeData.flagged_pages.length} of ${analyzeData.total_pages} page(s) look like scanned images with no extractable text (${escapeHtml(pageList)}).
      Sending them to OCR recovers their content but calls the Mistral OCR API (cost applies). Every other page is already extracted directly from the PDF at no cost.
    </p>
    <div class="upload-ocr-actions">
      <button type="button" id="ocr-yes-btn" class="upload-submit">Use OCR for these pages</button>
      <button type="button" id="ocr-no-btn" class="upload-submit upload-submit-secondary">Skip these pages</button>
    </div>
  `;

  document.getElementById('ocr-yes-btn').addEventListener('click', () =>
    confirmPdf({ upload_token: analyzeData.upload_token, filename: analyzeData.filename, title, alias, use_ocr_fallback: true })
  );
  document.getElementById('ocr-no-btn').addEventListener('click', () =>
    confirmPdf({ upload_token: analyzeData.upload_token, filename: analyzeData.filename, title, alias, use_ocr_fallback: false })
  );
}

async function confirmPdf({ upload_token, filename, title, alias, use_ocr_fallback }) {
  submitBtn.disabled = true;
  submitBtn.classList.add('is-loading');
  statusEl.textContent = use_ocr_fallback ? 'Running OCR on the flagged pages…' : 'Extracting text from your document…';
  statusEl.className = 'upload-status is-processing';
  resultEl.innerHTML = '';

  try {
    const response = await fetch('/upload/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_token, filename, title, alias, use_ocr_fallback })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${response.status})`);
    }

    const data = await response.json();
    statusEl.textContent = 'Text extraction complete.';
    statusEl.className = 'upload-status is-success';
    const methodNote = data.extraction_method === 'text_layer'
      ? 'Extracted directly from the PDF -- no OCR used.'
      : data.extraction_method === 'ocr'
        ? 'Extracted via OCR.'
        : `Extracted directly from the PDF, with OCR used for page(s) ${data.pages_via_ocr.join(', ')}.`;
    resultEl.innerHTML = `
      <div class="upload-result-heading"><span>Extracted text</span><strong>${escapeHtml(data.filename)}</strong></div>
      <p class="upload-saved-path">Saved as <code>documents/${escapeHtml(data.alias)}/doc.pdf</code></p>
      <p class="upload-saved-path">${escapeHtml(methodNote)} ${data.sentence_count} sentences, ${data.reference_count} references extracted.</p>
      <details open><summary>Preview</summary><pre>${escapeHtml(data.text)}</pre></details>
    `;
  } catch (err) {
    statusEl.textContent = `Unable to extract text: ${err.message}`;
    statusEl.className = 'upload-status is-error';
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('is-loading');
  }
}

async function submitUrl(title, alias) {
  const url = urlInput.value.trim();
  if (!url || !/^https?:\/\/([^/]+\.)?gov\.uk\//i.test(url)) {
    statusEl.textContent = 'Enter a valid gov.uk page URL.';
    statusEl.className = 'upload-status is-error';
    return;
  }

  submitBtn.disabled = true;
  submitBtn.classList.add('is-loading');
  statusEl.textContent = 'Fetching and parsing the page…';
  statusEl.className = 'upload-status is-processing';
  resultEl.innerHTML = '';

  try {
    const response = await fetch('/upload-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, alias, url })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${response.status})`);
    }

    const data = await response.json();
    statusEl.textContent = 'Page parsed successfully.';
    statusEl.className = 'upload-status is-success';
    resultEl.innerHTML = `
      <div class="upload-result-heading"><span>Parsed page</span><strong>${escapeHtml(data.title)}</strong></div>
      <p class="upload-saved-path">Saved as <code>documents/${escapeHtml(data.alias)}/processed.md</code>, <code>references.json</code>, and <code>links.json</code></p>
      <p class="upload-saved-path">${data.sentence_count} sentences, ${data.reference_count} references, ${data.link_count} links extracted.</p>
    `;
  } catch (err) {
    statusEl.textContent = `Unable to parse page: ${err.message}`;
    statusEl.className = 'upload-status is-error';
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('is-loading');
  }
}

async function submitCombine(title, alias) {
  if (combineSelection.length < 2) {
    statusEl.textContent = 'Choose at least two documents to combine.';
    statusEl.className = 'upload-status is-error';
    return;
  }

  submitBtn.disabled = true;
  submitBtn.classList.add('is-loading');
  statusEl.textContent = 'Combining documents…';
  statusEl.className = 'upload-status is-processing';
  resultEl.innerHTML = '';

  try {
    const response = await fetch('/upload/combine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, alias, sources: combineSelection })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${response.status})`);
    }

    const data = await response.json();
    statusEl.textContent = 'Documents combined.';
    statusEl.className = 'upload-status is-success';

    const breakdown = data.sources.map(source => `
      <li>
        <strong>${escapeHtml(source.title)}</strong> —
        ${source.sentence_count} sentences${source.first_sentence ? ` (${escapeHtml(source.first_sentence)}–${escapeHtml(source.last_sentence)})` : ''},
        ${source.reference_count} references${source.first_reference ? ` (#${escapeHtml(source.first_reference)}–#${escapeHtml(source.last_reference)})` : ''}
      </li>`).join('');

    const missingNote = data.missing_reference_count
      ? `<p class="upload-saved-path">${data.missing_reference_count} citation(s) had no definition in their source document and are listed as placeholders in the combined reference list.</p>`
      : '';

    resultEl.innerHTML = `
      <div class="upload-result-heading"><span>Combined document</span><strong>${escapeHtml(data.title)}</strong></div>
      <p class="upload-saved-path">Saved as <code>documents/${escapeHtml(data.alias)}/processed.md</code>, <code>references.json</code>, <code>links.json</code>, and <code>input.txt</code></p>
      <p class="upload-saved-path">${data.sentence_count} sentences and ${data.reference_count} references in one shared list.</p>
      ${missingNote}
      <ul class="upload-saved-path">${breakdown}</ul>
    `;
  } catch (err) {
    statusEl.textContent = `Unable to combine documents: ${err.message}`;
    statusEl.className = 'upload-status is-error';
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('is-loading');
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
