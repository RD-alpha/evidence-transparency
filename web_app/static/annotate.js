const annotateShell = document.querySelector('.annotate-shell');
const progressLabel = document.getElementById('progress-label');
const progressFill = document.getElementById('progress-fill');
const currentCodeEl = document.getElementById('current-code');
const decisionButtons = document.querySelectorAll('.decision-btn');

const initialAnnotations = JSON.parse(document.getElementById('initial-annotations').textContent || '{}');

const sentenceEls = Array.from(document.querySelectorAll('.sentence'));
const codes = sentenceEls.map((el) => el.dataset.code);
const annotations = { ...initialAnnotations };

const DECISION_CLASSES = ['decision-diagnosis', 'decision-proposal', 'decision-implementation', 'decision-testing', 'decision-none'];

// flowIndex: the "next unannotated" pointer the auto-advance uses.
// activeIndex: the sentence currently selected for annotation (may be a
// re-click on an already-annotated sentence, decoupled from the flow).
let flowIndex = -1;
let activeIndex = -1;
let isCorrecting = false;

function applyExistingDecision(el) {
  const code = el.dataset.code;
  const record = annotations[code];
  if (record && record.decision) {
    el.classList.add('is-annotated', `decision-${record.decision}`);
  }
}

function firstUnannotatedIndex() {
  for (let i = 0; i < codes.length; i += 1) {
    if (!annotations[codes[i]]) return i;
  }
  return -1;
}

function updateProgress() {
  const done = codes.filter((code) => annotations[code]).length;
  progressLabel.textContent = `${done} / ${codes.length} annotated`;
  progressFill.style.width = codes.length ? `${(done / codes.length) * 100}%` : '0%';
  if (done === codes.length) {
    annotateShell.classList.add('is-complete');
  }
}

function setActive(index, { scroll = true } = {}) {
  if (activeIndex >= 0 && sentenceEls[activeIndex]) {
    sentenceEls[activeIndex].classList.remove('is-current');
  }
  activeIndex = index;

  if (index < 0 || index >= sentenceEls.length) {
    currentCodeEl.textContent = 'All sentences annotated';
    isCorrecting = false;
    return;
  }

  const el = sentenceEls[index];
  el.classList.add('is-current');
  isCorrecting = Boolean(annotations[el.dataset.code]);
  currentCodeEl.classList.toggle('is-correcting', isCorrecting);
  currentCodeEl.textContent = isCorrecting
    ? `${el.dataset.code} (recategorising)`
    : el.dataset.code;
  if (scroll) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

async function saveAnnotation(code, text, decision) {
  const response = await fetch('/annotate/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, text, decision, confidence: null }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save annotation for ${code}`);
  }
  return response.json();
}

async function handleDecision(decision) {
  if (activeIndex < 0 || activeIndex >= sentenceEls.length) return;
  const wasFlowSentence = activeIndex === flowIndex;
  const el = sentenceEls[activeIndex];
  const code = el.dataset.code;
  const text = el.dataset.text;

  try {
    const result = await saveAnnotation(code, text, decision);
    annotations[code] = result.annotation;
  } catch (err) {
    console.error(err);
    return;
  }

  el.classList.remove(...DECISION_CLASSES);
  el.classList.add('is-annotated', `decision-${decision}`);
  updateProgress();

  if (wasFlowSentence) {
    // Normal progression: move on to the next unannotated sentence.
    flowIndex = firstUnannotatedIndex();
    setActive(flowIndex);
  } else {
    // A correction was made out of band; resume wherever the flow was.
    setActive(flowIndex, { scroll: true });
  }
}

decisionButtons.forEach((button) => {
  button.addEventListener('click', () => handleDecision(button.dataset.decision));
});

sentenceEls.forEach((el, index) => {
  el.addEventListener('click', () => {
    if (index === activeIndex) return;
    setActive(index, { scroll: false });
  });
});

const KEY_TO_DECISION = {
  '1': 'diagnosis',
  '2': 'proposal',
  '3': 'implementation',
  '4': 'testing',
  '5': 'none',
};

document.addEventListener('keydown', (event) => {
  const decision = KEY_TO_DECISION[event.key];
  if (decision) {
    event.preventDefault();
    handleDecision(decision);
  }
});

sentenceEls.forEach(applyExistingDecision);
updateProgress();
flowIndex = firstUnannotatedIndex() === -1 ? sentenceEls.length : firstUnannotatedIndex();
setActive(flowIndex);
