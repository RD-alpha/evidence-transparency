const rows = Array.from(document.querySelectorAll('.diff-row'));
const progressLabel = document.getElementById('progress-label');
const progressFill = document.getElementById('progress-fill');

let resolvedCount = JSON.parse(document.getElementById('initial-resolved-count').textContent || '0');
const total = rows.length;

function updateProgress() {
  progressLabel.textContent = `${resolvedCount} / ${total} resolved`;
  progressFill.style.width = total ? `${(resolvedCount / total) * 100}%` : '0%';
}

async function resolveDiff(code, action, claudeDecision) {
  const response = await fetch('/review/resolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, action, claude_decision: claudeDecision }),
  });
  if (!response.ok) {
    throw new Error(`Failed to resolve ${code}`);
  }
  return response.json();
}

rows.forEach((row) => {
  const code = row.dataset.code;
  const claudeDecision = row.dataset.claudeDecision;
  const buttons = row.querySelectorAll('.diff-btn');

  buttons.forEach((button) => {
    button.addEventListener('click', async () => {
      if (row.classList.contains('is-resolved')) return;
      const action = button.dataset.action;
      try {
        await resolveDiff(code, action, claudeDecision);
      } catch (err) {
        console.error(err);
        return;
      }
      row.classList.add('is-resolved');
      resolvedCount += 1;
      updateProgress();
    });
  });
});

updateProgress();
