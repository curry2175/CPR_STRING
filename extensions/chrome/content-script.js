(() => {
  let lastCapture = null;
  let lastRange = null;
  let button = null;

  function id() {
    try { return crypto.randomUUID(); } catch (_) {
      return `capture_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    }
  }
  function selectionText() { return String(window.getSelection()?.toString() || '').trim(); }
  function removeButton() { button?.remove(); button = null; }
  function rememberSelection() {
    const selection = window.getSelection();
    const text = selectionText();
    if (!text || !selection.rangeCount) return null;
    lastRange = selection.getRangeAt(0).cloneRange();
    lastCapture = { selectedText: text, title: document.title, url: location.href };
    return lastCapture;
  }
  async function showButton() {
    const { floatingEnabled = true } = await chrome.storage.local.get({ floatingEnabled: true });
    if (!floatingEnabled) return;
    const capture = rememberSelection();
    if (!capture || capture.selectedText === lastCapture?.selectedText && button) return;
    removeButton();
    button = document.createElement('button');
    button.id = 'vrg-audit-button';
    button.type = 'button';
    button.textContent = 'Analyze with STRING';
    button.setAttribute('aria-label', '선택한 텍스트를 STRING으로 분석');
    Object.assign(button.style, { position: 'fixed', zIndex: '2147483647', left: '12px', top: '12px', padding: '9px 12px', background: '#3157d5', color: '#fff', border: '0', borderRadius: '8px', fontWeight: '700', cursor: 'pointer' });
    button.addEventListener('mousedown', event => event.preventDefault());
    button.addEventListener('click', async event => {
      event.preventDefault();
      const saved = lastCapture;
      if (!saved?.selectedText) return;
      button.disabled = true;
      try {
        const response = await chrome.runtime.sendMessage({ type: 'CAPTURE_SELECTION', payload: { ...saved, mode: 'floating_button' } });
        if (!response?.ok) throw new Error(response?.error || 'capture failed');
      } catch (error) { console.warn('[STRING] capture failed', error); }
      finally { removeButton(); }
    });
    (document.body || document.documentElement).appendChild(button);
  }
  document.addEventListener('mouseup', () => setTimeout(showButton, 30));
  document.addEventListener('mousedown', event => { if (event.target !== button) removeButton(); });
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type !== 'GET_SELECTION') return;
    const capture = rememberSelection();
    sendResponse(capture ? { ok: true, payload: { ...capture, mode: 'active_tab' } } : { ok: false });
  });
})();
