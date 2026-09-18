const form = document.querySelector('#ask-form');
const question = document.querySelector('#question');
const result = document.querySelector('#result');
const status = document.querySelector('#status');
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  question.value = button.dataset.question;
  question.focus();
}));
form.addEventListener('submit', async event => {
  event.preventDefault();
  status.textContent = 'Searching documents and generating an answer…';
  result.hidden = true;
  const submit = form.querySelector('button[type="submit"]');
  submit.disabled = true;
  try {
    const response = await fetch('/api/v1/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: question.value.trim()})
    });
    if (!response.ok) throw new Error(`Request failed (${response.status}).`);
    const data = await response.json();
    document.querySelector('#answer').textContent = data.answer;
    document.querySelector('#source-list').textContent = data.sources.length ? `Retrieved files: ${data.sources.join(', ')}` : 'No source found';
    const chunks = document.querySelector('#chunks');
    chunks.replaceChildren();
    data.chunks.forEach(chunk => {
      const item = document.createElement('article'); item.className = 'chunk';
      const source = document.createElement('strong'); source.textContent = chunk.source.split(/[\\/]/).pop();
      const score = document.createElement('span'); score.textContent = `Similarity ${chunk.score.toFixed(3)}`;
      const body = document.createElement('p'); body.textContent = chunk.text;
      item.append(source, score, body); chunks.append(item);
    });
    result.hidden = false;
    status.textContent = '';
  } catch (error) { status.textContent = error.message; }
  finally { submit.disabled = false; }
});
