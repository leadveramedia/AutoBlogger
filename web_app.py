#!/usr/bin/env python3
"""
Web interface for the custom TikTok video generator.
Coworkers fill in script/setting/actions, watch live logs, and download from Google Drive.
"""

import os
import sys
import builtins
import threading
import queue
import uuid
import subprocess
import tempfile
import json
import time
from concurrent.futures import ThreadPoolExecutor as _TPE
from pathlib import Path
from datetime import datetime

# Load .env before importing auto_post
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# ── Thread-safe print capture ────────────────────────────────────────────────
# Monkey-patch builtins.print once. Background job threads route output to
# their per-job queue; all other threads use normal stdout.
_original_print = builtins.print
_thread_log_queues: dict[int, queue.Queue] = {}
_tq_lock = threading.Lock()


def _patched_print(*args, **kwargs):
    tid = threading.current_thread().ident
    with _tq_lock:
        q = _thread_log_queues.get(tid)
    if q is not None:
        msg = kwargs.get('sep', ' ').join(str(a) for a in args)
        q.put(msg)
    else:
        _original_print(*args, **kwargs)


builtins.print = _patched_print

# ── Propagate log queue to ThreadPoolExecutor child threads ──────────────────
# generate_three_videos() spawns its own ThreadPoolExecutor internally.
# Patch TPE.submit so child threads inherit the parent's log queue automatically.
_original_tpe_submit = _TPE.submit

def _propagating_submit(self, fn, *args, **kwargs):
    parent_tid = threading.current_thread().ident
    with _tq_lock:
        parent_q = _thread_log_queues.get(parent_tid)
    if parent_q is None:
        return _original_tpe_submit(self, fn, *args, **kwargs)

    def wrapped(*a, **kw):
        tid = threading.current_thread().ident
        with _tq_lock:
            _thread_log_queues[tid] = parent_q
        try:
            return fn(*a, **kw)
        finally:
            with _tq_lock:
                _thread_log_queues.pop(tid, None)

    return _original_tpe_submit(self, wrapped, *args, **kwargs)

_TPE.submit = _propagating_submit

# ── Import video module (after print patch so its prints are capturable) ─────
from auto_post.video import generate_three_videos

from flask import Flask, request, jsonify, Response, stream_with_context, send_file

app = Flask(__name__)

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ── Google Drive upload via rclone ───────────────────────────────────────────

def upload_to_drive(local_path: str) -> str | None:
    """Upload an MP4 to Google Drive via rclone. Returns shareable link or None."""
    config_data = os.environ.get('RCLONE_CONFIG', '')
    folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
    if not config_data or not folder_id:
        _original_print('  [Drive] RCLONE_CONFIG or GOOGLE_DRIVE_FOLDER_ID not set — skipping upload')
        return None

    filename = os.path.basename(local_path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_data)
        config_path = f.name

    try:
        subprocess.run(
            ['rclone', 'copy', local_path, f'gdrive:{folder_id}', '--config', config_path],
            check=True, capture_output=True
        )
        result = subprocess.run(
            ['rclone', 'link', f'gdrive:{folder_id}/{filename}', '--config', config_path],
            check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        _original_print(f'  [Drive] rclone error: {e.stderr}')
        return None
    finally:
        os.unlink(config_path)


# ── Background job worker ─────────────────────────────────────────────────────

def run_job(job_id: str, article_data: dict, script: str, setting: str,
            actions: str, formats: list[str]):
    tid = threading.current_thread().ident
    with jobs_lock:
        job = jobs[job_id]
    log_q = job['log_queue']

    with _tq_lock:
        _thread_log_queues[tid] = log_q

    try:
        log_q.put(f'=== Job {job_id} started ===')
        log_q.put(f'Formats: {", ".join(formats)}')
        log_q.put(f'Script ({len(script.split())} words): {script[:120]}...' if len(script) > 120 else f'Script: {script}')

        results = generate_three_videos(
            article_data,
            custom_script=script,
            custom_setting=setting,
            custom_actions=actions,
            formats=formats,
        )

        drive_links = []
        local_files = []
        if results:
            for path in results:
                size_mb = os.path.getsize(path) / 1024 / 1024
                local_files.append({'filename': os.path.basename(path), 'path': path, 'size_mb': round(size_mb, 1)})

            # Try Google Drive upload
            log_q.put(f'\n=== Uploading {len(results)} video(s) to Google Drive ===')
            for path in results:
                fname = os.path.basename(path)
                log_q.put(f'  Uploading {fname}...')
                link = upload_to_drive(path)
                if link:
                    log_q.put(f'  ✓ {fname} → {link}')
                    drive_links.append({'filename': fname, 'url': link})
                else:
                    log_q.put(f'  ✗ Drive upload failed for {fname} — direct download still available')
        else:
            log_q.put('\n✗ No videos were generated successfully')

        with jobs_lock:
            job['status'] = 'done'
            job['drive_links'] = drive_links
            job['local_files'] = local_files

        uploaded = len(drive_links)
        total = len(local_files)
        if uploaded == total and total > 0:
            log_q.put(f'\n=== Done — {uploaded} video(s) uploaded to Drive ===')
        elif total > 0:
            log_q.put(f'\n=== Done — {total} video(s) generated ({uploaded} uploaded to Drive) ===')

    except Exception as e:
        with jobs_lock:
            job['status'] = 'failed'
            job['error'] = str(e)
        log_q.put(f'\n✗ Job failed: {e}')

    finally:
        with _tq_lock:
            _thread_log_queues.pop(tid, None)
        log_q.put(None)  # sentinel — SSE stream closes


# ── Routes ────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video Generator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f0f0f; color: #e8e8e8; min-height: 100vh; display: flex; align-items: flex-start; justify-content: center; padding: 40px 16px; }
  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 32px; width: 100%; max-width: 680px; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 8px; }
  .subtitle { color: #888; font-size: 0.875rem; margin-bottom: 28px; }
  label { display: block; font-size: 0.8rem; font-weight: 500; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .hint { font-size: 0.75rem; color: #666; margin-top: 4px; }
  .word-count { color: #888; }
  .word-count.ok { color: #4ade80; }
  .word-count.warn { color: #facc15; }
  textarea, input[type=text] { width: 100%; background: #111; border: 1px solid #333; border-radius: 8px; padding: 10px 12px; color: #e8e8e8; font-size: 0.9rem; resize: vertical; outline: none; transition: border-color .15s; }
  textarea:focus, input[type=text]:focus { border-color: #555; }
  textarea { min-height: 90px; font-family: inherit; }
  .field { margin-bottom: 20px; }
  .formats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
  .fmt-label { display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 0.9rem; color: #ccc; }
  .fmt-label input { accent-color: #6366f1; width: 16px; height: 16px; }
  button[type=submit] { width: 100%; padding: 12px; background: #6366f1; color: #fff; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: background .15s; }
  button[type=submit]:hover { background: #4f46e5; }
  button[type=submit]:disabled { background: #333; color: #666; cursor: not-allowed; }
  /* Progress panel */
  #progress { display: none; }
  #log-box { background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; height: 380px; overflow-y: auto; font-family: 'Menlo', 'Monaco', monospace; font-size: 0.78rem; line-height: 1.6; color: #b0b0b0; white-space: pre-wrap; word-break: break-all; margin-bottom: 20px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #444; border-top-color: #6366f1; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #status-line { font-size: 0.85rem; color: #888; margin-bottom: 16px; }
  #links { display: none; }
  .link-btn { display: block; text-align: center; padding: 10px 16px; background: #166534; color: #4ade80; border: 1px solid #15803d; border-radius: 8px; text-decoration: none; font-size: 0.875rem; font-weight: 500; margin-bottom: 10px; transition: background .15s; }
  .link-btn:hover { background: #14532d; }
  .link-btn.drive { background: #1e2a3a; border-color: #1e40af; color: #60a5fa; }
  .link-btn.drive:hover { background: #172033; }
  .error-box { background: #1c0a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: 12px 16px; color: #f87171; font-size: 0.85rem; margin-bottom: 16px; display: none; }
  #new-btn { display: none; width: 100%; padding: 10px; background: #222; color: #ccc; border: 1px solid #333; border-radius: 8px; font-size: 0.875rem; cursor: pointer; transition: background .15s; }
  #new-btn:hover { background: #2a2a2a; }
</style>
</head>
<body>
<div class="card">
  <h1>TikTok Video Generator</h1>
  <p class="subtitle">Generates 3 format variants &mdash; static, walk-and-talk, location-tour</p>

  <form id="form">
    <div class="field">
      <label for="script">Script <span id="wc" class="word-count"></span></label>
      <textarea id="script" name="script" placeholder="Write the spokesperson's dialogue here..." required></textarea>
      <p class="hint">Target 42&ndash;45 words for a ~22 second video</p>
    </div>
    <div class="field">
      <label for="setting">Setting / Location</label>
      <textarea id="setting" name="setting" placeholder="e.g. Modern law office, warm lighting, bookshelves in background..." required></textarea>
    </div>
    <div class="field">
      <label for="actions">Movements / Actions</label>
      <textarea id="actions" name="actions" placeholder="e.g. Walks toward camera, gestures with hands, looks directly at lens..." required></textarea>
    </div>
    <div class="field">
      <label for="slug">Filename slug (optional)</label>
      <input type="text" id="slug" name="slug" placeholder="auto-generated if blank">
    </div>
    <div class="field">
      <label>Formats to generate</label>
      <div class="formats">
        <label class="fmt-label"><input type="checkbox" name="formats" value="static" checked> Static</label>
        <label class="fmt-label"><input type="checkbox" name="formats" value="walk-and-talk" checked> Walk-and-Talk</label>
        <label class="fmt-label"><input type="checkbox" name="formats" value="location-tour" checked> Location-Tour</label>
      </div>
    </div>
    <button type="submit" id="submit-btn">Generate Videos</button>
  </form>

  <div id="progress">
    <div id="status-line"><span class="spinner"></span> Generating videos&hellip;</div>
    <div id="error-box" class="error-box"></div>
    <div id="log-box"></div>
    <div id="links"></div>
    <button id="new-btn" onclick="resetForm()">Generate Another</button>
  </div>
</div>

<script>
const form = document.getElementById('form');
const progress = document.getElementById('progress');
const logBox = document.getElementById('log-box');
const statusLine = document.getElementById('status-line');
const linksDiv = document.getElementById('links');
const errorBox = document.getElementById('error-box');
const newBtn = document.getElementById('new-btn');
const scriptTA = document.getElementById('script');
const wcSpan = document.getElementById('wc');

scriptTA.addEventListener('input', () => {
  const words = scriptTA.value.trim().split(/\\s+/).filter(Boolean).length;
  wcSpan.textContent = words ? `(${words} words)` : '';
  wcSpan.className = 'word-count' + (words >= 38 && words <= 50 ? ' ok' : words > 0 ? ' warn' : '');
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formats = [...document.querySelectorAll('input[name=formats]:checked')].map(c => c.value);
  if (!formats.length) { alert('Select at least one format.'); return; }

  const body = {
    script: document.getElementById('script').value.trim(),
    setting: document.getElementById('setting').value.trim(),
    actions: document.getElementById('actions').value.trim(),
    slug: document.getElementById('slug').value.trim(),
    formats,
  };

  form.style.display = 'none';
  progress.style.display = 'block';
  logBox.textContent = '';
  linksDiv.style.display = 'none';
  linksDiv.innerHTML = '';
  errorBox.style.display = 'none';
  newBtn.style.display = 'none';

  const resp = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const { job_id, error } = await resp.json();

  if (error) {
    showError(error);
    return;
  }

  const es = new EventSource(`/stream/${job_id}`);
  es.onmessage = (event) => {
    if (event.data === '[DONE]') {
      es.close();
      pollStatus(job_id);
      return;
    }
    logBox.textContent += event.data + '\\n';
    logBox.scrollTop = logBox.scrollHeight;
  };
  es.onerror = () => { es.close(); pollStatus(job_id); };
});

async function pollStatus(job_id) {
  const r = await fetch(`/status/${job_id}`);
  const data = await r.json();

  if (data.status === 'done') {
    statusLine.innerHTML = '✓ Done';
    linksDiv.style.display = 'block';
    let html = '';

    // Direct download buttons (always shown if files exist)
    if (data.local_files && data.local_files.length) {
      html += '<p style="font-size:.85rem;color:#888;margin-bottom:12px;">Download videos:</p>';
      html += data.local_files.map(f =>
        `<a class="link-btn" href="/download/${job_id}/${encodeURIComponent(f.filename)}">⬇ ${f.filename} (${f.size_mb} MB)</a>`
      ).join('');
    }

    // Google Drive links (if upload succeeded)
    if (data.drive_links && data.drive_links.length) {
      html += '<p style="font-size:.85rem;color:#888;margin:16px 0 12px;">Also on Google Drive:</p>';
      html += data.drive_links.map(l =>
        `<a class="link-btn drive" href="${l.url}" target="_blank">☁ ${l.filename}</a>`
      ).join('');
    }

    if (!html) {
      html = '<p style="color:#888;font-size:.85rem;">No videos were generated.</p>';
    }
    linksDiv.innerHTML = html;
    newBtn.style.display = 'block';
  } else if (data.status === 'failed') {
    showError(data.error || 'Generation failed');
    newBtn.style.display = 'block';
  } else {
    setTimeout(() => pollStatus(job_id), 2000);
  }
}

function showError(msg) {
  statusLine.textContent = '✗ Failed';
  errorBox.textContent = msg;
  errorBox.style.display = 'block';
  newBtn.style.display = 'block';
}

function resetForm() {
  progress.style.display = 'none';
  form.style.display = 'block';
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return HTML


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    script = (data.get('script') or '').strip()
    setting = (data.get('setting') or '').strip()
    actions = (data.get('actions') or '').strip()
    slug = (data.get('slug') or '').strip()
    formats = data.get('formats') or []

    if not script:
        return jsonify({'error': 'Script is required'}), 400
    if not setting:
        return jsonify({'error': 'Setting is required'}), 400
    if not actions:
        return jsonify({'error': 'Movements/actions are required'}), 400
    valid_formats = {'static', 'walk-and-talk', 'location-tour'}
    formats = [f for f in formats if f in valid_formats]
    if not formats:
        return jsonify({'error': 'Select at least one valid format'}), 400

    if not slug:
        slug = f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    article_data = {
        'title': '',
        'slug': slug,
        'excerpt': '',
        'body_markdown': script,
        'categories': [],
        'keywords': [],
    }

    job_id = str(uuid.uuid4())[:8]
    job = {
        'status': 'running',
        'log_queue': queue.Queue(),
        'drive_links': [],
        'local_files': [],
        'error': None,
    }
    with jobs_lock:
        jobs[job_id] = job

    t = threading.Thread(
        target=run_job,
        args=(job_id, article_data, script, setting, actions, formats),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/stream/<job_id>')
def stream(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    def generate_sse():
        log_q = job['log_queue']
        while True:
            try:
                line = log_q.get(timeout=15)
            except queue.Empty:
                yield 'data: [keepalive]\n\n'
                continue
            if line is None:  # sentinel
                yield 'data: [DONE]\n\n'
                break
            # Escape SSE-sensitive characters
            safe = line.replace('\n', ' ')
            yield f'data: {safe}\n\n'

    return Response(
        stream_with_context(generate_sse()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/status/<job_id>')
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'drive_links': job['drive_links'],
        'local_files': [{'filename': f['filename'], 'size_mb': f['size_mb']} for f in job.get('local_files', [])],
        'error': job['error'],
    })


@app.route('/download/<job_id>/<filename>')
def download(job_id, filename):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    for f in job.get('local_files', []):
        if f['filename'] == filename and os.path.isfile(f['path']):
            return send_file(f['path'], as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
