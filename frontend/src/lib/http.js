import { API_BASE } from './constants';

// Shared response parser: normalizes FastAPI error shapes into a plain Error
// with a human-readable `.message` and, when present, a structured `.code`
// (e.g. "genome_not_initialized", "index_stale") for callers that branch on it.
async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail;
    let message = 'Request failed.';
    let code = null;
    if (typeof detail === 'string') {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((item) => item?.msg || item?.message || 'Request failed').join('; ');
    } else if (detail && typeof detail === 'object') {
      message = detail.message || detail.error || message;
      code = detail.error || null;
    } else if (body?.message) {
      message = body.message;
    }
    const error = new Error(message);
    error.code = code;
    throw error;
  }
  return body;
}

export async function apiFetch(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  return readJson(response);
}

export async function apiGet(path) {
  return apiFetch(path);
}

export async function apiPost(path, body) {
  return apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}
