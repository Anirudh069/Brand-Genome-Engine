import { API_BASE } from './constants';

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail;
    if (typeof detail === 'string') {
      throw new Error(detail);
    }
    if (Array.isArray(detail)) {
      throw new Error(detail.map((item) => item?.msg || item?.message || 'Request failed').join('; '));
    }
    if (detail && typeof detail === 'object') {
      throw new Error(detail.message || detail.error || 'Request failed');
    }
    throw new Error(body?.message || 'Request failed');
  }
  return body;
}

export async function fetchBenchmarkBrands() {
  const response = await fetch(`${API_BASE}/benchmark/brands`);
  return readJson(response);
}

export async function runBenchmark({ competitorBrandId, metric }) {
  const response = await fetch(`${API_BASE}/benchmark/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      competitor_brand_id: competitorBrandId,
      metric,
    }),
  });
  return readJson(response);
}