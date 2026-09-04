import { apiGet, apiPost } from './http';

export async function fetchBenchmarkBrands() {
  return apiGet('/benchmark/brands');
}

export async function runBenchmark({ competitorBrandId, metric }) {
  return apiPost('/benchmark/run', {
    competitor_brand_id: competitorBrandId,
    metric,
  });
}