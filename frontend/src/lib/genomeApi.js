import { apiGet, apiPost } from './http';

export async function fetchGenome() {
  return apiGet('/genome');
}

export async function initGenome({ designation, mission_core_vision, snippets }) {
  return apiPost('/genome/init', { designation, mission_core_vision, snippets });
}
