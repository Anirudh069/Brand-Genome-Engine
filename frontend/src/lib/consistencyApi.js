import { apiPost } from './http';

export async function scoreConsistency(text) {
  return apiPost('/consistency/score', { text });
}
