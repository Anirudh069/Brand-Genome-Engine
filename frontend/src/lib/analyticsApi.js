import { apiGet } from './http';

export async function fetchAnalytics() {
  return apiGet('/analytics');
}
