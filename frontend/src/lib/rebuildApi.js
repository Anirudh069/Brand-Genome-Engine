import { apiPost } from './http';

export async function rebuildProfile() {
  return apiPost('/rebuild/profile');
}

export async function rebuildChunks() {
  return apiPost('/rebuild/chunks');
}

export async function rebuildIndex() {
  return apiPost('/rebuild/index');
}