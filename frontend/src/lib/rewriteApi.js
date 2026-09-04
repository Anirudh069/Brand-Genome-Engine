import { apiPost } from './http';

export async function runRewrite({ text, topK }) {
  return apiPost('/rewrite', {
    text,
    ...(topK ? { top_k: topK } : {}),
  });
}
