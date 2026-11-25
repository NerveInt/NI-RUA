// Shared JSON fetch client preserving HTTP status and typed error payloads.

export async function requestJSON(path, options = {}, { resolvePath = (target) => target, beforeRequest = null, cache = undefined } = {}) {
  if (typeof beforeRequest === 'function') beforeRequest(path, options);
  const fetchOptions = cache === undefined || options?.cache !== undefined ? options : { cache, ...options };
  const response = await fetch(resolvePath(path), fetchOptions);
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      body = {};
    }
    const message = body?.error || body?.reason || `${response.status} ${response.statusText}`;
    const error = new Error(message);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return await response.json();
}
