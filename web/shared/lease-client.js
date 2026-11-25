// Shared writer-lease and idempotent request_id control client.

export function createLeaseClient({
  clientId = crypto.randomUUID(),
  requestJSON,
  controlPath = '/v1/control',
  onBusy = null,
  onClaimError = null,
} = {}) {
  let writerToken;
  let leaseClaim = null;
  let leaseClaimedAt = 0;
  let renewing = false;
  let claimRefusedBusy = false;
  const attempts = new Map();

  async function ensureLease({ quiet = false } = {}) {
    if (writerToken) return true;
    leaseClaim ??= requestJSON(controlPath, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 'claim_input', request_id: crypto.randomUUID(), client_id: clientId }),
    }).finally(() => {
      leaseClaim = null;
    });
    try {
      writerToken = (await leaseClaim).writer_token;
      leaseClaimedAt = Date.now();
      claimRefusedBusy = false;
      return true;
    } catch (error) {
      if (String(error.message).startsWith('input_owner_busy_step')) {
        claimRefusedBusy = true;
        if (typeof onBusy === 'function') onBusy(error);
      }
      if (!quiet && typeof onClaimError === 'function') onClaimError(error);
      return false;
    }
  }

  async function renewLease() {
    if (!writerToken || renewing || Date.now() - leaseClaimedAt < 10_000) return false;
    renewing = true;
    try {
      await requestJSON(controlPath, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'renew_input',
          request_id: crypto.randomUUID(),
          client_id: clientId,
          writer_token: writerToken,
        }),
      });
      leaseClaimedAt = Date.now();
      return true;
    } catch (error) {
      if (error.status !== undefined) writerToken = undefined;
      return false;
    } finally {
      renewing = false;
    }
  }

  async function releaseLease() {
    if (!writerToken) return false;
    try {
      await requestJSON(controlPath, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command: 'release_input',
          request_id: crypto.randomUUID(),
          client_id: clientId,
          writer_token: writerToken,
        }),
      });
      writerToken = undefined;
      return true;
    } catch {
      return false;
    }
  }

  async function sendIdempotent(path, fields = {}, { lease = true, key = null, reclaimOnExpired = false } = {}) {
    for (let retry = 0; ; retry += 1) {
      if (lease && !(await ensureLease())) {
        throw Object.assign(new Error('input_owner_conflict'), { status: 409, body: {} });
      }
      const signature = JSON.stringify(fields);
      const attemptKey = key || `${path}:${signature}`;
      let attempt = attempts.get(attemptKey);
      if (!attempt || attempt.signature !== signature) {
        attempt = { signature, request_id: crypto.randomUUID() };
        attempts.set(attemptKey, attempt);
      }
      const body = {
        request_id: attempt.request_id,
        ...fields,
        ...(lease ? { client_id: clientId, writer_token: writerToken } : {}),
      };
      try {
        const result = await requestJSON(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        attempts.delete(attemptKey);
        return result;
      } catch (error) {
        if (error.status === undefined || error.status >= 500 || error.body?.status === 'unknown') throw error;
        attempts.delete(attemptKey);
        if (lease && reclaimOnExpired && error.message === 'input_owner_claim_required' && retry === 0) {
          writerToken = undefined;
          continue;
        }
        throw error;
      }
    }
  }

  return {
    clientId,
    get token() { return writerToken; },
    set token(value) { writerToken = value; },
    get claimedAt() { return leaseClaimedAt; },
    set claimedAt(value) { leaseClaimedAt = value; },
    get claimRefusedBusy() { return claimRefusedBusy; },
    set claimRefusedBusy(value) { claimRefusedBusy = Boolean(value); },
    ensureLease,
    renewLease,
    releaseLease,
    sendIdempotent,
    clearToken() { writerToken = undefined; },
  };
}
