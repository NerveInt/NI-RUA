import test from 'node:test';
import assert from 'node:assert/strict';

import { requestJSON } from './api-client.js';
import { createLeaseClient } from './lease-client.js';

test('requestJSON preserves status, error body, and resolves paths', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => ({
    ok: false,
    status: 409,
    statusText: 'Conflict',
    json: async () => ({ error: 'input_owner_conflict', status: 'unknown' }),
  });
  try {
    await assert.rejects(
      () => requestJSON('/v1/control', { method: 'POST' }, { resolvePath: (p) => `/runs/r1${p}` }),
      (error) => error.message === 'input_owner_conflict' && error.status === 409 && error.body.status === 'unknown',
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('requestJSON rejects malformed JSON on HTTP 200 OK instead of returning {}', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => {
      throw new SyntaxError('Unexpected end of JSON input');
    },
  });
  try {
    await assert.rejects(
      () => requestJSON('/v1/control', { method: 'POST' }),
      (error) => error instanceof SyntaxError && error.status === undefined,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('requestJSON falls back to empty body on non-JSON HTTP errors', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 502,
    statusText: 'Bad Gateway',
    json: async () => {
      throw new SyntaxError('Unexpected token <');
    },
  });
  try {
    await assert.rejects(
      () => requestJSON('/v1/run'),
      (error) => error.message === '502 Bad Gateway' && error.status === 502 && Object.keys(error.body).length === 0,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('createLeaseClient reuses request_id across network failures and re-claims when expired', async () => {
  const requests = [];
  let attemptCount = 0;
  const fakeRequestJSON = async (path, options) => {
    const body = JSON.parse(options.body);
    requests.push(body);
    if (body.command === 'claim_input') {
      return { writer_token: `tok-${requests.length}` };
    }
    attemptCount += 1;
    if (attemptCount === 1) {
      throw new Error('network_down'); // status undefined: keep request_id
    }
    if (attemptCount === 2) {
      const expired = new Error('input_owner_claim_required');
      expired.status = 400;
      throw expired;
    }
    return { status: 'ok', request_id: body.request_id };
  };
  const client = createLeaseClient({ clientId: 'c1', requestJSON: fakeRequestJSON });
  await assert.rejects(() => client.sendIdempotent('/v1/control', { command: 'go_home' }, { key: 'go_home' }), /network_down/);
  const result = await client.sendIdempotent('/v1/control', { command: 'go_home' }, { key: 'go_home', reclaimOnExpired: true });
  assert.equal(result.status, 'ok');
  const goHomes = requests.filter((r) => r.command === 'go_home');
  assert.equal(goHomes.length, 3);
  assert.equal(goHomes[0].request_id, goHomes[1].request_id, 'network retry reuses same request_id');
  assert.notEqual(goHomes[1].request_id, goHomes[2].request_id, 'HTTP 400 claim_required clears attempt and re-claims');
});

test('createLeaseClient preserves request_id across HTTP 409 status=unknown responses', async () => {
  const requests = [];
  let attemptCount = 0;
  const fakeRequestJSON = async (path, options) => {
    const body = JSON.parse(options.body);
    requests.push(body);
    if (body.command === 'claim_input') {
      return { writer_token: 'tok-unknown' };
    }
    attemptCount += 1;
    if (attemptCount === 1) {
      const unknownErr = new Error('operator_stop_unconfirmed');
      unknownErr.status = 409;
      unknownErr.body = { status: 'unknown', error: 'operator_stop_unconfirmed' };
      throw unknownErr;
    }
    return { status: 'unknown', replayed: true, request_id: body.request_id };
  };
  const client = createLeaseClient({ clientId: 'c1', requestJSON: fakeRequestJSON });
  await assert.rejects(
    () => client.sendIdempotent('/v1/control', { command: 'go_home' }, { key: 'go_home' }),
    (error) => error.status === 409 && error.body?.status === 'unknown',
  );
  const replayed = await client.sendIdempotent('/v1/control', { command: 'go_home' }, { key: 'go_home' });
  const goHomes = requests.filter((r) => r.command === 'go_home');
  assert.equal(goHomes.length, 2);
  assert.equal(goHomes[0].request_id, goHomes[1].request_id, 'HTTP 409 unknown retry reuses same request_id');
  assert.equal(replayed.request_id, goHomes[0].request_id);
});

test('createLeaseClient preserves request_id across HTTP 5xx errors and clears on deterministic HTTP 4xx', async () => {
  const requests = [];
  let attemptCount = 0;
  const fakeRequestJSON = async (_path, options) => {
    const body = JSON.parse(options.body);
    requests.push(body);
    if (body.command === 'claim_input') {
      return { writer_token: 'tok-5xx' };
    }
    attemptCount += 1;
    if (attemptCount === 1) {
      throw Object.assign(new Error('live_backend_unavailable_do_not_retry_writes'), { status: 503, body: {} });
    }
    if (attemptCount === 2) {
      throw Object.assign(new Error('TypeError: boom'), { status: 500, body: { status: 'unknown' } });
    }
    if (attemptCount === 3) {
      throw Object.assign(new Error('run_not_ready_for_new_task'), { status: 400, body: { error: 'run_not_ready_for_new_task' } });
    }
    return { status: 'running', request_id: body.request_id, thread_id: 't1', turn_id: 'turn-1' };
  };
  const client = createLeaseClient({ clientId: 'c1', requestJSON: fakeRequestJSON });
  await assert.rejects(() => client.sendIdempotent('/v1/control', { command: 'prompt', prompt: 'stack' }, { key: 'task-prompt' }), /503|live_backend/);
  await assert.rejects(() => client.sendIdempotent('/v1/control', { command: 'prompt', prompt: 'stack' }, { key: 'task-prompt' }), /boom/);
  await assert.rejects(() => client.sendIdempotent('/v1/control', { command: 'prompt', prompt: 'stack' }, { key: 'task-prompt' }), /run_not_ready/);
  const ok = await client.sendIdempotent('/v1/control', { command: 'prompt', prompt: 'stack' }, { key: 'task-prompt' });
  const prompts = requests.filter((r) => r.command === 'prompt');
  assert.equal(prompts.length, 4);
  assert.equal(prompts[0].request_id, prompts[1].request_id, 'HTTP 503 retry reuses request_id');
  assert.equal(prompts[1].request_id, prompts[2].request_id, 'HTTP 500 retry reuses request_id');
  assert.notEqual(prompts[2].request_id, prompts[3].request_id, 'HTTP 400 rejection clears cached request_id');
  assert.equal(ok.request_id, prompts[3].request_id);
});
