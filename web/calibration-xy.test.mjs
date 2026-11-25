import test from 'node:test';
import assert from 'node:assert/strict';
import { imagePixel, xyThresholdFields } from './calibration-xy.js';

test('XY threshold form uses server defaults without an independent numeric fallback', () => {
  const fields = xyThresholdFields({ max_xy_error_m: .004, max_reprojection_error_px: 2.5, min_parallax_deg: 1.2 });
  assert.deepEqual(fields.map(row => row[2]), [4, 2.5, 1.2]);
  assert.deepEqual(xyThresholdFields().map(row => row[2]), ['', '', '']);
});

test('XY clicks retain original image coordinates under CSS resizing', () => {
  const image = { naturalWidth: 640, naturalHeight: 480,
    getBoundingClientRect: () => ({ left: 20, top: 30, width: 320, height: 240 }) };
  assert.deepEqual(imagePixel({ clientX: 180, clientY: 150 }, image), [320, 240]);
  assert.equal(imagePixel({ clientX: 340, clientY: 150 }, image), null);
  assert.equal(imagePixel({ clientX: 10, clientY: 150 }, image), null);
  assert.equal(imagePixel({ clientX: 180, clientY: 150 }, { ...image, naturalWidth: 0 }), null);
});
