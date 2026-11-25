// Safety header of the operator pages (/console, /calibration): the header is sticky; when its layout wraps so that
// the E-STOP button sits on a lower row (tablet and phone widths), the rows above it scroll away and the E-STOP row
// stays pinned at the top of the viewport. On one-row layouts nothing collapses. Presentation only: the button, its
// handler and its command are untouched.
//
// collapseHeaderToEstop(header, estop) sets --bar-collapse on the header (px the header may scroll up while sticky)
// and keeps it current on resize and layout changes. Returns a function that recomputes immediately.
export function collapseHeaderToEstop(header, estop, { gap = 6 } = {}) {
  if (!header || !estop) return () => {};
  let frame = 0, last = null;
  const measure = () => {
    frame = 0;
    // Top of the E-STOP inside the header: > the header's own top padding only when it wrapped to a lower row.
    const top = estop.getBoundingClientRect().top - header.getBoundingClientRect().top;
    const padding = Number.parseFloat(getComputedStyle(header).paddingTop) || 0;
    const collapse = top > padding + 8 ? Math.max(0, Math.round(top - gap)) : 0;
    if (collapse !== last) {
      last = collapse;
      header.style.setProperty('--bar-collapse', `${collapse}px`);
      header.dataset.collapsible = collapse > 0 ? 'true' : 'false';
    }
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(measure); };
  new ResizeObserver(schedule).observe(header);
  addEventListener('resize', schedule);
  measure();
  return measure;
}
