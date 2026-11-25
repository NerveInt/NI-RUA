import './workspace-navigation.css';

// Root is the live hardware entry. Explicit /index.html and library run URLs
// retain their review workspace; loading a saved run never opens robot controls.
export function updateWorkspaceNavigation(run) {
  const standalone = !location.pathname.startsWith('/runs/');
  const live = run && run.library?.live !== false && run.view_mode !== 'replay' && run.status !== 'replay' && (run.agent_status ?? run.codex_status) !== 'replay';
  const so101 = run?.capabilities?.robot_model_id === 'so101';
  const navigation = document.getElementById('workspace-nav');
  if (navigation) navigation.hidden = !(standalone && live && so101);
  if (location.pathname === '/' && standalone && live && so101) {
    const lang = new URLSearchParams(location.search || '').get('lang');
    location.replace(lang ? `/console?lang=${encodeURIComponent(lang)}` : '/console');
    return true;
  }
  return false;
}
