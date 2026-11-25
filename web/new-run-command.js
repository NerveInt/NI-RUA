import { pick } from './shared/i18n.js';

// Shell command the console shows for starting a new run. The real SO-101 arm (backend.environment `so101_real`)
// starts from its own config (`configs/so101_gateway.json`: deployment file, calibrated observation profile,
// experiment root); every other environment keeps the RoboDojo-era command text unchanged.
const PROMPT_ZH = '调用 robot_capabilities 与 robot_observe，报告看到的场景与机械臂状态；不要移动。';
const PROMPT_EN = 'Call robot_capabilities and robot_observe, then report the observed scene and arm state; do not move.';

export function newRunCommand({ model, observationProfile, environment, category }) {
  const config = environment === 'so101_real' ? 'configs/so101_gateway.json' : 'configs/ni-rua.json';
  const title = pick('<任务名>', '<task-name>');
  const prompt = pick(PROMPT_ZH, PROMPT_EN);
  return ['mamba run -n ni-rua python -m ni_rua.apps.gateway \\', `  --config ${config} \\`,
    `  --model ${model} --observation-profile ${observationProfile} \\`,
    `  --port 8781 --category ${environment === 'so101_real' && category ? category : 'so101'} --title "${title}" \\`,
    `  --prompt "${prompt}"`].join('\n');
}
