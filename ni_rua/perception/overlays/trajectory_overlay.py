"""Recorded commands or measured robot FK on calibrated, original observed RGB.

What is drawn comes from the run's ``EmbodimentProfile`` (``profile``; None = ``ARX_X5_PROFILE``): one colour per
profile arm in ``profile.arms`` order (``ARM_COLORS``; X5 left blue, right orange), the stored path of the TCP link
(``profile.tcp_link``: X5 link6, SO-101 gripper_frame_link), text labels on the links ``labelled_links`` names and the
grasp reference of ``pose_axis_overlay.grasp_frame``.
"""
from copy import deepcopy
import json

from ...core.contracts import IDENTITY

import cv2
import numpy as np

from ...robot.model.geometry import project, transform

CAMERA_SOURCES = {'fixed_calibration', 'robot_kinematics', 'measurement', 'estimate'}
#: Skeleton colours (BGR) and their legend names, by the arm's index in ``profile.arms``.
ARM_COLORS = ((255, 180, 0), (40, 90, 255), (90, 220, 90), (220, 90, 220))
ARM_COLOR_NAMES = ('Blue', 'Orange', 'Green', 'Magenta')
RIBBON_PALETTES = (
    ((145, 95, 0), (255, 215, 45)),
    ((28, 52, 155), (45, 115, 255)),
    ((40, 120, 40), (110, 235, 110)),
    ((120, 40, 120), (235, 110, 235)),
)
TRAJECTORY_LABELS = {
    'candidate_command_trajectory_not_measurement': ('CANDIDATE', 'CURRENT RGB + hypothetical robot joint frames'),
    'recorded_planned_commands_not_measurement': ('RECORDED PLAN', 'START RGB + recorded commanded joint frames'),
    'measured_action_trajectory': ('MEASURED', 'START RGB + measured robot FK; scene may have changed'),
    'unexecuted_arm_candidate_not_synchronized': ('LOCAL ARM PLAN', 'START RGB + local arm plan; future timing unknown'),
}


def labelled_links(profile, joints, arm):
    """Links whose projected origin gets an ``<arm initial>:<link>`` label: the child link of the arm's third arm
    joint, the TCP link and the child links of its jaw joints, looked up in the trajectory's ``robot_model`` joints
    (unnamed joints label only the TCP link). X5: link3, link6, link7, link8; SO-101: lower_arm_link,
    gripper_frame_link, moving_jaw_so101_v1_link."""
    children = {joint.get('name'): joint.get('child') for joint in joints if joint.get('name')}
    arm_joints = list(profile.arm_joint_names.get(arm, ()))
    named = ([arm_joints[2]] if len(arm_joints) > 2 else []) + list(profile.jaw_joint_names.get(arm, ()))
    return {profile.tcp_link} | {children[name] for name in named if name in children}


def _keyframe_alphas(count):
    """为单面板起/中/终关键姿态分配递增不透明度（早步半透明残影，末步实线高亮）。"""
    if count <= 1:
        return [1.0]
    if count == 2:
        return [0.40, 1.0]
    return [0.32, 0.58, 1.0]


def _blend_color(start_bgr, end_bgr, ratio):
    t = float(np.clip(ratio, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(start_bgr, end_bgr))


def render_trajectory_overlay(camera, image, trajectory, *, native_steps=None, profile=None):
    """在单张原始校准 RGB 底图（如 640×480）上合成连续末端轨迹色带与半透明关键姿态骨架。

    所有关键步均投影于同一张起始相机位姿图像上，输出分辨率与原图严格一致（不再水平拼接三联图）。
    ``profile``: the run's EmbodimentProfile (None = ARX_X5_PROFILE); a sample arm outside ``profile.arms`` is
    rejected (``trajectory_overlay_arm_not_in_embodiment``).
    """
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE as profile
    if camera.get('pose_source') not in CAMERA_SOURCES:
        raise ValueError('trajectory_overlay_requires_deployable_camera_calibration')
    source = trajectory.get('source_kind')
    if source not in TRAJECTORY_LABELS:
        raise ValueError('trajectory_overlay_requires_supported_trajectory')
    if trajectory.get('frame_id') != camera['reference_frame_id']:
        raise ValueError('trajectory_overlay_frame_mismatch')
    observation, frame = camera['observation'], camera['frame']
    if observation.get('world_kind') != 'observed' or frame.get('pixel_transform', 'identity') != 'identity':
        raise ValueError('trajectory_overlay_requires_original_observed_rgb')
    width, height = camera['width'], camera['height']
    if image.shape != (height, width, 3) or image.dtype != np.uint8:
        raise ValueError('trajectory_overlay_image_size_mismatch')
    transform(camera['T'])
    samples = trajectory.get('samples', [])
    by_step = {s['native_step']: s for s in samples}
    if not samples:
        raise ValueError('trajectory_overlay_empty_path')
    if native_steps is None:
        native_steps = list(dict.fromkeys([samples[0]['native_step'], samples[len(samples)//2]['native_step'], samples[-1]['native_step']]))
    if (not isinstance(native_steps, list) or not 1 <= len(native_steps) <= 3
            or any(type(i) is not int or i not in by_step for i in native_steps)):
        raise ValueError('trajectory_overlay_requires_one_to_three_existing_steps')
    joints = trajectory.get('robot_model', {}).get('joints', [])
    colors = {arm: ARM_COLORS[index % len(ARM_COLORS)] for index, arm in enumerate(profile.arms)}
    ribbon_palettes = {arm: RIBBON_PALETTES[index % len(RIBBON_PALETTES)] for index, arm in enumerate(profile.arms)}
    if any(arm not in colors for sample in samples for arm in sample.get('arms', {})):
        raise ValueError('trajectory_overlay_arm_not_in_embodiment')
    tcp = profile.tcp_link
    joint_names = trajectory.get('joint_names') if isinstance(trajectory.get('joint_names'), dict) else {}
    canvas = image.copy()
    path_counts = {}
    total_segments = max(1, len(samples) - 1)
    for arm, color in colors.items():
        # Every stored path sample is projected, without resizing the RGB or
        # inventing intermediate measurements. Lines only connect samples.
        positions = [sample.get('arms', {}).get(arm, {}).get('link_positions_m', {}).get(tcp) for sample in samples]
        previous = None
        path_counts[arm] = sum(point is not None for point in positions)
        c_start, c_end = ribbon_palettes[arm]
        sample_idx = 0
        for start in range(0, len(positions), 2000):
            chunk = positions[start:start+2000]
            valid = [point for point in chunk if point is not None]
            if not valid:
                previous = None
                sample_idx += len(chunk)
                continue
            pixels, depth, _ = project(camera, valid)
            iterator = iter(zip(pixels, depth))
            for point in chunk:
                if point is None:
                    previous = None
                    sample_idx += 1
                    continue
                pixel, z = next(iterator)
                if not np.isfinite(pixel).all():  # no image location (beyond the lens model range): break the path
                    previous = None
                    sample_idx += 1
                    continue
                pixel = tuple(np.clip(np.rint(pixel), -100000, 100000).astype(int))
                if previous is not None and z > 0 and previous[1] > 0:
                    visible, a, b = cv2.clipLine((0, 0, width, height), previous[0], pixel)
                    if visible:
                        seg_color = _blend_color(c_start, c_end, sample_idx / total_segments)
                        cv2.line(canvas, a, b, seg_color, 2, cv2.LINE_AA)
                previous = pixel, z
                sample_idx += 1
    from .pose_axis_overlay import (
        draw_grasp_center_path,
        draw_object_long_axis_overlay,
        draw_pose_axis_frame,
    )
    draw_grasp_center_path(canvas, camera, samples, width, height, profile=profile, joint_names=joint_names)
    targets_3d = trajectory.get('grasp_3d_targets')
    if targets_3d:
        draw_object_long_axis_overlay(canvas, camera, targets_3d, width, height)
    labels = {arm: labelled_links(profile, joints, arm) for arm in colors}
    ribbon_label = f'{tcp}+pad' if profile.gripper.grasp_reference == 'tcp_offset' else tcp
    legend = ' | '.join(f'{ARM_COLOR_NAMES[index % len(ARM_COLOR_NAMES)]}: {arm}' for index, arm in enumerate(colors))
    legend += f' | Ghost->Solid: steps | Ribbon: {ribbon_label}'
    details = []
    alphas = _keyframe_alphas(len(native_steps))
    for idx, (step, alpha) in enumerate(zip(native_steps, alphas)):
        is_final = (idx == len(native_steps) - 1)
        layer = canvas if alpha >= 0.999 else canvas.copy()
        record = dict(native_step=step, arms={})
        for arm, arm_data in by_step[step]['arms'].items():
            points = arm_data.get('link_positions_m', {})
            if not points:
                continue
            names = list(points)
            pixels, depth, _ = project(camera, [points[n] for n in names])
            # A link without an image location (beyond the lens model range) is neither marked nor connected.
            projected = {name: (tuple(np.clip(np.rint(pixel), -100000, 100000).astype(int)), z)
                         for name, pixel, z in zip(names, pixels, depth) if np.isfinite(pixel).all()}
            line_thickness = 3 if is_final else 2
            node_radius = 4 if is_final else 3
            for joint in joints:
                parent, child = joint['parent'], joint['child']
                if parent not in projected or child not in projected:
                    continue
                a, za = projected[parent]; b, zb = projected[child]
                if za <= 0 or zb <= 0:
                    continue
                visible, a, b = cv2.clipLine((0, 0, width, height), a, b)
                if visible:
                    cv2.line(layer, a, b, colors[arm], line_thickness, cv2.LINE_AA)
            for name, (pixel, z) in projected.items():
                if z > 0 and 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    cv2.circle(layer, pixel, node_radius, colors[arm], -1, cv2.LINE_AA)
                    if is_final and name in labels[arm]:
                        tag = f'{arm[0]}:{name}(+{step})' if name == tcp and len(native_steps) > 1 else f'{arm[0]}:{name}'
                        cv2.putText(layer, tag, pixel, cv2.FONT_HERSHEY_SIMPLEX, .35, colors[arm], 1, cv2.LINE_AA)
                    elif not is_final and name == tcp:
                        cv2.putText(layer, f'+{step}', pixel, cv2.FONT_HERSHEY_SIMPLEX, .33, colors[arm], 1, cv2.LINE_AA)
            draw_pose_axis_frame(layer, camera, arm, arm_data, colors[arm], width, height, profile=profile,
                                 joint_names=joint_names.get(arm))
            record['arms'][arm] = dict(phase=arm_data.get('phase'), projected_link_count=len(points))
        execution = by_step[step].get('execution_state', '')
        if execution:
            record['execution_state'] = execution
        if alpha < 0.999 and record['arms']:
            canvas = cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0.0)
        details.append(record)
    if not any(p['arms'] for p in details):
        raise ValueError('trajectory_overlay_link_geometry_unavailable')
    label, caption = TRAJECTORY_LABELS[source]
    axis = 'boundary' if source == 'measured_action_trajectory' else 'row' if source == 'unexecuted_arm_candidate_not_synchronized' else 'native'
    terminal_execution = by_step[native_steps[-1]].get('execution_state', '')
    display_execution = {'completed_command': 'completed',
        'not_recorded_completed': 'not completed / may be partial',
        'initial_measurement': 'measured start', 'not_executed': 'not executed'}.get(terminal_execution, terminal_execution)
    step_text = '/'.join(f'+{s}' for s in native_steps)
    heading = f'{label} | {axis} {step_text}' + (f' | {display_execution}' if display_execution else '')
    name = trajectory.get('label', '')
    # 单面板不透明信息条，保持原图 640×480 像素坐标与安全标注边距不变
    cv2.rectangle(canvas, (0, 0), (width-1, 56), (18, 18, 18), -1)
    cv2.rectangle(canvas, (0, height-38), (width-1, height-1), (18, 18, 18), -1)
    for y, message, scale in ((18, heading, .49), (35, name, .43),
            (51, legend, .41),
            (height-23, caption, .43),
            (height-7, 'No object geometry / occlusion / collision proof', .43)):
        text_width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width > width-12:
            scale *= (width-12)/text_width
        cv2.putText(canvas, message, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, dict(
        observation_id=observation['observation_id'], camera_id=frame['camera_id'],
        source_kind='candidate_fk_projected_on_permitted_rgb' if source == 'candidate_command_trajectory_not_measurement' else source+'_on_execution_start_rgb',
        camera_pose_source=camera['pose_source'],
        original_rgb_ref=deepcopy(frame['rgb_ref']), frame_id=camera['reference_frame_id'],
        panels=details, panel_width=width, panel_height=height, path_sample_counts=path_counts,
        executed_native_steps=trajectory.get('executed_native_steps', 0), physics_advanced=False,
        limitations=('Unexecuted candidate geometry; future execution replans and may differ. ' if source == 'candidate_command_trajectory_not_measurement'
                     else 'Historical execution-start RGB and camera pose; later scene changes are not reconstructed. ')+
                    'Joint-frame skeleton has no volume. Path lines connect stored samples, not observed swept geometry. '
                    'Image overlap is not 3D collision; occlusion, held objects and task geometry are unknown.')


def deployable_camera(store, identity, observation_id, camera_id, trajectory):
    """Use static camera calibration and measured encoders, never pose samples.

    The simulation adapter supplies calibration as a hardware setup substitute;
    deployment must supply measured intrinsics/mount/extrinsics in the same frame.
    Dynamic simulator camera/base/link transforms and their residuals are ignored.
    """
    from ...robot.model.geometry import array, fk
    observation = store.read('observation', observation_id)
    if any(observation.get(key) != identity[key] for key in IDENTITY):
        raise ValueError('stale_environment_identity')

    def read_json(reference):
        artifact, raw = store.resolve_artifact(reference, allowed_scopes=('actor',))
        if artifact['mime_type'] != 'application/json':
            raise ValueError('trajectory_overlay_requires_actor_json')
        return json.loads(raw)
    frame = next((f for f in observation['cameras'] if f['camera_id'] == camera_id), None)
    if frame is None or not frame.get('calibration_ref'):
        raise ValueError('trajectory_overlay_camera_calibration_unavailable')
    config = read_json(frame['calibration_ref'])
    calibration = config.get('cameras', {}).get(camera_id, config)
    reference = config.get('reference_frame_id')
    if not reference:
        raise ValueError('trajectory_overlay_missing_calibrated_frame')
    if reference != trajectory['frame_id']:
        raise ValueError('trajectory_overlay_frame_mismatch')
    width, height = frame['width'], frame['height']
    if list(calibration.get('image_size', [])) != [width, height]:
        raise ValueError('calibration_image_size_or_crop_mismatch')
    k = array(calibration.get('K'), (3, 3))
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('invalid_camera_intrinsics')
    distortion = calibration.get('distortion')
    if distortion is None:
        raise ValueError('missing_distortion_calibration')
    if isinstance(distortion, dict):
        if distortion.get('model') not in ('opencv', 'opencv_plumb_bob'):
            raise ValueError('unsupported_distortion_model')
        distortion = distortion['coefficients']
    if calibration.get('distortion_model') not in (None, 'pinhole_no_distortion', 'opencv', 'opencv_plumb_bob'):
        raise ValueError('unsupported_distortion_model')
    arm = calibration.get('arm')
    if arm:
        state = read_json(observation['robot_state_ref'])
        if state.get('sim_step') != observation.get('sim_step') or state.get('step_domain') != observation.get('step_domain'):
            raise ValueError('unsynchronized_observation')
        measured = state['arms'][arm]
        jaw = measured.get('gripper') or {}
        if 'position_m' in jaw:
            if measured.get('source_kind') != 'measurement' or jaw.get('position_source_kind') != 'measurement':
                raise ValueError('trajectory_overlay_requires_measured_joints')
            joint_names = list(measured['joint_names']) + list(jaw['joint_names'])
            q_rad = list(measured['q_rad']) + list(jaw['position_m'])
        else:
            if measured.get('source_kind') != 'measurement' or jaw.get('position_source_kind', 'measurement') != 'measurement':
                raise ValueError('trajectory_overlay_requires_measured_joints')
            joint_names = list(measured['joint_names'])
            q_rad = list(measured['q_rad'])
        initial = trajectory.get('camera_start_q_rad', {}).get(arm, trajectory['samples'][0]['arms'].get(arm, {}).get('q_rad'))
        measured_q = measured['q_rad'][:5] if (initial is not None and len(initial) == 5 and len(measured['q_rad']) == 6) else measured['q_rad']
        if initial is None or len(measured_q) != len(initial) or not np.allclose(measured_q, initial, atol=1e-8, rtol=0):
            raise ValueError('trajectory_overlay_measurement_mismatch')
        raw_model = trajectory.get('robot_model') or read_json(observation['robot_model_ref'])
        base_T = (trajectory.get('T_world_bases') or {}).get(arm, np.eye(4).tolist())
        model = {**raw_model, 'T_world_base': base_T}
        poses = fk(model, dict(source_kind='measurement',
            joint_names=joint_names,
            q_rad=q_rad))
        pose = poses[calibration['link_name']] @ transform(calibration['T_link_camera'])
        source = 'robot_kinematics'
    else:
        pose = transform(calibration.get('T_world_camera', calibration.get('T_base_camera')))
        source = 'fixed_calibration'
    if frame.get('sim_step') != observation.get('sim_step') or frame.get('step_domain') != observation.get('step_domain'):
        raise ValueError('unsynchronized_observation')
    return dict(K=k, D=array(distortion), T=pose, width=width, height=height,
        frame=frame, observation=observation, reference_frame_id=reference, pose_source=source)
