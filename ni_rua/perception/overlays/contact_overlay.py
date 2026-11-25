"""Project disclosed gripper contact references onto an original observed RGB image.

Inputs are a GeometryTools.camera result and its actor-permitted original image.
There is no simulator, object detector, scene query or contact inference here.

The gripper comes from the run's ``EmbodimentProfile`` (``profile``; None = ``ARX_X5_PROFILE``):

* a gripper with ``contact_sections`` (ARX-X5): the measured link6 pose plus the named pad_center / finger_middle /
  fingertip sections (``grasp_contact``) and the caller's ``contact_region`` / ``contact_depth_m`` selection;
* a ``jaw_fingertip_midpoint`` gripper without contact sections (SO-101): the measured TCP pose
  (gripper_frame_link) and the measured jaw angle (rad) give the TCP, the CAD inner fixed and moving fingertips and
  their midpoint, the selected (only) reference; ``selections`` are rejected
  (``contact_overlay_contact_selection_unavailable``), ``targets`` request that midpoint.
"""

from copy import deepcopy

import cv2
import numpy as np

from ...robot.model.geometry import array, project, transform
from ...robot.model.grasp_contact import CONTACT_DEPTH_PRESETS_M, CONTACT_FIELDS, resolve_contact_spec
from ...robot.model.roboprobe_pose import quat_to_matrix
from .pose_axis_overlay import tcp_axis


COLORS = {'pad_center': (30, 220, 250), 'finger_middle': (90, 210, 70),
          'fingertip': (240, 170, 30), 'selected': (255, 255, 255), 'requested': (210, 60, 240),
          'tcp': (30, 220, 250), 'fixed_tip': (90, 210, 70), 'moving_tip': (240, 170, 30)}
#: Length (m) of the drawn approach arrows from the selected reference.
APPROACH_ARROW_M = .06


def _vector(value, size, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size
            or any(isinstance(v, (bool, np.bool_)) for v in value)):
        raise ValueError('invalid_contact_overlay_' + name)
    try:
        return array(value, (size,))
    except (ValueError, TypeError) as error:
        raise ValueError('invalid_contact_overlay_' + name) from error


def render_contact_overlay(camera, image_bgr, *, arms=None, selections=None, targets=None, profile=None):
    """Return ``(new_bgr_image, metadata)``; leave all inputs unchanged.

    ``arms``: unique arms of ``profile.arms`` to draw (default: all of them). ``selections`` maps arms to
    contact_region OR contact_depth_m (contact-section grippers only). ``targets`` maps arms to caller-requested
    selected-reference ``position_m`` (m, frame of the calibration) and optional ``approach_axis_world``. Target
    positions are contact references, not the TCP. Current references derive only from the camera observation's
    measured TCP pose (and, for a jaw-angle gripper, its measured jaw angle).
    """
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE as profile
    arms = list(profile.arms) if arms is None else arms
    if (not isinstance(arms, (tuple, list)) or not 1 <= len(arms) <= len(profile.arms)
            or any(not isinstance(arm, str) or arm not in profile.arms for arm in arms) or len(set(arms)) != len(arms)):
        raise ValueError('contact_overlay_requires_unique_' + '_'.join(profile.arms) + '_arms')
    if profile.gripper.contact_sections is None:
        return _render_fingertip_overlay(camera, image_bgr, arms=arms, selections=selections, targets=targets,
                                         profile=profile)
    selections = {} if selections is None else selections
    targets = {} if targets is None else targets
    if any(not isinstance(value, dict) or set(value) - set(arms) for value in (selections, targets)):
        raise ValueError('contact_overlay_options_require_selected_arms')
    observation, frame, state, width, height, reference_frame = _checked_camera(camera, image_bgr)
    step, domain = observation.get('sim_step'), observation.get('step_domain')
    output = image_bgr.copy()
    points, lines, result_arms = [], [], {}
    if profile.robot_model_id == 'arx-x5':
        contact_presets = CONTACT_DEPTH_PRESETS_M
        resolve_spec = resolve_contact_spec
        measured_source = 'derived_from_measured_link6'
        measured_pos_key = 'measured_link6_position_m'
        semantics_text = 'actual labels use the synchronized measured link6 pose; requested labels use only caller inputs; selected references lie between finger faces, not on a measured object'
    else:
        from ...robot.embodiments.libero_franka_panda import grasp_contact as panda_contact

        contact_presets = panda_contact.CONTACT_DEPTH_PRESETS_M
        resolve_spec = panda_contact.resolve_contact_spec
        measured_source = f'derived_from_measured_{profile.tcp_link}'
        measured_pos_key = f'measured_{profile.tcp_link}_position_m'
        semantics_text = f'actual labels use the synchronized measured {profile.tcp_link} pose; requested labels use only caller inputs; selected references lie between finger faces, not on a measured object'

    def add(arm, name, position, source, color, label=None, marker='circle'):
        index = len(points)
        points.append(dict(arm=arm, reference=name, position_m=position.tolist(), source_kind=source,
                           label=label, marker=marker, color_bgr=color))
        return index

    for arm in arms:
        selection = selections.get(arm, {})
        if not isinstance(selection, dict) or set(selection) - CONTACT_FIELDS:
            raise ValueError('contact_overlay_invalid_contact_selection')
        spec = resolve_spec(selection)
        origin, rotation = _measured_tcp(state, arm, reference_frame, profile)
        # Contact sections lie along the TCP approach axis (X5 link6 +X, Panda gripper0_grip_site +Z).
        axis = tcp_axis(rotation, profile.gripper.approach_axis_local)
        prefix = arm[:1].upper()
        references = {}
        for region in contact_presets:
            reference = resolve_spec({'contact_region': region})
            position = origin + reference['offset_m'] * axis
            index = add(arm, region, position, measured_source, COLORS[region],
                        f'{prefix} actual ' + {'pad_center': 'pad', 'finger_middle': 'middle', 'fingertip': 'tip'}[region])
            references[region] = dict(**reference, position_m=position.tolist(), point_index=index)
        selected = origin + spec['offset_m'] * axis
        current_index = add(arm, 'selected', selected, measured_source, COLORS['selected'],
                            f'{prefix} actual selected ({spec["contact_region"]})', 'ring')
        current_end = add(arm, 'actual_approach_end', selected + APPROACH_ARROW_M * axis,
                          measured_source, COLORS['selected'])
        lines.append(dict(arm=arm, kind='actual_approach', start=current_index, end=current_end,
                          source_kind=measured_source, color_bgr=COLORS['selected']))
        entry = {
            'selected_contact': spec,
            measured_pos_key: origin.tolist(),
            'actual_references': references,
            'actual_selected_position_m': selected.tolist(),
            'actual_selected_point_index': current_index,
            'actual_approach_axis_world': axis.tolist(),
        }
        _add_target(entry, arm, targets, selected, prefix, add, lines)
        result_arms[arm] = entry
    _draw(output, camera, points, lines, result_arms, arms, width, height)
    metadata = _metadata(camera, observation, frame, reference_frame, step, domain, width, height, result_arms,
        points, lines,
        semantics=semantics_text)
    return output, metadata


def _checked_camera(camera, image_bgr):
    """Validated ``(observation, frame, state, width, height, reference_frame_id)`` of a camera bundle."""
    observation, frame, state = camera['observation'], camera['frame'], camera['state']
    if observation.get('world_kind') != 'observed':
        raise ValueError('contact_overlay_requires_observed_image')
    step, domain = observation.get('sim_step'), observation.get('step_domain')
    if (step is None or not domain or any(value.get('sim_step') != step or value.get('step_domain') != domain
                                       for value in (frame, state))):
        raise ValueError('contact_overlay_requires_synchronized_proprioception')
    if frame.get('pixel_transform', 'identity') != 'identity':
        raise ValueError('contact_overlay_requires_original_image_coordinates')
    width, height = camera['width'], camera['height']
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.shape != (height, width, 3)):
        raise ValueError('contact_overlay_original_rgb_size_mismatch')
    transform(camera['T'])
    intrinsic = array(camera['K'], (3, 3))
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError('contact_overlay_invalid_intrinsics')
    return observation, frame, state, width, height, camera['reference_frame_id']


def _measured_tcp(state, arm, reference_frame, profile):
    """Measured TCP ``(origin m, rotation 3x3)`` of ``arm`` in ``reference_frame``, with the profile's TCP id."""
    pose = state.get('arms', {}).get(arm, {}).get('ee_pose', {})
    if (pose.get('source_kind') != 'measurement' or pose.get('frame_id') != reference_frame
            or pose.get('tcp_config_id') != profile.tcp_config_id):
        raise ValueError(f'contact_overlay_requires_measured_{profile.tcp_link}:' + arm)
    origin = _vector(pose.get('position_m'), 3, 'measured_position')
    xyzw = _vector(pose.get('quaternion_xyzw'), 4, 'measured_quaternion')
    if abs(float(np.linalg.norm(xyzw)) - 1.) > 1e-4:
        raise ValueError('contact_overlay_requires_normalized_xyzw_quaternion')
    return origin, quat_to_matrix(np.r_[xyzw[3], xyzw[:3]])


def _add_target(entry, arm, targets, selected, prefix, add, lines):
    """Caller-requested selected reference (and approach) of ``arm``, kept apart from the measured one."""
    if arm not in targets:
        return
    target = targets[arm]
    if (not isinstance(target, dict) or 'position_m' not in target
            or set(target) - {'position_m', 'approach_axis_world'}):
        raise ValueError('contact_overlay_invalid_requested_target')
    wanted = _vector(target['position_m'], 3, 'requested_position')
    target_index = add(arm, 'requested_selected', wanted, 'caller_request', COLORS['requested'],
                       f'{prefix} REQUESTED selected', 'cross')
    entry.update(requested_selected_position_m=wanted.tolist(), requested_point_index=target_index,
                 selected_position_error_m=float(np.linalg.norm(selected-wanted)),
                 target_semantics='caller requested selected contact reference; not a measured object or arrival claim')
    if 'approach_axis_world' in target:
        wanted_axis = _vector(target['approach_axis_world'], 3, 'requested_approach')
        norm = float(np.linalg.norm(wanted_axis))
        if norm <= 1e-9:
            raise ValueError('contact_overlay_requested_approach_must_be_nonzero')
        wanted_axis = wanted_axis / norm
        target_end = add(arm, 'requested_approach_end', wanted + APPROACH_ARROW_M*wanted_axis,
                         'caller_request', COLORS['requested'])
        lines.append(dict(arm=arm, kind='requested_approach', start=target_index, end=target_end,
                          source_kind='caller_request', color_bgr=COLORS['requested']))
        entry['requested_approach_axis_world'] = wanted_axis.tolist()


def _draw(output, camera, points, lines, result_arms, arms, width, height):
    """Project ``points`` (adds pixel_xy, depth_m, in_image), draw arrows, markers, labels and the legend."""
    pixels, depths, in_image = project(camera, [point['position_m'] for point in points])
    for index, point in enumerate(points):
        pixel = pixels[index]
        finite = bool(np.isfinite(pixel).all())
        point.update(pixel_xy=pixel.tolist() if finite else None, depth_m=float(depths[index]),
                     in_front_of_camera=bool(depths[index] > 0), in_image=bool(in_image[index] and finite),
                     occlusion='not_evaluated')
    for line in lines:
        start, end = (points[line[key]] for key in ('start', 'end'))
        line['drawn'] = start['in_image'] and end['in_image']
        if line['drawn']:
            cv2.arrowedLine(output, tuple(np.rint(start['pixel_xy']).astype(int)),
                            tuple(np.rint(end['pixel_xy']).astype(int)), line['color_bgr'], 2,
                            line_type=cv2.LINE_AA, tipLength=.25)
    label_count = {}
    # Put two arm labels on opposite outer sides of their measured points.
    # Their actual references are only millimetres apart; overlapping long
    # "selected" labels would otherwise hide which arm owns each marker.
    ordered = sorted(arms, key=lambda arm: points[result_arms[arm]['actual_selected_point_index']]['pixel_xy'][0]
                     if points[result_arms[arm]['actual_selected_point_index']]['pixel_xy'] is not None else 0.)
    label_side = {arm: (-1 if len(ordered) == 2 and index == 0 else 1) for index, arm in enumerate(ordered)}
    for point in points:
        if not point['in_image'] or point['label'] is None:
            continue
        pixel = tuple(np.rint(point['pixel_xy']).astype(int))
        color = point['color_bgr']
        if point['marker'] == 'cross':
            cv2.drawMarker(output, pixel, color, cv2.MARKER_TILTED_CROSS, 15, 2, cv2.LINE_AA)
        else:
            cv2.circle(output, pixel, 9 if point['marker'] == 'ring' else 4, color, 2, cv2.LINE_AA)
        count = label_count.get(point['arm'], 0)
        label_count[point['arm']] = count+1
        y = max(12, min(height-3, pixel[1]-28+count*15))
        text_size = cv2.getTextSize(point['label'], cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0]
        x = pixel[0]+14 if label_side[point['arm']] > 0 else pixel[0]-14-text_size[0]
        x = max(2, min(x, width-text_size[0]-2))
        cv2.putText(output, point['label'], (x,y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(output, point['label'], (x,y), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    for index, legend in enumerate(('ACTUAL: measured robot pose | REQUESTED: caller target',
                                     'Contact references do not establish object contact or retention')):
        position = (5, max(12,height-23+index*15))
        cv2.putText(output, legend, position, cv2.FONT_HERSHEY_SIMPLEX, .38, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(output, legend, position, cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)


def _metadata(camera, observation, frame, reference_frame, step, domain, width, height, result_arms, points, lines,
              *, semantics):
    return dict(world_kind='observed', observation_id=observation['observation_id'],
        camera_id=frame['camera_id'], frame_id=reference_frame, image_frame_id=frame.get('frame_id'),
        sim_step=step, step_domain=domain, original_rgb_ref=deepcopy(frame.get('rgb_ref')),
        width=width, height=height, original_image_coordinates=True, image_resized=False,
        source_kind='derived_from_permitted_measured_robot_pose_and_disclosed_camera_calibration',
        camera_pose_source=camera.get('pose_source'), arms=result_arms, points=points, arrows=lines,
        approach_arrow_length_m=APPROACH_ARROW_M, physics_advanced=False,
        executed_native_steps=0, executed_control_steps=0,
        semantics=semantics,
        limitation='Projection does not infer objects, certify contact/retention/free space, or resolve visual occlusion. In-image means camera frustum only.')


def _render_fingertip_overlay(camera, image_bgr, *, arms, selections, targets, profile):
    """Contact overlay of a ``jaw_fingertip_midpoint`` gripper without contact sections (SO-101).

    Per arm: the measured TCP (``profile.tcp_link``), the CAD inner fixed and moving fingertips at the measured jaw
    angle (rad, the arm's jaw joint in ``robot_state.arms[arm].joint_names``), their midpoint as the selected
    reference and a 0.06 m approach arrow along the TCP approach axis. No contact-section selection exists."""
    from .pose_axis_overlay import grasp_frame, jaw_angle_rad
    if selections:
        raise ValueError('contact_overlay_contact_selection_unavailable')
    targets = {} if targets is None else targets
    if not isinstance(targets, dict) or set(targets) - set(arms):
        raise ValueError('contact_overlay_options_require_selected_arms')
    observation, frame, state, width, height, reference_frame = _checked_camera(camera, image_bgr)
    step, domain = observation.get('sim_step'), observation.get('step_domain')
    tcp = profile.tcp_link
    output = image_bgr.copy()
    points, lines, result_arms = [], [], {}

    def add(arm, name, position, source, color, label=None, marker='circle'):
        index = len(points)
        points.append(dict(arm=arm, reference=name, position_m=position.tolist(), source_kind=source,
                           label=label, marker=marker, color_bgr=color))
        return index

    measured_tcp, measured_tips = f'derived_from_measured_{tcp}', f'derived_from_measured_{tcp}_and_jaw_angle'
    for arm in arms:
        origin, rotation = _measured_tcp(state, arm, reference_frame, profile)
        arm_state = state.get('arms', {}).get(arm, {})
        jaw = (jaw_angle_rad(profile, arm, arm_state, arm_state.get('joint_names'))
               if arm_state.get('source_kind') == 'measurement' else None)
        if jaw is None:
            raise ValueError('contact_overlay_requires_measured_jaw_angle:' + arm)
        reference = grasp_frame(profile, origin, rotation, jaw)
        fixed, moving = reference['fingertips']
        axis = reference['approach_axis']
        prefix = arm[:1].upper()
        references = dict(tcp=dict(position_m=origin.tolist(), point_index=add(arm, 'tcp', origin, measured_tcp,
                                                                                COLORS['tcp'], f'{prefix} actual TCP')),
                          fixed_tip=dict(position_m=fixed.tolist(), point_index=add(arm, 'fixed_tip', fixed,
                              measured_tips, COLORS['fixed_tip'], f'{prefix} actual fixed tip')),
                          moving_tip=dict(position_m=moving.tolist(), point_index=add(arm, 'moving_tip', moving,
                              measured_tips, COLORS['moving_tip'], f'{prefix} actual moving tip')))
        selected = reference['grasp_center']
        current_index = add(arm, 'selected', selected, measured_tips, COLORS['selected'],
                            f'{prefix} actual selected (fingertip midpoint)', 'ring')
        current_end = add(arm, 'actual_approach_end', selected + APPROACH_ARROW_M * axis, measured_tips,
                          COLORS['selected'])
        lines.append(dict(arm=arm, kind='actual_approach', start=current_index, end=current_end,
                          source_kind=measured_tips, color_bgr=COLORS['selected']))
        entry = dict(selected_contact=dict(reference=reference['reference'], jaw_angle_rad=jaw,
                                           fingertip_opening_m=reference['opening_m']),
                     **{f'measured_{tcp}_position_m': origin.tolist()},
                     actual_references=references, actual_selected_position_m=selected.tolist(),
                     actual_selected_point_index=current_index, actual_approach_axis_world=axis.tolist(),
                     jaw_closing_axis_world=reference['jaw_axis'].tolist() if reference['jaw_axis'] is not None else None)
        _add_target(entry, arm, targets, selected, prefix, add, lines)
        result_arms[arm] = entry
    _draw(output, camera, points, lines, result_arms, arms, width, height)
    metadata = _metadata(camera, observation, frame, reference_frame, step, domain, width, height, result_arms,
        points, lines,
        semantics=(f'actual labels use the synchronized measured {tcp} pose and measured jaw angle; the CAD inner '
                   'fingertips and their midpoint (the selected reference) are geometric references between the '
                   'finger faces, not a measured object contact; requested labels use only caller inputs'))
    return output, metadata
