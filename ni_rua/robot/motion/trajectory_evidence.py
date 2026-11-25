"""Candidate robot paths from public encoders, FK and hardware model only.

No simulator import, articulation root, object state, collision query or camera
truth. A base transform is derived from measured link6 and URDF FK, requiring
both arms' poses in the same disclosed calibrated frame. Sphere proximity is
discrete, approximate and advisory; it is not physical contact/collision proof.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

ARMS=('left','right')
NEAR_CLEARANCE_M=.03


def compact_trajectory_evidence(full):
    """Bounded display evidence; exact commands and geometry stay in the artifact."""
    samples=full['samples']
    geometry=full.get('geometry',dict(status='unknown'))
    proximity=full.get('inter_arm_proximity',dict(status='unknown'))
    summary=dict(available=True,source_kind=full['source_kind'],native_steps=len(samples)-1,
        total_samples=len(samples),frame_id=full['frame_id'],tcp_config_id=full['tcp_config_id'],
        geometry=deepcopy(geometry),inter_arm_proximity={k:deepcopy(proximity[k]) for k in
            ('status','reason','min_clearance_m','near_threshold_m','sampled_boundaries',
             'missing_finger_coverage','semantics') if k in proximity},
        joint_names=deepcopy(full.get('joint_names',{})),q_unit='rad',q_precision='exact_stored_values',
        key_pose_precision_decimals=4,anomalies=[],
        limits=['No task-object geometry, contact, retention, environment collision or continuous-time safety proof.',
                'Future commands and FK poses are candidates, not measured future state; execution replans.',
                'Inferred bases require the declared common measured frame; the base fit is not a calibration check.'],
        details='robot_inspect_trajectory reads exact q; use pagination for all rows and detail=full for FK details.')
    compact_proximity=summary['inter_arm_proximity']
    witness=proximity.get('witness')
    if witness is not None:
        compact_proximity['witness']=dict(native_step=witness['native_step'],
            left={k:witness['left'][k] for k in ('link_name','sphere_index')},
            right={k:witness['right'][k] for k in ('link_name','sphere_index')})
    if 'near_native_steps' in proximity:
        near=proximity['near_native_steps']
        compact_proximity.update(near_boundary_count=len(near),
            first_near_native_step=near[0] if near else None,last_near_native_step=near[-1] if near else None)
    if geometry['status']=='unknown':
        summary['anomalies'].append(dict(kind='public_geometry_unknown',reason=geometry.get('reason','unavailable')))
    elif proximity['status']=='unknown':
        summary['anomalies'].append(dict(kind='inter_arm_proximity_unknown',reason=proximity.get('reason','unavailable')))
    if proximity['status']=='model_near_approach':
        summary['anomalies'].append(dict(kind='model_near_approach',native_step=witness['native_step'] if witness else None))
    reasons={}
    for sample,reason in [(samples[0],'start'),(samples[len(samples)//4],'quarter'),
            (samples[len(samples)//2],'midpoint'),(samples[3*len(samples)//4],'three_quarters'),(samples[-1],'target')]:
        reasons.setdefault(sample['native_step'],[]).append(reason)
    if witness is not None:
        reasons.setdefault(witness['native_step'],[]).append('minimum_model_clearance')
    summary['key_poses']=[]
    for sample in samples:
        step=sample['native_step']
        if step not in reasons:continue
        arms={}
        for arm,command in sample['arms'].items():
            value=dict(phase=command['phase'],q_rad=deepcopy(command['q_rad']))
            if 'jaw_goal_normalized' in command:
                value['jaw_goal_normalized']=round(command['jaw_goal_normalized'],4)
            if 'link6_pose' in command:
                pose=command['link6_pose']
                value['link6_pose']=dict(position_m=np.round(pose['position_m'],4).tolist(),
                    rotation_matrix=np.round(pose['rotation_matrix'],4).tolist())
            elif 'link6' in command.get('link_positions_m',{}):
                # Historical v1 artifacts stored link origins but not orientations.
                value['link6_position_m']=np.round(command['link_positions_m']['link6'],4).tolist()
            arms[arm]=value
        summary['key_poses'].append(dict(native_step=step,reasons=reasons[step],arms=arms))
    return summary


def nominal_timeline(before, compiled, progression, initial_goals):
    """Exactly the single-segment paired/independent nominal row scheduler."""
    moving=list(compiled)
    lengths={a:len(compiled[a][1]) for a in moving}
    jaws={a:len(compiled[a][2]) for a in moving}
    paired=len(moving)==2 and progression=='paired'
    motion_count=max(lengths.values()); jaw_count=max(jaws.values())
    count=motion_count+jaw_count if paired else max(lengths[a]+jaws[a] for a in moving)
    held={a:[*before['arms'][a]['q_rad'],float(initial_goals[a])] for a in ARMS}
    result=[dict(native_step=0,arms={a:dict(q_rad=list(held[a][:6]),jaw_goal_normalized=held[a][6],
                 phase='initial_measured_state') for a in ARMS})]
    for i in range(count):
        arms={}
        for a in ARMS:
            phase='hold'
            if a in compiled:
                motion,extra=compiled[a][1:3]
                row=None
                if i<len(motion):row,phase=motion[i],'motion'
                elif paired and i>=motion_count and i-motion_count<len(extra):
                    row,phase=extra[i-motion_count],'gripper'
                elif not paired and i-len(motion)<len(extra):row,phase=extra[i-len(motion)],'gripper'
                if row is not None:
                    offset=0 if a=='left' else 7
                    held[a]=list(row[offset:offset+7])
            arms[a]=dict(q_rad=list(held[a][:6]),jaw_goal_normalized=float(held[a][6]),phase=phase)
        result.append(dict(native_step=i+1,arms=arms))
    return result


def _hardware(adapter):
    """Read only the pinned public robot asset directory, never task assets."""
    from ..model.robot_model import urdf_manifest
    root = getattr(adapter, 'root', None)
    if not root:
        raise ValueError('robot_geometry_asset_unavailable')
    try:
        import yaml
    except ImportError as err:
        raise ValueError('optional_yaml_dependency_unavailable') from err
    directory=Path(root)/'Assets/Robots/x5'
    paths={k:directory/name for k,name in [('urdf','X5A.urdf'),('spheres','curobo.yml'),('jaws','robot_config.yml')]}
    model=urdf_manifest(paths['urdf'])
    raw_spheres=yaml.safe_load(paths['spheres'].read_text())
    if not isinstance(raw_spheres,dict) or not isinstance(raw_spheres.get('robot_cfg'),dict) or not isinstance(raw_spheres['robot_cfg'].get('kinematics'),dict):
        raise ValueError('invalid_public_robot_sphere_config')
    config=raw_spheres['robot_cfg']['kinematics']
    robot=yaml.safe_load(paths['jaws'].read_text())
    if not isinstance(robot,dict):
        raise ValueError('invalid_public_jaw_command_metadata')
    return dict(model=model,spheres=config.get('collision_spheres'),sphere_buffer_m=config.get('collision_sphere_buffer',0),
        jaws=robot,provenance={k:dict(filename=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})


_COUNT_WORDS=('zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve')


def _bases(model, before, frame_id, tcp_config_id, profile=None):
    """``{arm: T_world_base}`` fitted from each recorded arm's measured TCP pose and URDF FK of its measured joints.

    ``profile`` (EmbodimentProfile; None = ``ARX_X5_PROFILE``, the legacy default robot) supplies the TCP:
    ``tcp_config_id`` the measured ``ee_pose`` must carry and ``tcp_link``, the URDF link FK evaluates (X5 link6,
    SO-101 gripper_frame_link). Each arm's measured ``joint_names`` must hold the profile's arm-joint count
    (``len(arm_joint_names[arm])``; X5 6, SO-101 5), optionally followed by that arm's ``jaw_joint_names`` when
    the jaw is reported as a joint (SO-101 ``gripper``, rad); jaws reported in ``gripper.joint_names`` (X5 joint7,
    joint8, m) must be measured. Every recorded arm is fitted; ``frame_id`` is the frame all poses must share.
    """
    from ..model.geometry import fk, array, transform
    from ..model.roboprobe_pose import quat_to_matrix
    if profile is None:
        from ..model.embodiments import ARX_X5_PROFILE as profile
    tcp=profile.tcp_link
    bases={}
    for arm in before.get('arms') or {}:  # every recorded arm; ARX-X5 records left and right
        state=before['arms'][arm];pose=state.get('ee_pose') or {};jaw=state.get('gripper',{})
        if (state.get('source_kind')!='measurement' or pose.get('source_kind')!='measurement'
                or pose.get('frame_id')!=frame_id or pose.get('tcp_config_id')!=tcp_config_id
                or tcp_config_id!=profile.tcp_config_id):
            raise ValueError(f'common_calibrated_measured_{tcp}_frame_required')
        if arm not in profile.arm_joint_names:
            raise ValueError('measured_arm_not_in_embodiment:'+str(arm))
        if 'joint_names' not in state or 'q_rad' not in state:
            raise ValueError('measured_joint_names_unavailable')
        names=list(state['joint_names'])
        count=len(profile.arm_joint_names[arm])
        jaw_joints=list(profile.jaw_joint_names[arm])
        fingers=list(jaw.get('joint_names',[]))
        # A jaw not reported in gripper.joint_names may be one of the named joints (SO-101: the gripper angle in
        # rad follows the arm joints).
        inline=(not fingers and bool(jaw_joints) and len(names)==count+len(jaw_joints)
                and names[count:]==jaw_joints)
        values=array(state['q_rad'],(len(names) if inline else count,)).tolist()
        if len(names)!=count and not inline:
            raise ValueError(f'{_COUNT_WORDS[count] if count<len(_COUNT_WORDS) else count}_named_measured_arm_joints_required')
        if fingers:
            if jaw.get('position_source_kind')!='measurement':raise ValueError('measured_jaws_required')
            names+=fingers;values+=array(jaw['position_m'],(len(fingers),)).tolist()
        local=fk({**model,'T_world_base':np.eye(4)},dict(joint_names=names,q_rad=values,source_kind='measurement'))[tcp]
        q=array(pose['quaternion_xyzw'],(4,))
        if abs(np.linalg.norm(q)-1)>1e-4:raise ValueError('normalized_measured_pose_required')
        world=np.eye(4);world[:3,:3]=quat_to_matrix(q[[3,0,1,2]]);world[:3,3]=array(pose['position_m'],(3,))
        bases[arm]=transform(world@np.linalg.inv(local))
    return bases


def _joint_state(state, command, robot, *, initial=False):
    from ..model.geometry import array
    names=list(state['joint_names']);values=list(command['q_rad'])
    if initial:
        jaw=state['gripper'];names+=list(jaw['joint_names']);values+=list(jaw['position_m'])
    else:
        scale=array(robot['gripper_scale'],(2,));goal=command['jaw_goal_normalized']
        if not -0.25<=goal<=1 or scale[1]<=scale[0]:raise ValueError('invalid_public_jaw_command_metadata')
        goal=min(1.0,max(0.0,float(goal)))
        move=robot['gripper_move'];base=move['base']
        position=scale[0]+goal*(scale[1]-scale[0]) if move['sign']==1 else scale[1]-goal*(scale[1]-scale[0])
        positions={base:float(position)}
        mimic=move.get('mimic')
        if mimic:positions[mimic[0]]=float(position*mimic[1]+mimic[2])
        for name in robot['gripper_joints_name']:
            names.append(name);values.append(positions[name])
    return dict(joint_names=names,q_rad=values,source_kind='estimate')


def _sphere_set(config, poses, buffer):
    if not isinstance(config,dict) or not config or not np.isfinite(buffer) or buffer<0:
        raise ValueError('public_robot_sphere_model_unavailable')
    centers=[];radii=[];labels=[]
    for link,spheres in config.items():
        for index,s in enumerate(spheres):
            c=np.asarray(s['center'],dtype=float);radius=float(s['radius'])+buffer
            if c.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(radius) or radius<=0:
                raise ValueError('invalid_public_robot_sphere')
            transform=poses[link];centers.append(transform[:3,:3]@c+transform[:3,3]);radii.append(radius)
            labels.append(dict(link_name=link,sphere_index=index))
    return np.asarray(centers),np.asarray(radii),labels


def build_trajectory_evidence(adapter,before,compiled,progression,action):
    """Return compact actor evidence plus full artifact-ready nominal samples."""
    samples=nominal_timeline(before,compiled,progression,adapter.last_dispatched_goals)
    names={a:list(before['arms'][a].get('joint_names') or adapter.robot.get_robot_by_arm_name(a+'_arm').arm_joints_name) for a in ARMS}
    full=dict(schema_version='candidate-trajectory-v1',source_kind='candidate_command_trajectory_not_measurement',
        frame_id=action['frame_id'],tcp_config_id=action['tcp_config_id'],native_dt_s=1/adapter.control_hz,
        joint_names=names,samples=samples,execution_binding='not_cached; actual execution replans and safety-checks its own commands',
        jaw_scope='nominal commanded aperture; actual arrival jaw ramps recompile from future measurement and may differ')
    full.update(geometry=dict(status='unknown'),inter_arm_proximity=dict(status='unknown'))
    worst=None
    try:
        hardware=_hardware(adapter);model=hardware['model']
        bases=_bases(model,before,action['frame_id'],action['tcp_config_id'])
        from ..model.geometry import fk
        full.update(robot_model=model,hardware_provenance=hardware['provenance'],T_world_bases={a:v.tolist() for a,v in bases.items()},
            base_source='measured_link6_transform @ inverse(URDF_FK(current_encoders)); no simulator base pose')
        full['geometry']=dict(status='available',base_source=full['base_source'],
            calibration_consistency='same declared measured frame; base fit uses current state, not an independent calibration check',
            urdf_sha256=model['source_sha256'])
        proximity_error=None;near_steps=[]
        for sample in samples:
            sphere_sets={}
            for arm in ARMS:
                poses=fk({**model,'T_world_base':bases[arm]},_joint_state(before['arms'][arm],sample['arms'][arm],hardware['jaws'],initial=sample['native_step']==0))
                sample['arms'][arm]['link_positions_m']={link:t[:3,3].tolist() for link,t in poses.items()}
                sample['arms'][arm]['link6_pose']=dict(position_m=poses['link6'][:3,3].tolist(),
                    rotation_matrix=poses['link6'][:3,:3].tolist())
                if proximity_error is None:
                    try:sphere_sets[arm]=_sphere_set(hardware['spheres'],poses,hardware['sphere_buffer_m'])
                    except (ValueError,KeyError) as error:
                        from ...core.tool_errors import extract_domain_code
                        proximity_error = (
                            extract_domain_code(str(error))
                            if isinstance(error, ValueError) and extract_domain_code(str(error))
                            else 'public_robot_sphere_model_unavailable'
                        )
            if proximity_error is None:
                left,lr,ll=sphere_sets['left'];right,rr,rl=sphere_sets['right']
                distances=np.linalg.norm(left[:,None,:]-right[None,:,:],axis=2)-lr[:,None]-rr[None,:]
                i,j=np.unravel_index(np.argmin(distances),distances.shape);gap=float(distances[i,j])
                witness=dict(native_step=sample['native_step'],clearance_m=gap,
                    left={**ll[i], 'center_m':left[i].tolist(),'radius_m':float(lr[i])},
                    right={**rl[j], 'center_m':right[j].tolist(),'radius_m':float(rr[j])})
                if worst is None or gap<worst['clearance_m']:worst=witness
                if gap<=NEAR_CLEARANCE_M:near_steps.append(sample['native_step'])
        if proximity_error is None:
            full['inter_arm_proximity']=dict(status='model_near_approach' if near_steps else 'no_near_approach_at_sampled_boundaries',
                min_clearance_m=worst['clearance_m'],witness=worst,near_native_steps=near_steps,near_threshold_m=NEAR_CLEARANCE_M,
                sampled_boundaries=len(samples),source='public hardware cuRobo collision sphere approximation + URDF FK + public inferred bases',
                checked_links=sorted(hardware['spheres']),missing_finger_coverage=not all(a in hardware['spheres'] for a in ('link7','link8')),
                semantics='discrete model-sphere clearance only; not actual collision/contact; unmodeled jaws/held objects and between-boundary motion remain unknown')
        else:full['inter_arm_proximity']=dict(status='unknown',reason=proximity_error)
    except (OSError, ValueError) as error:
        from ...core.tool_errors import extract_domain_code
        safe_reason = (
            extract_domain_code(str(error))
            if isinstance(error, ValueError) and extract_domain_code(str(error))
            else 'robot_geometry_asset_unavailable'
        )
        full['geometry']=dict(status='unknown',reason=safe_reason,error_type=type(error).__name__)
        full['inter_arm_proximity']=dict(status='unknown',reason='public_geometry_or_relative_calibration_unavailable')
        # Avoid partially populated FK data being mistaken for complete geometry.
        for sample in samples:
            for arm in ARMS:
                sample['arms'][arm].pop('link_positions_m',None)
                sample['arms'][arm].pop('link6_pose',None)
    return compact_trajectory_evidence(full),full
