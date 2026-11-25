"""Synthetic UI fixture, no Codex/model/GPU. Never use as runtime evidence.

Run: mamba run -n ni-rua python web/fixture-server.py --port 8778

The console's Agent bar endpoints (design 16 §8) are mocked here: GET /v1/console/agent-options,
GET /v1/console/profiles/{id}/models, POST /v1/console/agent-options/check, POST /v1/console/profiles[/{id}/remove],
and the control commands apply_agent_settings / handoff_run. Every key, host and model below is a placeholder.
Scenario knobs: POST /fixture/agent with a JSON object (keys of AGENT_KNOBS), GET reads the agent state back
(never a key). Model ids with the prefixes bad- / http400- / net- are refused in a receipt, refused with HTTP 400,
or answered by a dropped connection. The payloads follow the real gateway (ni_rua/apps/gateway/agent_settings.py, design
16 §15-16): profile_schema.fields is the real contract (profiles.FIELD_HELP, imported), matrix cells carry checks,
catalogues summarise the cached listings, profiles[].error is a code plus error_message, apply_agent_settings answers
handoff_required for a session-scoped change (effort on Claude Code), and a service-tier probe reports served_tier.
Knobs added for WP-F: options_delay_s (slow agent-options, the ~3 s first call of a real gateway), served_tier
("requested": the probe serves the requested tier; or a tier name the provider serves instead), stale_effort_scope (the
options claim live effort on Claude Code, as a stale snapshot would, so the UI meets handoff_required).
"""
import argparse
import asyncio
import base64
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.cortex.agents import profiles as profile_contract  # noqa: E402  (the real profile field contract)

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNI7SwEAAK2AWBQR1eEAAAAAElFTkSuQmCC')
IDENTITY = dict(run_id='synthetic-web', episode_id='ep-fixture', env_instance_id='env-fixture', env_generation=1)
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\[\]-]{0,199}\Z")
PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")
EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
BACKENDS = [dict(id='codex', label='Codex', wires=['openai_responses'], live_effort=True, service_tier=True, effort_choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh']),
            dict(id='claude_code', label='Claude Code', wires=['anthropic_messages'], live_effort=False, service_tier=False, effort_choices=['low', 'medium', 'high', 'xhigh', 'max']),
            dict(id='opencode', label='OpenCode', wires=['anthropic_messages', 'openai_responses'], live_effort=True, service_tier=False, effort_choices=['low', 'medium', 'high', 'xhigh', 'max'])]
OPENCODE_EFFORTS = {'anthropic_messages': ['low', 'medium', 'high', 'xhigh', 'max'], 'openai_responses': ['none', 'minimal', 'low', 'medium', 'high', 'xhigh']}
WIRE_PROTOCOLS = {'anthropic_messages': 'MESSAGES', 'openai_responses': 'RESPONSES'}
AGENT_KNOBS = ('apply_delay_s', 'handoff_finalize_s', 'handoff_down_s', 'handoff_outcome', 'blockers', 'catalogue_delay_s', 'unavailable_backends',
               'options_delay_s', 'served_tier', 'stale_effort_scope', 'external')
AGENT_KNOB_DEFAULTS = dict(apply_delay_s=0.0, handoff_finalize_s=0.8, handoff_down_s=2.0, handoff_outcome='success', blockers=[], catalogue_delay_s=0.35,
                           unavailable_backends={}, options_delay_s=0.0, served_tier='requested', stale_effort_scope=False, external=None)
PROFILE_SCHEMA = dict(schema=profile_contract.SCHEMA, wires=list(profile_contract.PROFILE_WIRES), auth_schemes=list(profile_contract.AUTH_SCHEMES),
                      tool_result_images=list(profile_contract.TOOL_RESULT_IMAGES), thinking_display=list(profile_contract.THINKING_DISPLAY_CHOICES),
                      efforts=list(profile_contract.EFFORT_CHOICES), fields=[dict(name=name, help=text) for name, text in profile_contract.FIELD_HELP])
TIER_LABELS = {'default': '正常', 'priority': 'Fast', 'ultrafast': 'Ultrafast'}
THINKING_LABELS = {None: '不发送（默认）', 'summarized': 'summarized（摘要）', 'omitted': 'omitted（不返回思考文本）'}


def catalogue_model(id, name=None, protocols=None, tools=None, image=None, context=None, output=None, efforts=None):
    return dict(id=id, name=name, protocols=protocols, tools=tools, image_input=image, context_window=context, max_output_tokens=output,
                reasoning=None if efforts is None else bool(efforts), reasoning_efforts=efforts)


def initial_profiles():
    both = ['MESSAGES', 'RESPONSES']
    return {
        'provider-alpha': dict(name='Provider Alpha', key='sk-fixture-placeholder-1', wires={'anthropic_messages': dict(base_url='https://api.provider-alpha.example/anthropic', auth_scheme='bearer')},
                               compat=dict(tool_result_images='user_blocks', tool_schema_max_depth=13), backends={'claude_code': {'env': {'CLAUDE_CODE_THINKING_DISPLAY_UPDATES': '0'}}},
                               defaults=dict(model='qwen3.8-flash', effort='high'),
                               catalogue={'anthropic_messages': dict(status='ok', reason=None, age_s=120, stale=False, models=[
                                   catalogue_model('qwen3.8-flash', 'Qwen 3.8 Flash', ['MESSAGES'], True, True, 256000, 32000),
                                   catalogue_model('deepseek-flash', 'DeepSeek Flash', ['MESSAGES'], True, True, 128000, 16000),
                                   catalogue_model('kimi-k2.6', 'Kimi K2.6', ['MESSAGES'], True, True, 256000, 32000, ['low', 'high', 'max']),
                                   catalogue_model('glm-text-only', 'GLM Text', ['MESSAGES'], True, False, 128000),
                                   catalogue_model('mystery-model'),
                                   catalogue_model('gpt-x', 'GPT X', ['RESPONSES'], True, True, 400000, 128000)])}),
        'opencode-console': dict(name='OpenCode Console', key='sk-fixture-placeholder-2',
                                 wires={'anthropic_messages': dict(base_url='https://inference.opencode.example/anthropic', auth_scheme='x-api-key'),
                                        'openai_responses': dict(base_url='https://inference.opencode.example/v1', auth_scheme='bearer')},
                                 compat=dict(tool_result_images='inline'), backends={'opencode': {'thinking_display': None}}, defaults=dict(model='kimi-k2.6'),
                                 catalogue={wire: dict(status='ok', reason=None, age_s=40, stale=False, models=[
                                     catalogue_model('kimi-k2.6', 'Kimi K2.6', both, True, True, 256000, 32000, ['low', 'high', 'max']),
                                     catalogue_model('qwen3.8-flash', 'Qwen 3.8 Flash', both, True, True, 256000, 32000),
                                     catalogue_model('gpt-6.1-sol', 'GPT 6.1', ['RESPONSES'], True, True, 272000, 128000, ['low', 'medium', 'high', 'xhigh'])])
                                     for wire in ('anthropic_messages', 'openai_responses')}),
        'provider-beta': dict(name='Provider Beta', key='sk-fixture-placeholder-3', wires={'openai_responses': dict(base_url='https://provider-beta.example/v1', auth_scheme='bearer')},
                              compat={}, backends={}, defaults={},
                              catalogue={'openai_responses': dict(status='unavailable', reason='timeout', age_s=3 * 3600, stale=True, models=[
                                  catalogue_model('gpt-6.1-sol', 'GPT 6.1', ['RESPONSES'], True, True, 272000, 128000, ['low', 'medium', 'high', 'xhigh']),
                                  catalogue_model('gpt-6-mini', 'GPT 6 Mini', ['RESPONSES'], True, True, 272000, 128000)])}),
        'broken': dict(name='Broken Example', key=None, invalid='profile_key_missing', wires={'anthropic_messages': dict(base_url='https://broken.example', auth_scheme='bearer')},
                       compat={}, backends={}, defaults={}, catalogue={}),
    }


def application():
    blobs = {}
    def artifact(value, binary=False):
        data = value if binary else json.dumps(value).encode()
        key = hashlib.sha256(data).hexdigest()
        blobs[key] = (data, 'image/png' if binary else 'application/json')
        return 'artifact:' + key
    image = artifact(PNG, True)
    observations = [dict(**IDENTITY, observation_id=f'obs-{index}', step_domain='physics_fixture', sim_step=step,
                         data_origin='synthetic_fixture', calibration_id=None, last_action_id=action,
                         cameras=[dict(camera_id='head', frame_id=f'frame-{index}', rgb_ref=image if index < 5 else 'artifact:missing', width=1, height=1)])
                    for index, step, action in [(1, 0, None), (2, 0, None), (3, 4, 'action-failed'), (4, 10, 'action-completed'), (5, 11, None)]]
    raw = artifact(dict(jsonrpc='2.0', method='item/completed', params=dict(item=dict(id='item-motion',type='dynamicToolCall',status='completed'))))
    items = [dict(id='item-text', type='agentMessage', status='completed', text='合成可见消息，不包含隐藏推理', thread_id='thread-fixture', turn_id='turn-fixture', raw_refs=[raw]),
             dict(id='item-motion', type='dynamicToolCall', status='completed', arguments={'target':[.1,.2,.3]}, result={'status':'failed'}, raw_refs=[raw]),
             dict(id='item-file', type='fileChange', status='completed', changes=[{'path':'fixture.py','diff':'-old\n+new'}], raw_refs=[raw]),
             dict(id='item-error', type='error', status='failed', error={'message':'synthetic failure'}, raw_refs=[raw])]
    actions = {}
    for name, status, start, end in [('action-failed','failed',1,3), ('action-completed','completed',3,4)]:
        actions[name] = dict(request=dict(**IDENTITY,action_id=name,based_on_observation_id=f'obs-{start}',path={'target':[.1,.2,.3]},data_origin='synthetic_fixture'),
                             receipt=dict(status=status,execution_start_observation_id=f'obs-{start+1}' if start==1 else 'obs-3',result_observation_id=f'obs-{end}',
                                          planned_trace_ref=artifact({'kind':'planned','points':[[0,0,0],[.1,.2,.3]]}),dispatched_trace_ref=artifact({'kind':'dispatched','points':[[0,0,0],[.08,.2,.3]]}),
                                          actual_trace_ref=artifact({'kind':'actual','points':[[0,0,0],[.07,.2,.3]]}) if status=='completed' else None,before_state_ref=None))
    pending = [dict(request_id=21,method='item/commandExecution/requestApproval',status='pending',thread_id='thread-fixture',turn_id='turn-fixture',item_id='item-approval',params={'command':'synthetic echo','availableDecisions':['decline','accept']},deadline_at=None),
               dict(request_id=22,method='item/tool/requestUserInput',status='pending',thread_id='thread-fixture',turn_id='turn-fixture',item_id='item-question',params={'questions':[{'id':'direction','question':'合成测试方向','options':[{'label':'left','description':'左侧'}]}]},deadline_at=None)]
    events = []
    for event_type, correlation, payload, world in [
        ('observation.available',{'observation_id':'obs-1'},{},'observed'),
        ('observation.delivered',{'observation_id':'obs-1'},{'audience':'primary_model','delivered_to_model':True},'observed'),
        ('observation.delivered',{'observation_id':'obs-3','program_id':'program-fixture'},{'audience':'actor_program','delivered_to_model':False},'observed'),
        ('action.failed',{'action_id':'action-failed','observation_id':'obs-1','item_id':'item-motion','tool_call_id':'item-motion'},actions['action-failed']['receipt'],'observed'),
        ('action.completed',{'action_id':'action-completed','observation_id':'obs-3','item_id':'item-motion','tool_call_id':'item-motion'},actions['action-completed']['receipt'],'observed'),
        ('job.completed',{'job_id':'job-internal'},{'status':'succeeded','scene_version':1},'internal')]:
        events.append(dict(**IDENTITY,event_id=f'event-{len(events)+1}',server_seq=len(events)+1,type=event_type,world_kind=world,correlation=correlation,payload=payload,data_origin='synthetic_fixture'))
    state = dict(owner=None, token=None, expires=0, codex_status='completed', active_action=None, prompts=0, replies=0, resizes=0, terminal_inputs=0, ws_gap=False, sse_gap=False,
                 accepting=True, torque='on', telemetry_stale=False, operator=[], unknown=[], model='gpt-6-mini', speed_scale_cap=0.9)
    cached = {}
    sockets = set()
    streams = set()
    terminal_data = '\x1b[32mSYNTHETIC native terminal fixture\x1b[0m\r\n'.encode()
    terminal = dict(session_id='pty-fixture',thread_id='thread-fixture',pid=101,cursor=len(terminal_data),cols=100,rows=32,mode='read_only',exit_code=None)
    def owner():
        if state['expires'] < time.time(): state.update(owner=None,token=None)
        return dict(owner_client_id=state['owner'],expires_at=state['expires'],mode='gateway',can_claim=not state['owner'])
    def run():
        return dict({**IDENTITY,'run_id':agent['run_id']},status='running',thread_id='thread-fixture',turn_id=agent['turn_id'],agent_backend='external' if agent['external'] is not None else agent['backend'],handoff=agent['handoff_info'],codex_status=state['codex_status'],accepting_actions=state['accepting'],motion_status='running' if state['active_action'] else 'idle',action_count=2,controller=owner(),motion={'active_action':state['active_action'],'status':'running' if state['active_action'] else 'idle'},terminal=terminal,data_origin='synthetic_fixture')
    @web.middleware
    async def guard(request,handler):
        if time.time()<agent['down_until'] and (request.path.startswith('/v1/') or request.path=='/console'):
            request.transport.close()  # listener down during a handoff: the browser sees a network error
            return web.Response(status=503)
        if request.method=='POST' or request.path=='/v1/terminal':
            if request.headers.get('Origin') not in (None,f'http://{request.host}'):
                return web.json_response({'error':'same_origin_required'},status=403)
        return await handler(request)
    app=web.Application(middlewares=[guard])
    dist=Path(__file__).parent/'dist'
    async def api(request):
        path=request.path
        if path=='/v1/run': result=run()
        elif path=='/v1/observations': result={'observations':observations}
        elif path.startswith('/v1/observations/'): result=observations[3] if path.endswith('/latest') else next((item for item in observations if item['observation_id']==request.match_info['id']),None)
        elif path=='/v1/items': result={'items':items,'cursor':len(events)}
        elif path.startswith('/v1/items/'): result=next((item for item in items if item['id']==request.match_info['id']),None)
        elif path=='/v1/requests': result={'requests':pending}
        elif path.startswith('/v1/actions/'): result=actions.get(request.match_info['id'])
        elif path=='/v1/scene/head': result={'scene_version':2,'world_kind':'internal','capabilities':{'physics':'unsupported','rgb_reconstruction':'synthetic_only'}}
        elif path=='/v1/scene/versions': result={'versions':[1,2]}
        elif path.startswith('/v1/scene/versions/'): result={'scene_version':int(request.match_info['id']),'entities':[],'observation_ids':['obs-3','obs-4'],'world_kind':'internal'}
        elif path.startswith('/v1/jobs/'): result={'job_id':request.match_info['id'],'status':'succeeded','result':{'status':'ok','data_origin':'synthetic_fixture'}}
        elif path=='/fixture/stats': result=state
        else: result=None
        return web.json_response(result if result is not None else {'error':'missing'},status=200 if result is not None else 404)
    async def controls(request):
        value=await request.json(); key=value.get('request_id'); command=value.get('command'); owner()
        if key in cached:
            old,result=cached[key]
            if old!=value: return web.json_response({'error':'request_content_conflict'},status=409)
            return web.json_response(result)
        if command=='claim_input':
            if state['owner'] and state['owner']!=value.get('client_id'): return web.json_response({'error':'writer_conflict'},status=409)
            if state['owner'] and state['token']!=value.get('writer_token'): return web.json_response({'error':'writer_conflict'},status=409)
            state.update(owner=value['client_id'],token='token-'+value['client_id'],expires=time.time()+30)
            result={'writer_token':state['token'],'expires_at':state['expires']}
        elif command in ('prompt','reply','renew_input','release_input'):
            if value.get('client_id')!=state['owner'] or value.get('writer_token')!=state['token'] or not state['owner']: return web.json_response({'error':'writer_required'},status=409)
            if command=='renew_input': state['expires']=time.time()+30; result={'expires_at':state['expires']}
            elif command=='release_input': state.update(owner=None,token=None); result={'status':'released'}
            elif command=='prompt':
                state['prompts']+=1; agent['turn_id']=f"turn-{state['prompts']+1}"; agent['served']=dict(model=agent['model'],at=utc())
                result={'thread_id':'thread-fixture','turn_id':agent['turn_id'],'status':'running'}
            else:
                native=next((item for item in pending if item['request_id']==value['native_request_id']),None)
                if not native or native['status']!='pending': return web.json_response({'error':'request_expired'},status=409)
                native.update(status='resolved',response=value['response']); state['replies']+=1; result={'status':'resolved','native_request_id':native['request_id']}
        elif command in ('hold','estop','stop'):
            state['accepting']=False; state['operator'].append(command); result={'status':'stopped','admission':'closed'}
        elif command in ('relax','torque_on','go_home','resume','set_runtime_limits','set_effort','set_model'):
            if value.get('client_id')!=state['owner'] or value.get('writer_token')!=state['token'] or not state['owner']: return web.json_response({'error':'input_owner_claim_required'},status=400)
            if command=='relax' and value.get('confirm')!='relax': return web.json_response({'error':'relax_requires_confirm:relax'},status=400)
            if command=='resume' and sorted(value.get('acknowledge_unknown',[]))!=sorted(item['id'] for item in state['unknown']): return web.json_response({'error':'resume_requires_unknown_acknowledgement'},status=400)
            if command=='relax': state['torque']='off'; state['accepting']=False
            if command=='torque_on': state['torque']='on'
            if command=='resume': state.update(accepting=True,unknown=[])
            if command=='set_runtime_limits': state['speed_scale_cap']=float(value['speed_scale_cap'])
            if command=='set_model': agent['model']=value['model']; agent['changed_at']=utc()
            if command=='set_effort':
                if agent['backend']=='claude_code': agent['effort_next_session']=value['effort']   # applies at the next session start
                else: agent['effort']=value['effort']; agent['changed_at']=utc()
            state['operator'].append(command); result={'status':'resumed' if command=='resume' else 'applied' if command in ('set_runtime_limits','set_effort','set_model') else 'completed','admission':'open' if state['accepting'] else 'closed',**({'model':agent['model']} if command=='set_model' else {})}
        elif command in ('apply_agent_settings','handoff_run'):
            if value.get('client_id')!=state['owner'] or value.get('writer_token')!=state['token'] or not state['owner']: return web.json_response({'error':'input_owner_claim_required'},status=400)
            if command=='apply_agent_settings':
                model=value.get('model')
                if isinstance(model,str) and model.startswith('net-'): request.transport.close(); return web.Response(status=503)
                if isinstance(model,str) and model.startswith('http400-'): return web.json_response({'request_id':key,'error':f'agent_model_rejected:{model}: provider refused the model (fixture)'},status=400)
                await asyncio.sleep(agent['apply_delay_s'])
                result=apply_agent_settings(value)
            else:
                refusal=handoff_refusal(value)
                if refusal: return web.json_response({'request_id':key,'error':refusal},status=400)
                result=start_handoff(value)
            result['request_id']=key
        else: result={'status':'unsupported'}
        cached[key]=(value,result)
        return web.json_response(result)
    joints = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
    limits = dict(zip(joints, [[-1.92,1.92],[-1.75,1.75],[-1.69,1.69],[-1.66,1.66],[-2.74,2.84],[-0.17,1.75]]))
    def telemetry_state():
        now = datetime.now(timezone.utc) - timedelta(seconds=30 if state['telemetry_stale'] else 0)
        phase = time.time()
        q = {name: round(0.6 * math.sin(phase / 3 + index), 4) for index, name in enumerate(joints)}
        return dict(schema='ni_rua.so101.telemetry.v1', seq=int(phase * 10), captured_at=now.isoformat(), monotonic_s=phase,
                    connected=True, torque_enabled=state['torque'] == 'on', q_rad=q, q_goal_rad={k: round(v + .02, 4) for k, v in q.items()},
                    gripper_normalized=.42, tcp_position_m=[.21, .01, .12], tcp_quaternion_xyzw=[0, .707, 0, .707],
                    motor={name: dict(temperature_c=34.5 + index, load=.1 * index, voltage_v=12.1) for index, name in enumerate(joints)},
                    motion=dict(active_action_id=None, phase='idle' if state['torque'] == 'on' else 'fault', tracking_error_rad=.004),
                    cameras=dict(wrist=dict(ok=True, fps=5.0, last_frame_at=now.isoformat(), error=None),
                                 fixed=dict(ok=False, fps=None, last_frame_at=None, error='synthetic_camera_disconnected')),
                    faults=[], data_origin='synthetic_fixture')
    async def console_telemetry(request):
        value = telemetry_state()
        return web.json_response(dict(available=True, stale=state['telemetry_stale'], age_s=30.0 if state['telemetry_stale'] else .1,
                                      stale_after_s=2.0, reason='telemetry_stale' if state['telemetry_stale'] else None, state=value, joint_limits_rad=limits))
    async def console_camera(request):
        # SVG stands in for the worker jpg; the console only sets <img src>.
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240"><rect width="320" height="240" fill="#203028"/><text x="20" y="120" fill="#cfe" font-size="18">SYNTHETIC {request.match_info["camera_id"]} {time.strftime("%H:%M:%S")}</text></svg>'
        return web.Response(text=svg, content_type='image/svg+xml', headers={'Cache-Control': 'no-store'})
    model_options = [dict(value='gpt-6.1-sol', label='GPT 6.1 (gpt-6.1-sol)'), dict(value='gpt-6-mini', label='GPT 6 Mini (gpt-6-mini)'),
                     dict(value='gpt-6.1-terra', label='Terra (gpt-6.1-terra)'), dict(value='gpt-6-luna', label='Luna (gpt-6-luna)')]
    async def console_state(request):
        return web.json_response(dict({**IDENTITY, 'run_id': agent['run_id']}, status='running', codex_status=state['codex_status'], accepting_actions=state['accepting'],
            active_action=None, operator_gate=dict(torque=state['torque'], unknown=state['unknown'], speed_scale_cap=state['speed_scale_cap'], last=None),
            resume=dict(blockers=['torque_off_requires_torque_on'] if state['torque'] == 'off' else [], requires_acknowledgement=[item['id'] for item in state['unknown']]),
            latest_observation_id='obs-4', recent_actions=[dict(action_id=name, type='MoveAction', **actions[name]['receipt'], position_residual_m=.0031) for name in actions],
            task_plan=dict(revision=1, plan=['观察场景', '抓取方块', '放入碗中'], attention=[]), operator_supported=True, joint_limits_rad=limits,
            settings=dict(**({'agent': external_current()} if agent['external'] is not None else {}), runtime=dict(speed_scale_cap=dict(value=state['speed_scale_cap'], default=0.9, range=[.1, 1]),
                                       model=dict(value=agent['model'], choices=[item['value'] for item in model_options], options=model_options, free_text=True),
                                       effort=dict(value=agent['effort_next_session'] or agent['effort'], choices=backend_info(agent['backend'])['effort_choices'],
                                                   applies='next Codex turn' if backend_info(agent['backend'])['live_effort'] else 'next session start'),
                                       service_tier=dict(value=agent['service_tier'], choices=['default', 'priority', 'ultrafast'], probes=agent['probes'])),
                          new_run=dict(model='gpt-6-mini', observation_profile='rgb_proprio_v1', environment='so101_real', robot_model_id='so101', backend={})),
            effective_config={'model': {'name': 'gpt-6-mini'}, 'data_origin': 'synthetic_fixture'}))
    # ---------- Agent bar (design 16 §8) ----------
    agent = dict(run_id=IDENTITY['run_id'], turn_id='turn-fixture', backend='claude_code', profile='provider-alpha', wire_api='anthropic_messages',
                 model='qwen3.8-flash', effort='high', service_tier='default', thinking_display=None, changed_at=None, served=None, probes={},
                 handoff_info=None, down_until=0.0, runs=1, effort_next_session=None, presets={'provider-alpha/claude_code/kimi-k2.6': {'effort': 'max'}},
                 **json.loads(json.dumps(AGENT_KNOB_DEFAULTS)))
    profiles = initial_profiles()
    def utc(): return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    def backend_info(backend):
        info = next(item for item in BACKENDS if item['id'] == backend)
        problem = agent['unavailable_backends'].get(backend)
        return dict(info, available=not problem, reason=f'backend_unavailable:{problem}' if problem else None,
                    message=f"{info['label']} 运行时未安装（{problem}）" if problem else None)
    def profile_info(pid):
        profile = profiles[pid]
        return dict(id=pid, name=profile['name'], valid=not profile.get('invalid'), error=profile.get('invalid'),
                    error_message=f"{profile['invalid']}:/fixture/configs/profiles/{pid}.local.json: profile 缺少 credential.api_key" if profile.get('invalid') else None,
                    wires={wire: dict(host=value['base_url'].split('/')[2], base_url=value['base_url'], auth_scheme=value['auth_scheme']) for wire, value in profile['wires'].items()},
                    compat=profile['compat'], backends=profile['backends'], defaults=profile['defaults'], key_present=bool(profile.get('key')), in_use=pid == agent['profile'])
    def cell(backend, pid):
        info, profile = backend_info(backend), profiles.get(pid)
        if not info['available']: return dict(compatible=False, reason=info['reason'], message=info['message'], wires=[])
        if profile is None: return dict(compatible=False, reason='profile_unknown', message=f'未知 profile {pid}', wires=[])
        if profile.get('invalid'): return dict(compatible=False, reason=f"profile_invalid:{profile['invalid']}", message='profile 缺少 credential.api_key', wires=[])
        wires = [wire for wire in info['wires'] if wire in profile['wires']]
        if not wires: return dict(compatible=False, reason='no_common_wire', wires=[],
                                  message=f"该 profile 只提供 {'、'.join(profile['wires'])}，{info['label']} 需要 {' 或 '.join(info['wires'])}")
        return dict(compatible=True, reason=None, message=None, wires=wires, checks=[dict(wire=wire, ok=True, reason=None, message=None, effort='high') for wire in wires])
    def model_entry(pid, wire, model):
        return next((item for item in profiles.get(pid, {}).get('catalogue', {}).get(wire, {}).get('models', []) if item['id'] == model), None)
    def reasoning(backend, pid, wire, model, effort):
        info = backend_info(backend)
        accepted = OPENCODE_EFFORTS.get(wire, info['effort_choices']) if backend == 'opencode' else info['effort_choices']
        entry = model_entry(pid, wire, model)
        listed = entry.get('reasoning_efforts') if entry else None
        live = info['live_effort'] or (backend == 'claude_code' and agent['stale_effort_scope'])
        result = dict(effort=dict(available=True, value=effort, command='set_effort' if live else 'handoff_run', applies='next_turn' if live else 'next_session',
                                  model_source='catalogue' if listed else None, note=None,
                                  choices=[dict(value=item, label=item, description=None, backend=item in accepted, model=None if listed is None else item in listed,
                                                selectable=item in accepted, note=None if item in accepted else '当前后端不支持') for item in EFFORTS]),
                      service_tier=dict(available=True, value=agent['service_tier'], command='set_service_tier', applies='next_model_request',
                                        served_tier=(agent['probes'].get(agent['service_tier']) or {}).get('served_tier') if agent['service_tier'] != 'default' else None,
                                        choices=[dict(value=tier, label=TIER_LABELS[tier], probe=agent['probes'].get(tier),
                                                      selectable=(agent['probes'].get(tier) or {}).get('supported') is not False, note=None) for tier in ('default', 'priority', 'ultrafast')])
                      if info['service_tier'] else dict(available=False, reason='service_tier_unsupported_by_backend'),
                      thinking_display=dict(available=True, value=agent['thinking_display'], applies='next_session', applies_label='需要新会话', command='handoff_run',
                                            choices=[dict(value=item, label=THINKING_LABELS[item], selectable=True) for item in (None, 'summarized', 'omitted')])
                      if backend == 'opencode' and wire == 'anthropic_messages' else dict(available=False, reason='backend_has_no_thinking_display'))
        return result
    def external_current():
        # The Python side's agent_settings_view() for an external run (knob `external` = {endpoint, client, last_call_at}).
        return dict(backend='external', profile=None, provider=None, wire_api=None, model=None, effort=None, service_tier=None, thinking_display=None, upstream_source=None,
                    revision=0, changed_at=None, turn_count=0, mixed=False, external=dict(agent['external']))
    def handoff_view():
        if agent['external'] is not None: return dict(allowed=False, blockers=[dict(code='handoff_unsupported_by_backend', message=None)], disable_torque_on_disconnect=False, last=None)
        blockers = [dict(code=code, message={'turn_running': '需要 Agent 空闲', 'benchmark_run': '评测 run 由队列管理，不能在页面上切换'}.get(code)) for code in agent['blockers']]
        return dict(allowed=not blockers, blockers=blockers, disable_torque_on_disconnect=False, last=agent['handoff_info'])
    def served_view():
        tier = (agent['probes'].get(agent['service_tier']) or {}).get('served_tier') if agent['service_tier'] != 'default' else None
        if not agent['served'] and tier is None: return None
        return dict(model=(agent['served'] or {}).get('model'), at=(agent['served'] or {}).get('at'), models=None, service_tier=tier, service_tier_source='probe' if tier else None)
    async def agent_options(request):
        await asyncio.sleep(agent['options_delay_s'])
        current = {key: agent[key] for key in ('backend', 'profile', 'wire_api', 'model', 'effort', 'service_tier', 'thinking_display', 'changed_at')}
        current['provider'] = agent['profile']
        if agent['external'] is not None: current = external_current()
        if not backend_info(agent['backend'])['service_tier']: current['service_tier'] = None
        if agent['effort_next_session'] and agent['effort_next_session'] != agent['effort']: current['effort_next_session'] = agent['effort_next_session']
        catalogues = [dict(profile=pid, wire_api=wire, status=entry.get('status'), reason=entry.get('reason'), failure=None, fetched_at=None, age_s=entry.get('age_s'),
                           stale=bool(entry.get('stale')), refreshing=False, count=len(entry.get('models') or []))
                      for pid, profile in profiles.items() if not profile.get('invalid') for wire, entry in profile.get('catalogue', {}).items()]
        return web.json_response(dict(
            current=current, backends=[backend_info(item['id']) for item in BACKENDS], profiles=[profile_info(pid) for pid in profiles],
            matrix=[dict(backend=item['id'], profile=pid, **{'checks': [], **cell(item['id'], pid)}) for item in BACKENDS for pid in profiles],
            reasoning=reasoning(agent['backend'], agent['profile'], agent['wire_api'], agent['model'], agent['effort']), catalogues=catalogues, handoff=handoff_view(),
            served=served_view(), presets=agent['presets'], profiles_dir='/fixture/configs/profiles', profile_schema=PROFILE_SCHEMA), headers={'Cache-Control': 'no-store'})
    async def profile_models(request):
        pid, wire = request.match_info['id'], request.query.get('wire')
        if pid not in profiles: return web.json_response({'error': 'profile_unknown'}, status=404)
        await asyncio.sleep(agent['catalogue_delay_s'])
        entry = dict(profiles[pid].get('catalogue', {}).get(wire) or dict(status='not_read', reason=None, age_s=None, stale=False, models=[]))
        if request.query.get('refresh') == '1' and entry['status'] == 'unavailable':
            entry.update(status='ok', reason=None, age_s=0, stale=False)
            profiles[pid]['catalogue'][wire] = entry
        return web.json_response(dict(entry, fetched_at=utc()), headers={'Cache-Control': 'no-store'})
    async def agent_check(request):
        value = await request.json()
        verdict = cell(value.get('backend'), value.get('profile'))
        result = dict(ok=verdict['compatible'], reason=verdict['reason'], message=verdict['message'])
        if verdict['compatible']:
            if value.get('wire_api') not in verdict['wires']:
                result.update(ok=False, reason='gate:wire_not_served', message=f"该组合不提供 {value.get('wire_api')}")
            elif value.get('effort') and not next(item for item in reasoning(value['backend'], value['profile'], value['wire_api'], value.get('model'), value['effort'])['effort']['choices'] if item['value'] == value['effort'])['backend']:
                result.update(ok=False, reason=f"gate:agent_effort_unsupported:{value['backend']}:{value['effort']}",
                              message=f"agent_effort_unsupported:{value['backend']}:{value['effort']}: 在 {value['wire_api']} 上请选择其他推理强度")
            elif not MODEL_ID.match(str(value.get('model') or '')):
                result.update(ok=False, reason='gate:invalid_model', message='模型名格式无效')
        result['reasoning'] = reasoning(value.get('backend'), value.get('profile'), value.get('wire_api'), value.get('model'), value.get('effort')) if result['ok'] or verdict['compatible'] else None
        await asyncio.sleep(0.15)
        return web.json_response(result)
    def apply_agent_settings(value):
        # Same rule as the real gateway: a session-scoped change applies nothing and answers handoff_required.
        if agent['backend'] == 'claude_code' and 'effort' in value and value['effort'] != agent['effort']:
            diff = {field: dict(current=agent[field], requested=value[field]) for field in ('model', 'effort', 'service_tier') if field in value and value[field] != agent[field]}
            return dict(status='handoff_required', reason='handoff_required', session_fields=['effort'], diff=diff, results={},
                        handoff=dict(command='handoff_run', available=True), message='推理强度 需要新会话：当前 run 不变，用 handoff_run 收尾并启动后继 run；本次未应用任何设置')
        results, stopped = {}, False
        for field in ('model', 'effort', 'service_tier'):
            if field not in value: continue
            if stopped: results[field] = dict(status='skipped', message='前一项被拒绝，本项未执行'); continue
            selected = value[field]
            if field == 'model':
                if not isinstance(selected, str) or not MODEL_ID.match(selected): results[field] = dict(status='rejected', reason='invalid_model', message='invalid_model')
                elif selected.startswith('bad-'): results[field] = dict(status='rejected', reason='agent_model_rejected', message=f'agent_model_rejected:{selected}: 供应商拒绝该模型（fixture）')
                else: agent.update(model=selected, changed_at=utc()); results[field] = dict(status='applied', applies='next_turn', model=selected)
            elif field == 'effort':
                info = backend_info(agent['backend'])
                choices = OPENCODE_EFFORTS[agent['wire_api']] if agent['backend'] == 'opencode' else info['effort_choices']
                if selected not in choices: results[field] = dict(status='rejected', reason='invalid_effort', message='invalid_effort')
                else: agent.update(effort=selected, changed_at=utc()); results[field] = dict(status='applied', applies='next_turn' if info['live_effort'] else 'next_session')
            else:
                if not backend_info(agent['backend'])['service_tier']: results[field] = dict(status='unsupported', reason='service_tier_unsupported_by_backend', message='当前 agent 后端不支持服务档位设置')
                elif selected == 'ultrafast':
                    agent['probes']['ultrafast'] = dict(supported=False, rejection='HTTP 400 unsupported service_tier')
                    results[field] = dict(status='unsupported', probe=agent['probes']['ultrafast'], message='当前供应商不支持 ultrafast（HTTP 400 unsupported service_tier）')
                else:
                    served = selected if agent['served_tier'] == 'requested' else agent['served_tier']
                    probe = None if selected == 'default' else agent['probes'].setdefault(selected, dict(supported=True, served_tier=served))
                    agent['service_tier'] = selected
                    results[field] = dict(status='applied', applies='next_model_request', probe=probe, tier=selected, served_tier=(probe or {}).get('served_tier'))
                    if probe is not None and probe.get('served_tier') != selected:
                        results[field]['message'] = f"供应商接受了 {selected}，但探测请求实际以 {probe.get('served_tier') or '未报告的'} 档位服务（以每次响应为准）"
            stopped = results[field]['status'] != 'applied'
            if results[field]['status'] == 'applied':
                agent['presets'][f"{agent['profile']}/{agent['backend']}/{agent['model']}"] = dict(effort=agent['effort'], service_tier=agent['service_tier'])
        return dict(status='applied' if all(item['status'] == 'applied' for item in results.values()) else 'partial' if any(item['status'] == 'applied' for item in results.values()) else 'rejected',
                    results=results)
    def handoff_refusal(value):
        if agent['blockers']: return f"handoff_blocked:{','.join(agent['blockers'])}"
        verdict = cell(value.get('backend'), value.get('profile'))
        if not verdict['compatible']: return f"handoff_{verdict['reason']}: {verdict['message']}"
        if value.get('wire_api') not in verdict['wires']: return 'handoff_wire_not_served'
        if not MODEL_ID.match(str(value.get('model') or '')): return 'handoff_invalid_model'
        return None
    def start_handoff(value):
        agent['runs'] += 1
        successor = f"synthetic-web-{agent['runs']}"
        asyncio.get_running_loop().create_task(perform_handoff(value, agent['run_id'], successor, agent['handoff_outcome']))
        return dict(status='handoff_accepted', successor_run_id=successor)
    async def perform_handoff(value, predecessor, successor, outcome):
        await asyncio.sleep(agent['handoff_finalize_s'])   # old run finalizes and still answers meanwhile
        agent['down_until'] = time.time() + (3600 if outcome == 'silent' else agent['handoff_down_s'])
        for socket in list(sockets): await socket.close()
        for response in list(streams):
            try: await response.write_eof()
            except (ConnectionResetError, RuntimeError): pass
        if outcome == 'silent': return
        await asyncio.sleep(agent['handoff_down_s'])
        state.update(owner=None, token=None, expires=0); cached.clear()
        if outcome == 'fallback':
            agent['runs'] += 1
            agent.update(run_id=f"synthetic-web-{agent['runs']}", turn_id='turn-fixture', handoff_info=dict(predecessor_run_id=predecessor, request_id=None,
                         fallback=dict(failed_run_id=successor, reason='successor_start_failed: claude_code_sdk_missing（fixture）')))
        else:
            agent.update(run_id=successor, turn_id='turn-fixture', backend=value['backend'], profile=value['profile'], wire_api=value['wire_api'], model=value['model'],
                         effort=value.get('effort'), effort_next_session=None, thinking_display=value.get('thinking_display'), service_tier='default', changed_at=None, served=None,
                         handoff_info=dict(predecessor_run_id=predecessor, request_id=value.get('request_id')))
        agent['down_until'] = 0.0
    async def profile_save(request):
        value = await request.json()
        if value.get('client_id') != state['owner'] or value.get('writer_token') != state['token'] or not state['owner']: return web.json_response({'error': 'input_owner_claim_required'}, status=409)
        pid, document = value.get('id'), value.get('document') or {}
        if not isinstance(pid, str) or not PROFILE_ID.match(pid): return web.json_response({'error': 'profile_invalid_id'}, status=400)
        if document.get('schema') != 'ni_rua.provider_profile.v1': return web.json_response({'error': 'profile_schema_mismatch'}, status=400)
        key = (document.get('credential') or {}).get('api_key')
        if value.get('keep_key'):
            if pid not in profiles or not profiles[pid].get('key'): return web.json_response({'error': 'profile_keep_key_without_stored_key'}, status=400)
            key = profiles[pid]['key']
        if not isinstance(key, str) or not key: return web.json_response({'error': 'profile_key_missing'}, status=400)
        if not document.get('wires'): return web.json_response({'error': 'profile_no_wires'}, status=400)
        previous = profiles.get(pid, {})
        profiles[pid] = dict(name=document.get('name') or pid, key=key, wires=document['wires'], compat=document.get('compat') or {}, backends=document.get('backends') or {},
                             defaults=document.get('defaults') or {}, catalogue=previous.get('catalogue') or {wire: dict(status='ok', reason=None, age_s=0, stale=False, models=[
                                 catalogue_model('qwen3.8-flash', 'Qwen 3.8 Flash', list(WIRE_PROTOCOLS[w] for w in document['wires'] if w in WIRE_PROTOCOLS), True, True, 256000, 32000)])
                                 for wire in document['wires']})
        state['profile_saves'] = state.get('profile_saves', 0) + 1
        return web.json_response(profile_info(pid))
    async def profile_remove(request):
        value = await request.json()
        if value.get('client_id') != state['owner'] or value.get('writer_token') != state['token'] or not state['owner']: return web.json_response({'error': 'input_owner_claim_required'}, status=409)
        pid = request.match_info['id']
        if pid not in profiles: return web.json_response({'error': 'profile_unknown'}, status=404)
        if pid == agent['profile']: return web.json_response({'error': 'profile_in_use'}, status=409)
        del profiles[pid]
        return web.json_response(dict(status='removed', id=pid, renamed_to=f'{pid}.local.json.removed-{utc()}'))
    async def agent_knobs(request):
        if request.method == 'POST':
            value = await request.json()
            unknown = set(value) - set(AGENT_KNOBS) - {'reset'}
            if unknown: return web.json_response({'error': 'unknown_agent_knobs:' + ','.join(sorted(unknown))}, status=400)
            if value.pop('reset', False):
                profiles.clear(); profiles.update(initial_profiles())
                agent.update(run_id=IDENTITY['run_id'], turn_id='turn-fixture', backend='claude_code', profile='provider-alpha', wire_api='anthropic_messages', model='qwen3.8-flash',
                             effort='high', service_tier='default', thinking_display=None, changed_at=None, served=None, probes={}, handoff_info=None, down_until=0.0, runs=1,
                             effort_next_session=None, presets={'provider-alpha/claude_code/kimi-k2.6': {'effort': 'max'}}, **json.loads(json.dumps(AGENT_KNOB_DEFAULTS)))
                state.update(owner=None, token=None, expires=0, speed_scale_cap=0.9); cached.clear()
            agent.update(value)
        return web.json_response({key: value for key, value in agent.items() if key != 'presets'} | {'profiles': sorted(profiles)})
    async def console_page(request): return web.FileResponse(dist/'console.html')
    async def media(request):
        value=blobs.get(request.match_info['id'])
        return web.Response(body=value[0],content_type=value[1]) if value else web.Response(status=404)
    async def stream(request):
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache'}); await response.prepare(request); streams.add(response)
        after=max(int(request.query.get('after',0)),int(request.headers.get('Last-Event-ID',0)))
        try:
            if state['sse_gap']:
                state['sse_gap']=False
                await response.write(('event: gap\ndata: '+json.dumps({'current_seq':len(events),'reason':'synthetic_upstream_gap'})+'\n\n').encode())
            for event in events:
                if event['server_seq']>after:
                    data=(f"id: {event['server_seq']}\ndata: "+json.dumps(event)+'\n\n').encode(); await response.write(data); await response.write(data)
            await response.write(b'retry: 250\n\n')
            while True: await response.write(b': fixture\n\n'); await asyncio.sleep(.5)
        except (ConnectionResetError,asyncio.CancelledError,RuntimeError): pass
        finally: streams.discard(response)
        return response
    async def ws(request):
        socket=web.WebSocketResponse(); await socket.prepare(request); sockets.add(socket)
        after=int(request.query.get('after',0)); await socket.send_json({'type':'ready','status':terminal})
        if state['ws_gap'] or after>len(terminal_data):
            state['ws_gap']=False; await socket.send_json({'type':'gap','cursor':0,'reason':'synthetic_cursor_gap'}); await socket.close()
        elif after<len(terminal_data): await socket.send_json({'type':'output','after':after,'cursor':len(terminal_data),'data':base64.b64encode(terminal_data[after:]).decode()})
        async for message in socket:
            if message.type==WSMsgType.TEXT:
                value=json.loads(message.data)
                if value.get('type')=='resize': state['resizes']+=1; terminal.update(cols=value['cols'],rows=value['rows'])
                elif value.get('type')=='scroll': state['scrolls']=state.get('scrolls',0)+1
                else: state['terminal_inputs']+=1; await socket.send_json({'type':'error','message':'terminal_input_is_read_only'})
        sockets.discard(socket); return socket
    async def fault(request):
        state.update(await request.json())
        for socket in list(sockets): await socket.close()
        for response in list(streams): await response.write_eof()
        return web.json_response({'status':'injected'})
    async def scene_post(request):
        value=await request.json()
        if request.path.endswith('/snapshot'): result={'snapshot_id':'snapshot-fixture','scene_version':value['scene_version'],'anchor_observation_id':value.get('anchor_observation_id'),'world_kind':'internal','scope':'geometry_only'}
        elif request.path.endswith(('/alignment','/update')): result={'job_id':'job-alignment','status':'queued','world_kind':'internal','request':value}
        else: result={'status':'ok','value':{'scene_version':value.get('scene_version'),'hit':None},'world_kind':'internal'}
        return web.json_response(result)
    for route in ['/v1/run','/v1/observations','/v1/observations/{id}','/v1/items','/v1/items/{id}','/v1/requests','/v1/actions/{id}','/v1/scene/head','/v1/scene/versions','/v1/scene/versions/{id}','/v1/jobs/{id}','/fixture/stats']: app.router.add_get(route,api)
    app.router.add_get('/v1/events',stream); app.router.add_get('/v1/terminal',ws); app.router.add_get('/v1/artifacts/{id}',media)
    app.router.add_post('/v1/control',controls); app.router.add_post('/fixture/fault',fault)
    app.router.add_get('/v1/console/telemetry',console_telemetry); app.router.add_get('/v1/console/cameras/{camera_id}.jpg',console_camera)
    app.router.add_get('/v1/console/state',console_state); app.router.add_get('/console',console_page)
    app.router.add_get('/v1/console/agent-options',agent_options); app.router.add_post('/v1/console/agent-options/check',agent_check)
    app.router.add_get('/v1/console/profiles/{id}/models',profile_models); app.router.add_post('/v1/console/profiles',profile_save)
    app.router.add_post('/v1/console/profiles/{id}/remove',profile_remove); app.router.add_route('*','/fixture/agent',agent_knobs)
    for route in ['/v1/scene/query','/v1/scene/snapshot','/v1/scene/alignment','/v1/scene/update','/v1/jobs/{id}/cancel']: app.router.add_post(route,scene_post)
    async def index(request): return web.FileResponse(dist/'index.html')
    app.router.add_get('/',index); app.router.add_static('/',dist)
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8778); args=parser.parse_args()
    web.run_app(application(),host='127.0.0.1',port=args.port)
