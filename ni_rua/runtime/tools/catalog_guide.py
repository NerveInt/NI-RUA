"""Compact field guide of a tool's input contract (``features.tool_catalog_profile`` ``compact_v1`` / ``layered_v1``).

``definitions._with_input_guide`` (profile ``full_v1``) writes every schema field into the tool description as prose
("Type: number; Minimum (inclusive): 0; Maximum (inclusive): 0.15."), because the Codex and OpenAI-compatible clients
drop bounds, conditionals and nested field descriptions from the schema they forward. This module renders the SAME
schema tree (same field help, same bounds, same structural rules) in a typed notation:

    stages[].max_upward_dz_m: number 0..0.15 -- Move only: optional upward +Z compliance window ...

``compact_v1`` states the whole contract: every ``description`` of the schema verbatim, every bound, value set and
default, and every structural rule (required fields, closed objects, alternatives, conditionals, the forms of a
discriminated union). Constructs the notation does not state faithfully are refused with ``UnsupportedSchema``
(unknown keywords, nested or double unions, field declarations inside plain alternatives or conditionals, one
keyword with two values for a field, value sets narrowed by another keyword, bounds on a node without a type);
``definitions._publish`` then publishes that tool with the prose guide instead.

A layered profile (``LAYERS``) additionally shortens the fields named cold by ``COLD``: ``layered_v1`` lists their
path and signature, ``layered_v2`` also the start of their help, names the rarely used forms of a union without their
rules, replaces a field line that an earlier loaded tool states identically by a reference to it (``share_lines``)
and names the tools published for on-demand discovery in the startup instructions (``on_demand_index``). The full
text is published in the tool reference document (``reference_document``), which the actor reads in its working
directory. ``LAYERS`` lists what each layered profile does and why the earlier ones are kept. Cold fields and tools are
an empirical selection for one robot (``EVIDENCE_ROBOT``): a field or tool is cold only when it is listed, so a field
of another robot, a new field or a plugin tool is always described in full. No field is removed from a schema and
validation is the same in every profile.
"""
from copy import deepcopy
import re

from ...core.contracts import canonical

#: Values of ``features.tool_catalog_profile``; ``full_v1`` and ``guide_only_v1`` publish the prose guide.
PROFILES = ('full_v1', 'guide_only_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3')
PROSE_PROFILES = ('full_v1', 'guide_only_v1')
#: What each layered profile does. ``gist``: a cold field keeps the start of its help (else type and bounds only);
#: ``nested``: the fields nested in a cold field are listed too (else only in the reference); ``cold_forms``: the rule
#: lines of rarely used union forms stay in the description (else only their names); ``shared``: a field line identical
#: to that of an earlier loaded tool is replaced by a reference to it (``share_lines``); ``defer``: the tools of
#: ``COLD_TOOLS`` are published for on-demand discovery; ``index``: the tools published for on-demand discovery are
#: named in the startup instructions (``on_demand_index``); ``command``: the cold block suggests a grep command for
#: the reference file (else it describes the file and prescribes no command).
#: ``layered_v3`` is the current form. ``layered_v1`` and ``layered_v2`` are the forms of the first two evaluations,
#: kept so that their archived runs load. In those runs no deferred tool was ever loaded: on the model channels they
#: used, the model's tool-search calls came back in a form the Codex host rejects ("unsupported call"), so a deferred
#: tool was in effect a removed tool, and naming the deferred tools in the instructions could not change that.
LAYERS = {
    'layered_v1': dict(gist=False, nested=True, cold_forms=True, shared=False, defer=True, index=False, command=True),
    'layered_v2': dict(gist=True, nested=False, cold_forms=False, shared=True, defer=True, index=True, command=True),
    'layered_v3': dict(gist=True, nested=False, cold_forms=False, shared=True, defer=False, index=False, command=False),
}
REFERENCE_NAME = 'tool_reference.md'
#: Robot model (``EmbodimentProfile.robot_model_id``) whose archived episodes the cold selection below was made from.
EVIDENCE_ROBOT = 'arx-x5'
EVIDENCE_ROBOTS = frozenset({EVIDENCE_ROBOT, 'libero-franka-panda'})

# Starts like the prose guide ("Input field guide"): text that is placed before a tool's guide finds either form.
_INTRO = 'Input field guide (use only fields applicable to the selected mode/kind).'
#: Notation sentences; a guide's header names only the ones its lines use (``_header``).
_NOTATION = (
    ('list', '[T] list of T'),
    ('tuple', '[A, B] exactly these items in this order'),
    ('range', 'a..b inclusive range'),
    ('bound', '>a <b exclusive and >=a <=b inclusive bounds'),
    ('count', 'xN exactly N items'),
    ('default', '=v default'),
    ('values', 'a|b allowed values'),
    ('alternatives', '" / " separates signatures or rules that hold in different forms'),
    ('forms', '"when X:" holds only in form X of the enclosing object (X is a value of its selecting field), '
              '"except X:" in its other forms'),
    ('closed', '"closed" objects accept only the listed fields'),
    ('needs', '"a needs b" allows a only together with b'),
)
#: Heads the block of cold fields of a layered guide: signatures only (``layered_v1``) / with gists (``layered_v2``).
#: Every line that heads a block of cold items starts with this (``share_lines`` stops at the first one).
_COLD_MARK = 'Rarely needed '
COLD_HEADER = (_COLD_MARK + 'fields, listed with type and bounds only. Their meaning, units, defaults and rules are '
               'in {name} in your working directory: read the line of a field (for example grep -n "^{example}:" '
               '{name}) before first using it.')
COLD_GIST_HEADER = (_COLD_MARK + 'fields: type, bounds and the description, cut where it ends in "...". The complete '
                    'line of a field, its rules and its nested fields are in {name} in your working directory; read '
                    'them (for example grep -n "^{example}" {name}) before first using a field whose line is cut or '
                    'that has nested fields.')
COLD_LINE_HEADER = (_COLD_MARK + 'fields: type, bounds and the description, cut where it ends in "...". The file {name} '
                    'in your working directory has one complete line per field, starting with the field path and a '
                    'colon, with its rules, and lines for its nested fields; read them before first using a field '
                    'whose line is cut or that has nested fields.')
COLD_FORMS_HEADER = _COLD_MARK + 'forms of {path} (their rule lines start with "  {path} when" in {name}): {forms}.'
#: Longest gist of a cold field's help, in characters.
GIST_LIMIT = 140
#: A field line is shared between tools only when its text after the path has at least this many characters.
SHARED_MIN = 150
SHARED_TEXT = 'same as in {tool}'
ON_DEMAND_HEADER = ('## Tools loaded on demand\nThese tools are available but not loaded. To use one, search for its '
                    'name with your tool-search tool, then call it.')


class UnsupportedSchema(ValueError):
    """The schema uses a construct this notation cannot state faithfully (the message names it)."""


# ---- cold fields and tools (layered_v1) ----------------------------------------------------------------------------
# Evidence: 327 Full-47 baseline episodes of gpt-6.1-sol on the ARX-X5 (2026-11-19). A field is listed here when
# fewer than 5% of those episodes used it and it is not one of the per-stage tolerance, compliance and guard controls
# (those stay described whatever their use). The per-field counts are recorded with the experiment.
_COLD_ROBOT_ACTION = frozenset({
    # batch-wide defaults, inspection, obstacle review and bookkeeping fields
    'action_id', 'allow_approach_tilt', 'allow_jaw_symmetry', 'auto_execute_if_clean', 'camera_id',
    'carry_slip_guard', 'carry_slip_scope', 'checkpoint_after_gripper', 'clearance_height_m', 'clearance_preset',
    'contact_entity_id', 'detail', 'grip_guard_preset', 'held_object', 'held_objects', 'ignore_obstacle_ids',
    'image_delivery', 'initial_contact', 'joint_travel_preset', 'max_aperture_m', 'max_backtracking_rad',
    'max_downward_dz_m', 'max_joint_travel_rad', 'max_samples', 'max_tracking_divergence_m', 'max_upward_dz_m',
    'min_aperture_m', 'native_steps', 'obstacle_object_ids', 'obstacle_policy', 'obstacles',
    'orientation_tolerance_rad', 'path_kind', 'placement_contact', 'position_tolerance_m', 'settle',
    'skip_failed_stage', 'slip_consecutive_steps', 'stall_guard_preset', 'start_native_step',
    'target_exemption_preset', 'target_exemption_radius_m', 'timeout_s', 'tolerance_preset', 'trajectory_ref',
    # stage fields
    'stages[].allow_approach_tilt', 'stages[].clearance_height_m', 'stages[].clearance_preset',
    'stages[].gripper_speed_scale', 'stages[].max_backtracking_rad', 'stages[].max_joint_travel_rad',
    'stages[].max_tracking_divergence_m', 'stages[].slip_consecutive_steps',
})
# geometry_query request kinds used by fewer than 5% of those episodes; a request field is cold when every kind
# that declares it is listed here.
_COLD_GEOMETRY_KINDS = frozenset({
    'fk', 'camera_pose', 'pixel_ray', 'ray_surface', 'project_geometry', 'preview_ee', 'sparse_track', 'build_scene',
    'alignment', 'help', 'scene3d',
})
#: ``{tool: predicate(path, forms that declare the field) -> True when cold}`` for ``EVIDENCE_ROBOT``.
COLD = {
    'robot_action': lambda path, forms: _listed(path, _COLD_ROBOT_ACTION),
    'geometry_query': lambda path, forms: (path.startswith('request.') and path != 'request.kind'
                                            and bool(forms) and forms <= _COLD_GEOMETRY_KINDS),
}
#: ``{tool: {path of a union: labels of its rarely used forms}}`` for ``EVIDENCE_ROBOT``.
COLD_FORMS = {'geometry_query': {'request': _COLD_GEOMETRY_KINDS}}
#: Tools a deferring layered profile publishes for on-demand discovery (used in under 5% of those episodes).
COLD_TOOLS = frozenset({'robot_contact_overlay', 'history_list_actions', 'operation_memory'})


def _listed(path, names):
    """True when ``path`` or one of its ancestors (``a[].b`` -> ``a[]`` -> ``a``) is in ``names``."""
    while path:
        if path in names:
            return True
        cut = max(path.rfind('.'), path.rfind('['))
        if cut <= 0:
            return False
        path = path[:cut]
    return False


def cold_fields(tool, robot=EVIDENCE_ROBOT):
    """The cold predicate of ``tool`` for ``robot``, or None (no evidence for that robot or tool: nothing is cold)."""
    return COLD.get(tool) if robot in EVIDENCE_ROBOTS else None


def cold_forms(tool, robot=EVIDENCE_ROBOT):
    """``{union path: labels of its rarely used forms}`` of ``tool`` for ``robot`` (empty without evidence)."""
    return COLD_FORMS.get(tool, {}) if robot in EVIDENCE_ROBOTS else {}


def gist(text, limit=GIST_LIMIT):
    """Start of a help text: ``text`` itself when it has at most ``limit`` characters, else its words that fit,
    ending in ``...``. Returns ``(text, cut)``."""
    if len(text) <= limit:
        return text, False
    end = text.rfind(' ', 0, limit)
    return text[:end if end > 0 else limit].rstrip(' .;,:(') + ' ...', True


def cold_tools(robot=EVIDENCE_ROBOT, profile='layered_v1'):
    """Names of the tools ``profile`` publishes for on-demand discovery for ``robot``: ``COLD_TOOLS`` for a layered
    profile that defers (``LAYERS[...]['defer']``) and the robot with evidence, else none."""
    defers = LAYERS.get(profile, {}).get('defer', False)
    return COLD_TOOLS if defers and robot in EVIDENCE_ROBOTS else frozenset()


# ---- signatures ------------------------------------------------------------------------------------------------------
_KEYWORDS = frozenset({
    'type', 'description', 'properties', 'required', 'additionalProperties', 'items', 'prefixItems', 'enum', 'const',
    'default', 'examples', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'minLength', 'maxLength',
    'pattern', 'minItems', 'maxItems', 'uniqueItems', 'minProperties', 'maxProperties', 'dependentRequired',
    'oneOf', 'anyOf', 'allOf', 'not', 'if', 'then', 'else'})
#: Keywords a signature states; two declarations of one field must not give one of them different values.
_SIGNATURE_KEYS = frozenset({
    'type', 'enum', 'const', 'default', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'minLength',
    'maxLength', 'pattern', 'items', 'prefixItems', 'minItems', 'maxItems', 'uniqueItems'})


def _g(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return canonical(value)


def _bounds(node):
    lo, hi = node.get('minimum'), node.get('maximum')
    lox, hix = node.get('exclusiveMinimum'), node.get('exclusiveMaximum')
    if lo is not None and hi is not None and lox is None and hix is None:
        return f' {_g(lo)}..{_g(hi)}'
    parts = []
    if lox is not None:
        parts.append('>' + _g(lox))
    if lo is not None:
        parts.append('>=' + _g(lo))
    if hix is not None:
        parts.append('<' + _g(hix))
    if hi is not None:
        parts.append('<=' + _g(hi))
    return ' ' + ' '.join(parts) if parts else ''


def _count(node, low, high, unit):
    lo, hi = node.get(low), node.get(high)
    if lo is None and hi is None:
        return ''
    if lo is not None and lo == hi:
        return f' x{lo}' if unit == 'items' else f' {lo} {unit}'
    if lo is not None and hi is not None:
        return f' {lo}..{hi} {unit}'
    return f' >={lo} {unit}' if lo is not None else f' <={hi} {unit}'


_TYPE_KEYWORDS = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'minLength', 'maxLength', 'pattern',
                  'items', 'prefixItems', 'minItems', 'maxItems', 'uniqueItems')


def _admits(node, value):
    """True when the bound and length keywords of ``node`` admit ``value`` (a member of its value set)."""
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return not (('minimum' in node and value < node['minimum']) or ('maximum' in node and value > node['maximum'])
                    or ('exclusiveMinimum' in node and value <= node['exclusiveMinimum'])
                    or ('exclusiveMaximum' in node and value >= node['exclusiveMaximum']))
    if isinstance(value, str):
        return not (('minLength' in node and len(value) < node['minLength'])
                    or ('maxLength' in node and len(value) > node['maxLength'])
                    or ('pattern' in node and re.search(node['pattern'], value) is None))
    return not any(key in node for key in ('minItems', 'maxItems', 'uniqueItems', 'items', 'prefixItems'))


def _typed(node, kind):
    if kind in ('number', 'integer'):
        return ('int' if kind == 'integer' else 'number') + _bounds(node)
    if kind == 'string':
        text = 'string' + _count(node, 'minLength', 'maxLength', 'chars')
        return text + (' pattern ' + _g(node['pattern']) if 'pattern' in node else '')
    if kind == 'boolean':
        return 'bool'
    if kind == 'array':
        prefix, items = node.get('prefixItems'), node.get('items')
        if prefix:
            parts = [signature(item) or 'any' for item in prefix]
            limit = node.get('maxItems')
            open_tail = items is not False and (limit is None or limit > len(prefix))
            if open_tail:
                parts.append('...' + ((signature(items) if isinstance(items, dict) else '') or 'any'))
            most = limit if open_tail else len(prefix) if limit is None else min(limit, len(prefix))
            count = dict(minItems=node.get('minItems'), maxItems=most)
            fixed = len(prefix) > 1 and count['minItems'] == most == len(prefix)
            text = '[' + ', '.join(parts) + ']' + ('' if fixed else _count(count, 'minItems', 'maxItems', 'items'))
        else:
            text = '[' + ((signature(items) if isinstance(items, dict) else '') or 'any') + ']' + _count(
                node, 'minItems', 'maxItems', 'items')
        return text + (' unique' if node.get('uniqueItems') else '')
    return str(kind)


def signature(node):
    """Type, value set, bounds and default of one schema node as text ('' for a node that states none of them).

    ``a..b`` inclusive range, ``>a`` / ``<b`` exclusive and ``>=a`` / ``<=b`` inclusive bounds, ``[T]`` list of T with
    ``xN`` (exactly N), ``a..b items`` or one-sided item counts, ``[A, B]`` a fixed tuple, ``a|b`` allowed values,
    ``=v`` default."""
    if not isinstance(node, dict):
        return ''
    if 'const' in node or 'enum' in node:
        values = [node['const']] if 'const' in node else list(node['enum'])
        if not all(_admits(node, value) for value in values):
            raise UnsupportedSchema('value set narrowed by another keyword')
        text = '|'.join(_g(value) for value in values)
    else:
        kind = node.get('type')
        if not kind and any(key in node for key in _TYPE_KEYWORDS):
            # Without a type these keywords constrain only values of their own type; a typed signature would say more.
            raise UnsupportedSchema('bounds on a node without a type')
        kinds = kind if isinstance(kind, list) else [kind] if kind else []
        text = ' or '.join(_typed(node, item) for item in kinds)
    if 'default' in node:
        text += ' =' + _g(node['default'])
    return text.strip()


# ---- conditions and rules ------------------------------------------------------------------------------------------
def _values(value):
    """Raw values a property schema fixes (``const`` / ``enum``), or None."""
    if isinstance(value, dict) and 'const' in value:
        return [value['const']]
    if isinstance(value, dict) and value.get('enum'):
        return list(value['enum'])
    return None


def _value_text(values):
    return '|'.join(_g(value) for value in values)


def _selector(node):
    """``{field: raw values}`` when ``node`` is a condition on constant property values only (an ``if`` that selects
    a form), else None."""
    if not isinstance(node, dict) or set(node) - {'properties', 'required', 'type'}:
        return None
    found = {}
    for field, value in (node.get('properties') or {}).items():
        values = _values(value)
        if values is None or set(value) - {'const', 'enum', 'type', 'description'}:
            return None
        found[field] = values
    if not found or set(node.get('required', [])) - set(found):
        return None
    return found


def _is_group(text):
    """True when ``text`` is one parenthesised group."""
    if not (text.startswith('(') and text.endswith(')')):
        return False
    depth = 0
    for index, char in enumerate(text):
        depth += char == '('
        depth -= char == ')'
        if depth == 0 and index < len(text) - 1:
            return False
    return True


def _compound(text):
    """True when ``text`` joins terms with and/or outside parentheses."""
    depth, index = 0, 0
    while index < len(text):
        char = text[index]
        depth += char == '('
        depth -= char == ')'
        if depth == 0 and (text.startswith(' and ', index) or text.startswith(' or ', index)):
            return True
        index += 1
    return False


def _group(text):
    return text if _is_group(text) or not _compound(text) else '(' + text + ')'


def _field_test(field, value, required):
    """Text of one property test of a condition. ``required``: the condition also requires the field, so the test
    fails when it is omitted; otherwise an omitted field passes (JSON Schema ``properties`` semantics)."""
    keys = set(value) - {'type', 'description'} if isinstance(value, dict) else None
    negated = value.get('not') if keys == {'not'} else None
    if keys == {'const'}:
        test, positive = f'{field}={_g(value["const"])}', True
    elif keys == {'enum'}:
        test, positive = f'{field} in {_g(value["enum"])}', True
    elif isinstance(negated, dict) and set(negated) - {'type'} == {'const'}:
        test, positive = f'{field}!={_g(negated["const"])}', False
    elif isinstance(negated, dict) and set(negated) - {'type'} == {'enum'}:
        test, positive = f'{field} not in {_g(negated["enum"])}', False
    else:
        raise UnsupportedSchema(f'condition on {field}')
    if not required:
        return f'({field} omitted or {test})'
    return test if positive else f'{field} given and {test}'  # a positive test already implies the field is given


def _condition(node):
    """Text of a condition schema (``if`` / ``not``): field values, field presence and their combinations."""
    if not isinstance(node, dict):
        raise UnsupportedSchema('condition')
    unknown = set(node) - {'properties', 'required', 'anyOf', 'allOf', 'oneOf', 'not', 'type'}
    if unknown:
        raise UnsupportedSchema('condition keyword ' + sorted(unknown)[0])
    properties, required = node.get('properties') or {}, list(node.get('required', []))
    terms = [_field_test(field, value, field in required) for field, value in properties.items()]
    given = [field for field in required if field not in properties]
    if given:
        terms.append(', '.join(given) + ' given')
    for key, word in (('anyOf', ' or '), ('allOf', ' and ')):
        parts = [_group(_condition(part)) for part in node.get(key, [])]
        if parts:
            terms.append(parts[0] if len(parts) == 1 else '(' + word.join(parts) + ')')
    if node.get('oneOf'):
        terms.append('exactly one of (' + ' | '.join(_condition(part) for part in node['oneOf']) + ')')
    if 'not' in node:
        terms.append('not ' + _parenthesised(_condition(node['not'])))
    return ' and '.join(terms) if terms else 'always'


def _parenthesised(text):
    return text if _is_group(text) else '(' + text + ')'


def _negation(node):
    """Compact text of ``not: node`` for the idioms the catalogue uses, else the negated condition."""
    keys = set(node) - {'type'} if isinstance(node, dict) else set()
    if keys == {'required'}:
        fields = node['required']
        if len(fields) == 1:
            return 'without ' + fields[0]
        return ('not both ' + ' + '.join(fields)) if len(fields) == 2 else ('not all of ' + ', '.join(fields))
    if keys == {'anyOf'} and all(isinstance(part, dict) and set(part) == {'required'} and len(part['required']) == 1
                                 for part in node['anyOf']):
        return 'none of ' + ', '.join(part['required'][0] for part in node['anyOf'])
    return 'not ' + _parenthesised(_condition(node))


def _conditional_form(part):
    """``{field: raw values}`` when the ``allOf`` item ``part`` is ``if <constant field values> then <form>`` with
    nothing else (an ``else`` makes it an ordinary conditional rule), else None."""
    if not isinstance(part, dict) or set(part) != {'if', 'then'} or not isinstance(part['then'], dict):
        return None
    return _selector(part['if'])


def _oneof_constants(node):
    """Per ``oneOf`` branch ``{field: raw values}`` restricted to the fields every branch fixes; None when the
    ``oneOf`` is not a discriminated union."""
    parts = node.get('oneOf') or []
    if not parts or not all(isinstance(part, dict) for part in parts):
        return None
    constants = [{field: values for field, value in (part.get('properties') or {}).items()
                  if (values := _values(value)) is not None} for part in parts]
    shared = [field for field in constants[0] if all(field in found for found in constants)]
    return [{field: found[field] for field in shared} for found in constants] if shared else None


def _union(node):
    """The discriminated union declared at ``node``, or None.

    Two forms: ``oneOf`` branches that all fix the same fields to constants, and ``allOf`` items
    ``if <constant field values> then <form>``. Returns ``kind`` (``oneOf`` / ``allOf``), ``fields`` (the selecting
    fields), ``values`` (``{field: [value text]}`` over all forms), ``forms`` ``[(label, selector text, schema)]`` and
    ``sealed``: every form is closed and every valid object is in a form, so a field can occur only where a form
    declares it."""
    one = _oneof_constants(node)
    conditional = [(found, part) for part in node.get('allOf') or [] if (found := _conditional_form(part)) is not None]
    if one and conditional:
        raise UnsupportedSchema('two discriminated unions on one object')
    if one:
        kind, pairs = 'oneOf', [(found, part, set(part.get('required', []))) for found, part in zip(one, node['oneOf'])]
    elif conditional:
        kind, pairs = 'allOf', [(found, part['then'], set(part['if'].get('required', []))) for found, part in conditional]
    else:
        return None
    fields = list(dict.fromkeys(field for found, _, _ in pairs for field in found))
    forms, values = [], {field: [] for field in fields}
    for found, schema, required in pairs:
        label = '+'.join('|'.join(str(value) for value in found[field]) for field in found)
        selector = ', '.join(f'{field}={_value_text(found[field])}' + ('' if field in required else f' (or {field} omitted)')
                             for field in found)
        forms.append((label, selector, schema))
        for field in found:
            values[field].extend(text for value in found[field] if (text := _g(value)) not in values[field])
    if len({label for label, _, _ in forms}) != len(forms):
        raise UnsupportedSchema('two forms with the same selecting values')
    closed = all(schema.get('additionalProperties') is False for _, _, schema in forms)
    if kind == 'oneOf':
        exhaustive = True  # a valid object matches exactly one branch
    else:
        base, required = node.get('properties') or {}, set(node.get('required', []))
        exhaustive = all(field in required and _values(base.get(field)) is not None
                         and {_g(value) for value in _values(base[field])} <= set(values[field]) for field in fields)
    return dict(kind=kind, fields=fields, values=values, forms=forms, sealed=closed and exhaustive)


def _clauses(node, *, own_fields=False, skip=()):
    """Structural rules of an object schema as a list of clauses.

    Value sets of declared properties are stated by their field lines, not here. ``own_fields``: name the optional
    fields too (a form of a discriminated union); ``skip``: fields the caller already states (the selecting fields)."""
    clauses = []
    required = [field for field in node.get('required', []) if field not in skip]
    if required:
        clauses.append('requires ' + ', '.join(required))
    properties = node.get('properties') or {}
    if own_fields:
        optional = [field for field in properties if field not in node.get('required', []) and field not in skip]
        if optional:
            clauses.append('optional ' + ', '.join(optional))
    for field, value in properties.items():
        negated = value.get('not') if isinstance(value, dict) else None
        if isinstance(negated, dict):
            if set(negated) - {'type'} == {'const'}:
                clauses.append(f'{field} must not be {_g(negated["const"])}')
            elif set(negated) - {'type'} == {'enum'}:
                clauses.append(f'{field} must not be one of {_g(negated["enum"])}')
            else:
                raise UnsupportedSchema(f'negated declaration of {field}')
    if node.get('additionalProperties') is False:
        clauses.append('closed')
    elif isinstance(node.get('additionalProperties'), dict):
        raise UnsupportedSchema('additionalProperties schema')
    for key, word in (('minProperties', 'at least'), ('maxProperties', 'at most')):
        if key in node:
            clauses.append(f'{word} {node[key]} fields')
    by_need = {}
    for field, needs in (node.get('dependentRequired') or {}).items():
        by_need.setdefault(', '.join(needs), []).append(field)
    for needs, fields in by_need.items():
        clauses.append(('each of ' if len(fields) > 1 else '') + ', '.join(fields) + ' needs ' + needs)
    if node.get('oneOf'):
        if _oneof_constants(node):
            clauses.append('exactly one of the forms below')
        else:
            clauses.append('exactly one of (' + ' | '.join(
                '; '.join(_clauses(part)) or 'no restriction' for part in node['oneOf']) + ')')
    if node.get('anyOf'):
        clauses.append('at least one of (' + ' | '.join(
            '; '.join(_clauses(part)) or 'no restriction' for part in node['anyOf']) + ')')
    for part in node.get('allOf', []):
        if _conditional_form(part) is None:  # a form is rendered on its own line
            clauses.extend(_clauses(part))
    if isinstance(node.get('not'), dict):
        clauses.append(_negation(node['not']))
    if 'if' in node:
        then = '; '.join(_clauses(node['then'])) if isinstance(node.get('then'), dict) else ''
        other = '; '.join(_clauses(node['else'])) if isinstance(node.get('else'), dict) else ''
        if then or other:
            rule = f'if {_condition(node["if"])} then ({then or "no restriction"})'
            clauses.append(rule + (f' else ({other})' if other else ''))
    return clauses


# ---- guide ---------------------------------------------------------------------------------------------------------
def _own_line(node):
    """A nested schema node needs its own guide line when it states more than a foldable signature."""
    return isinstance(node, dict) and (node.get('additionalProperties') is False or any(key in node for key in (
        'description', 'properties', 'oneOf', 'anyOf', 'allOf', 'if', 'not', 'dependentRequired', 'required',
        'minProperties', 'maxProperties')))


def _check_keywords(node):
    """Refuse a schema keyword this module does not render (it would otherwise vanish from the contract text)."""
    if not isinstance(node, dict):
        return
    unknown = set(node) - _KEYWORDS
    if unknown:
        raise UnsupportedSchema('keyword ' + sorted(unknown)[0])
    for value in (node.get('properties') or {}).values():
        _check_keywords(value)
    for key in ('items', 'not', 'if', 'then', 'else', 'additionalProperties'):
        _check_keywords(node.get(key))
    for key in ('oneOf', 'anyOf', 'allOf', 'prefixItems'):
        for value in node.get(key) or []:
            _check_keywords(value)


def _declares(node):
    return isinstance(node, dict) and any(key in node for key in ('properties', 'items', 'prefixItems'))


def _collect(schema):
    """Walk ``schema`` like ``definitions._with_input_guide``: ``{path: entry}`` in declaration order.

    entry: ``help`` ``{text: forms}`` and ``rules`` ``{clause text: forms}``, where a form is the label of the
    enclosing union form the text was declared in (None = outside the forms); ``nodes`` ``{form: [schema node]}``;
    ``union`` (``_union`` of the object at this path, or None). Raises ``UnsupportedSchema``."""
    _check_keywords(schema)
    entries = {}

    def visit(node, path, form, structural=True):
        if not isinstance(node, dict):
            return
        item = entries.setdefault(path or 'input', dict(help={}, nodes={}, rules={}, union=None))
        item['nodes'].setdefault(form, []).append(node)
        if node.get('description'):
            item['help'].setdefault(node['description'], set()).add(form)
        union = _union(node)
        if union is not None:
            if form is not None or not structural or item['union'] is not None:
                raise UnsupportedSchema('nested discriminated union at ' + (path or 'input'))
            item['union'] = union
        if structural:
            text = '; '.join(_clauses(node))
            if text:
                item['rules'].setdefault(text, set()).add(form)
        for field, child in (node.get('properties') or {}).items():
            visit(child, path + '.' + field if path else field, form)
        if _own_line(node.get('items')):
            visit(node['items'], path + '[]', form)
        for index, child in enumerate(node.get('prefixItems', [])):
            if _own_line(child):
                visit(child, f'{path}[{index}]', form)
        forms = {id(branch): label for label, _, branch in union['forms']} if union else {}
        for key in ('oneOf', 'anyOf'):
            for part in node.get(key, []):
                if id(part) in forms:
                    visit(part, path, forms[id(part)], structural=False)
                elif _declares(part):
                    # A plain alternative that declares fields: their constraints would hold only in that alternative.
                    raise UnsupportedSchema(f'field declarations in an {key} alternative at ' + (path or 'input'))
                else:
                    visit(part, path, form, structural=False)
        for part in node.get('allOf', []):
            if isinstance(part, dict) and id(part.get('then')) in forms:
                visit(part['then'], path, forms[id(part['then'])], structural=False)
            else:
                visit(part, path, form, structural=False)
        for branch in ('then', 'else'):
            target = node.get(branch)
            if not isinstance(target, dict):
                continue
            if _declares(target):
                # Only "if X is given then X satisfies ..." equals a plain declaration of X.
                condition = node.get('if')
                own = isinstance(condition, dict) and set(condition) == {'required'}
                if not (branch == 'then' and own and set(target) == {'properties'}
                        and set(target['properties']) <= set(condition['required'])):
                    raise UnsupportedSchema('conditional field declarations at ' + (path or 'input'))
            visit(target, path, form, structural=False)

    visit(schema, '', None)
    return entries


def _merged(nodes, path):
    """One declaration from several of the same field (the base and a form's, or a declaration split over ``allOf``):
    their signature keywords together; the same keyword with two values cannot be stated as one signature."""
    merged = {}
    for node in nodes:
        for key in _SIGNATURE_KEYS.intersection(node):
            if key in merged and merged[key] != node[key]:
                raise UnsupportedSchema(f'two values of {key} for {path}')
            merged[key] = node[key]
    return merged


def _signatures(item, union, path):
    """``{signature: forms}`` of one entry. A form's signature is the declaration outside the forms together with
    the form's own (both constrain the value). In a sealed union a field occurs only where a form declares it, so
    the declaration outside the forms is not listed by itself."""
    nodes = item['nodes']
    forms = [form for form in nodes if form is not None]
    base = _merged(nodes.get(None, []), path)
    found = {}
    if None in nodes and not (forms and union is not None and union['sealed']):
        text = signature(base)
        if text:
            found.setdefault(text, set()).add(None)
    for form in forms:
        text = signature(_merged([base, *nodes[form]], path))
        if text:
            found.setdefault(text, set()).add(form)
    return found


def _qualified(values, scope, notes, universe=frozenset()):
    """Texts of ``{text: forms}``. A text declared outside the forms is given as is, and so is one declared in every
    form that declares the field (``scope``) when nothing is declared outside the forms; any other names the forms
    it holds in (``when a|b:``). When every form of the
    enclosing union (``universe``) declares the field and the exceptions are fewer, it names those instead
    (``except c:`` = in every other form)."""
    parts = []
    outside = any(None in forms for forms in values.values())
    for text, forms in values.items():
        named = {form for form in forms if form is not None}
        if None in forms or (named >= scope and not outside):
            parts.append(text)
            continue
        notes.add('forms')
        missing = scope - named
        if scope >= universe and missing and len(missing) < len(named):
            parts.append('except ' + '|'.join(sorted(missing)) + ': ' + text)
        else:
            parts.append('when ' + '|'.join(sorted(named)) + ': ' + text)
    return parts


def _parent(path, entries):
    if path.endswith(']'):
        candidate = path[:path.rfind('[')]
    elif '.' in path:
        candidate = path[:path.rfind('.')]
    else:
        candidate = 'input'
    return entries.get(candidate) if candidate != path else None


_TUPLE = re.compile(r'\[[^\[\]]*, ')


def _note_signature(head, notes):
    for key, test in (('list', '[' in head), ('range', '..' in head), ('default', ' =' in head), ('values', '|' in head),
                      ('count', '] x' in head), ('bound', any(mark in head for mark in (' >', ' <'))),
                      ('tuple', _TUPLE.search(head) is not None), ('alternatives', ' / ' in head)):
        if test:
            notes.add(key)


def _ancestors(path):
    while True:
        cut = max(path.rfind('.'), path.rfind('['))
        if cut <= 0:
            return
        path = path[:cut]
        yield path


def _render(schema, cold=None, forms=None, layer=None):
    """``(lines, cold entries, cold forms, notation keys used)`` of ``schema``; see ``guide_lines``.

    cold entry: ``(path, signature, help text)``; cold form: ``(union path, selector text)``. ``layer``: a ``LAYERS``
    value deciding which cold entries and form lines are kept (None = the ``layered_v1`` behaviour)."""
    layer = layer or LAYERS['layered_v1']
    forms = forms or {}
    entries = _collect(schema)
    lines, cold_entries, cold_form_items, notes, cold_paths = [], [], [], set(), set()
    for path, item in entries.items():
        parent = _parent(path, entries)
        outer = parent['union'] if parent else None
        scope = {form for form in item['nodes'] if form is not None}
        universe = {label for label, _, _ in outer['forms']} if outer else frozenset()
        name = path.rsplit('.', 1)[-1]
        used = set()
        if path == 'input':
            head = ''
        elif outer is not None and name in outer['fields'] and outer['kind'] == 'allOf':
            # The selecting field of conditional forms: its declaration outside the forms gives every allowed value
            # (the forms select some of them and restate their own).
            head = signature(_merged(item['nodes'].get(None, []), path))
        elif outer is not None and name in outer['fields']:
            # The selecting field of a oneOf union: its values are the forms.
            declared = [node for nodes in item['nodes'].values() for node in nodes]
            if None in item['nodes'] or any('default' in node for node in declared):
                raise UnsupportedSchema(f'selecting field {path} declared outside the forms or with a default')
            head = '|'.join(outer['values'][name])
        else:
            head = ' / '.join(_qualified(_signatures(item, outer, path), scope, used, universe))
        body = ' '.join(_qualified(item['help'], scope, used, universe))
        rules = ' / '.join(_qualified(item['rules'], scope, used, universe))
        if ' / ' in rules:
            used.add('alternatives')
        union = item['union']
        if not (body or rules or union) and head in ('', 'object'):
            continue  # nothing beyond what the parent's signature already says
        if cold and path != 'input' and cold(path, scope):
            cold_paths.add(path)
            if layer['nested'] or not any(above in cold_paths for above in _ancestors(path)):
                cold_entries.append((path, head, body))
                _note_signature(head, notes)
            continue
        text = path + ':' + (' ' + head if head else '')
        if body:
            text += ' -- ' + body
        if rules:
            text += (' ' if body else ' -- ') + 'Rules: ' + rules + '.'
        lines.append(text)
        form_lines = []
        for label, selector, form in (union['forms'] if union else ()):
            if not layer['cold_forms'] and label in forms.get(path, ()):
                cold_form_items.append((path, selector))
                continue
            clauses = _clauses(form, own_fields=True, skip=union['fields'])
            form_lines.append(f'  {path} when {selector}: ' + ('; '.join(clauses) or 'no further rule') + '.')
        lines.extend(form_lines)
        notes.update(used)
        _note_signature(head, notes)
        technical = rules + ' '.join(form_lines)
        if 'closed' in technical:
            notes.add('closed')
        if ' needs ' in technical:
            notes.add('needs')
        if form_lines:
            notes.add('forms')
    return lines, cold_entries, cold_form_items, notes


def guide_lines(schema, cold=None):
    """Guide lines of ``schema``: ``(lines, cold items)``.

    ``cold``: ``predicate(path, forms that declare the field) -> bool`` naming the fields that are returned as
    ``path: signature`` in the second list instead of a described line, or None to describe every field. Raises
    ``UnsupportedSchema`` for a schema this notation cannot state."""
    lines, cold_entries, _, _ = _render(schema, cold)
    return lines, [path + (': ' + head if head else '') for path, head, _ in cold_entries]


def _header(notes):
    used = [text for key, text in _NOTATION if key in notes]
    return _INTRO + (' Notation: ' + '; '.join(used) + '.' if used else '')


def _grep_example(path):
    return path.replace('[', '\\[').replace(']', '\\]')


def with_guide(item, profile, robot=EVIDENCE_ROBOT):
    """Append the guide of ``item['inputSchema']`` to ``item['description']`` (``compact_v1`` or a ``LAYERS`` profile).

    ``robot``: the run's robot model; a layered profile lists cold fields only for ``EVIDENCE_ROBOT``. Raises
    ``UnsupportedSchema`` without changing ``item``."""
    layer = LAYERS.get(profile)
    cold = cold_fields(item['name'], robot) if layer else None
    lines, cold_entries, cold_form_items, notes = _render(
        item['inputSchema'], cold, cold_forms(item['name'], robot) if layer else None, layer)
    text = ''
    if lines or cold_entries:
        text += '\n' + _header(notes) + '\n' + '\n'.join(lines)
    for union in dict.fromkeys(path for path, _ in cold_form_items):
        text += '\n' + COLD_FORMS_HEADER.format(name=REFERENCE_NAME, path=union, forms='; '.join(
            selector for path, selector in cold_form_items if path == union))
    if cold_entries and layer['gist']:
        header = COLD_GIST_HEADER if layer['command'] else COLD_LINE_HEADER
        text += '\n' + header.format(name=REFERENCE_NAME, example=_grep_example(cold_entries[0][0]))
        for path, head, body in cold_entries:
            start = gist(body)[0] if body else ''
            text += '\n' + path + ':' + (' ' + head if head else '') + (' -- ' + start if start else '')
    elif cold_entries:
        text += ('\n' + COLD_HEADER.format(name=REFERENCE_NAME, example=_grep_example(cold_entries[0][0])) + '\n'
                 + '; '.join(path + (': ' + head if head else '') for path, head, _ in cold_entries))
    item['description'] += text
    return item


def reference_document(catalog, robot=EVIDENCE_ROBOT):
    """The tool reference text of a layered catalogue: the complete guide of every tool that lists cold fields or
    forms, or None when no tool does. ``catalog``: registry tool definitions (``name``, ``inputSchema``)."""
    sections = []
    for item in catalog:
        cold = cold_fields(item.get('name'), robot)
        if cold is None:
            continue
        try:
            if not _render(item['inputSchema'], cold)[1]:
                continue
            lines, _, _, notes = _render(item['inputSchema'])
        except UnsupportedSchema:
            continue  # published with the prose guide, which lists no cold fields
        sections.append(f'## {item["name"]}\n' + _header(notes) + '\n' + '\n'.join(lines))
    if not sections:
        return None
    return ('# Tool reference\nComplete input field text of the tools whose descriptions shorten rarely needed fields '
            'and forms. One line per field, starting with its path; a line "  <path> when <field>=<value>:" gives the '
            'rules of one form.\n\n' + '\n\n'.join(sections) + '\n')


def share_lines(tools):
    """Tool definitions in which a field line that an earlier loaded tool of the same catalogue states identically
    (same path, same text of at least ``SHARED_MIN`` characters) is replaced by ``<path>: same as in <tool>``.

    ``tools``: the catalogue as published (after availability filtering), so a reference never names a tool that is
    absent or not loaded; tools published for on-demand discovery neither give nor take references. Only the typed
    guide (after its header line) is read; everything else is returned unchanged."""
    seen, result = {}, []
    for tool in tools:
        description = tool.get('description')
        if tool.get('deferLoading') or not isinstance(description, str) or ('\n' + _INTRO) not in description:
            result.append(tool)
            continue
        head, _, guide = description.partition('\n' + _INTRO)
        rows = guide.split('\n')
        stop = next((index for index, row in enumerate(rows) if row.startswith(_COLD_MARK)), len(rows))
        for index in range(1, stop):
            path, mark, text = rows[index].partition(': ')
            if not mark or path.startswith(' ') or len(text) < SHARED_MIN:
                continue
            owner = seen.setdefault((path, text), tool['name'])
            if owner != tool['name']:
                rows[index] = path + ': ' + SHARED_TEXT.format(tool=owner)
        result.append(dict(tool, description=head + '\n' + _INTRO + '\n'.join(rows)))
    return result


def on_demand_index(tools):
    """Instruction section naming the tools of ``tools`` (the published catalogue) that are loaded on demand, one
    line each with the first sentence of its description; None when every tool is loaded."""
    rows = []
    for tool in tools:
        if tool.get('deferLoading'):
            sentence = (tool.get('description') or '').split('\n', 1)[0].split('. ', 1)[0].rstrip('.')
            rows.append(f'- {tool["name"]}: {sentence}.')
    return ON_DEMAND_HEADER + '\n' + '\n'.join(rows) + '\n' if rows else None


from ...core.schema_guide import strip_guided_schema_descriptions  # noqa: F401
