"""JSON Schema 与工具描述中 ``Input field guide`` 的纯数据同路径去重辅助函数（零上层依赖）。"""

from copy import deepcopy

_GUIDE_MARKER = 'Input field guide'
_COLD_MARK = 'Rarely needed '
_COLD_FIELDS_MARK = 'Rarely needed fields'
_COLD_LIST_HINT = 'listed with type and bounds only'
_SHARED_PREFIX = 'same as in '


def _parse_guide(description):
    """提取工具描述中 ``Input field guide`` 之后的字段路径行映射与冷字段路径集合；无指南标记时返回 ``(None, frozenset())``。"""
    if not isinstance(description, str):
        return None, frozenset()
    marker_pos = description.find('\n' + _GUIDE_MARKER)
    if marker_pos >= 0:
        guide_text = description[marker_pos + 1:]
    elif description.startswith(_GUIDE_MARKER):
        guide_text = description
    else:
        return None, frozenset()
    lines_by_path = {}
    cold_paths = set()
    cold_mode = None
    for row in guide_text.splitlines()[1:]:
        if not row or row.startswith(' '):
            continue
        if row.startswith(_COLD_MARK):
            if row.startswith(_COLD_FIELDS_MARK):
                cold_mode = 'list' if _COLD_LIST_HINT in row else 'lines'
            continue
        if cold_mode == 'list':
            for entry in row.split('; '):
                path = entry.partition(':')[0].strip()
                if path:
                    cold_paths.add(path)
            continue
        path, sep, rest = row.partition(': ')
        if sep and path:
            lines_by_path.setdefault(path, []).append(rest)
            if cold_mode == 'lines':
                cold_paths.add(path)
    return lines_by_path, frozenset(cold_paths)


def _guide_lines_by_path(description):
    """提取工具描述中 ``Input field guide`` 之后的字段路径行映射；无指南标记时返回 None。"""
    return _parse_guide(description)[0]


def _is_cold_path(path, cold_paths):
    """判断 ``path`` 或其任意祖先路径是否已在分层指南的 ``Rarely needed fields`` 区块声明。"""
    while path:
        if path in cold_paths:
            return True
        cut = max(path.rfind('.'), path.rfind('['))
        if cut <= 0:
            return False
        path = path[:cut]
    return False


def strip_guided_schema_descriptions(schema, description):
    """传输层同路径去重：仅沿 JSON Schema 结构节点递归，剔除已被工具 ``Input field guide``
    同路径逐字说明、以 ``same as in <tool>`` 共享引用，或在分层模式（``layered_v1``/``layered_v2``/``layered_v3``）
    ``Rarely needed fields`` 区块列出并交由 ``tool_reference.md`` 承载完整文档的字段及其嵌套子字段的 ``description``。

    不进入 ``examples``/``default``/``enum``/``const`` 等数据关键字，避免误删业务同名键；
    无 ``Input field guide`` 的插件工具保留全部原描述。
    """
    if not isinstance(schema, dict):
        return deepcopy(schema)
    lines_by_path, cold_paths = _parse_guide(description)
    if lines_by_path is None:
        return deepcopy(schema)

    def _matches(path, text):
        if cold_paths and _is_cold_path(path, cold_paths):
            return True
        candidates = lines_by_path.get(path or 'input', ())
        return any(text in rest or rest.startswith(_SHARED_PREFIX) for rest in candidates)

    def _walk(node, path):
        if not isinstance(node, dict):
            return deepcopy(node)
        desc = node.get('description')
        drop_desc = isinstance(desc, str) and _matches(path, desc)
        out = {}
        for key, value in node.items():
            if key == 'description' and drop_desc:
                continue
            if key == 'properties' and isinstance(value, dict):
                out[key] = {
                    field: _walk(child, f'{path}.{field}' if path else field)
                    for field, child in value.items()
                }
            elif key in ('patternProperties', '$defs', 'definitions', 'dependentSchemas') and isinstance(value, dict):
                out[key] = {name: _walk(child, '') for name, child in value.items()}
            elif key == 'items' and isinstance(value, dict):
                out[key] = _walk(value, f'{path}[]')
            elif key == 'prefixItems' and isinstance(value, list):
                out[key] = [
                    _walk(child, f'{path}[{idx}]') if isinstance(child, dict) else deepcopy(child)
                    for idx, child in enumerate(value)
                ]
            elif key in ('oneOf', 'anyOf', 'allOf') and isinstance(value, list):
                out[key] = [
                    _walk(child, path) if isinstance(child, dict) else deepcopy(child)
                    for child in value
                ]
            elif key in ('additionalProperties', 'then', 'else') and isinstance(value, dict):
                out[key] = _walk(value, path)
            else:
                out[key] = deepcopy(value)
        return out

    return _walk(schema, '')
