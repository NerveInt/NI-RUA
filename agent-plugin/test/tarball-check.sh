#!/usr/bin/env bash
# Tarball install check (design 20 section 5): npm pack, unpack, run the installer from the unpacked tarball into
# isolated homes on the pinned host binaries, check plugin / MCP entry / skills, uninstall, compare the home.
#
# usage: agent-plugin/test/tarball-check.sh [claude-pinned codex-pinned claude-user codex-user opencode-pinned]
# Env overrides (absolute paths): CLAUDE_PINNED CODEX_PINNED OPENCODE_PINNED CLAUDE_USER CODEX_USER
#   SPIKE_DIR  fixtures of the spike (default: none; a scratch directory holding the spike fixtures): real NiRuaMcpServer (mcp_server_spike.py,
#              run with `mamba run -n ni-rua python` from this checkout) and fake_model.py. Without it a tiny Node
#              MCP server is used and the "skills reach the model" checks are skipped.
# Never touches the real ~/.claude, ~/.codex, ~/.config/opencode: every host runs under `env -i` with a fresh HOME.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$HERE/.." && pwd)
REPO=$(cd "$PKG/.." && pwd)
NODE=$(command -v node)
SPIKE=${SPIKE_DIR:-}
VERSION=$("$NODE" -p 'require(process.argv[1]).version' "$PKG/package.json")
# the pinned hosts live in the ni-rua environment (claude-agent-sdk bundles Claude Code, codex-cli-bin bundles Codex)
SITE=$(mamba run -n ni-rua python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])' 2>/dev/null | tail -1)
CLAUDE_PINNED=${CLAUDE_PINNED:-$SITE/claude_agent_sdk/_bundled/claude}
CODEX_PINNED=${CODEX_PINNED:-$SITE/codex_cli_bin/bin/codex}
OPENCODE_PINNED=${OPENCODE_PINNED:-$HOME/.local/share/ni-rua/opencode-2.2.7/opencode}
CLAUDE_USER=${CLAUDE_USER:-$([ -x "$HOME/.local/bin/claude" ] && echo "$HOME/.local/bin/claude" || command -v claude || true)}
CODEX_USER=${CODEX_USER:-$([ -x "$HOME/.local/bin/codex" ] && echo "$HOME/.local/bin/codex" || command -v codex || true)}
WANT=${*:-claude-pinned codex-pinned claude-user codex-user opencode-pinned}
T=$(mktemp -d /tmp/ni-rua-tarball-check.XXXXXX)
PASS=0; FAIL=0; SKIP=0; PIDS=()
trap 'for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done' EXIT

say() { printf '%s\n' "$*"; }
ok() { PASS=$((PASS+1)); say "  PASS  $1"; }
bad() { FAIL=$((FAIL+1)); say "  FAIL  $1"; }
skip() { SKIP=$((SKIP+1)); say "  SKIP  $1"; }
check() { local what=$1; shift; if "$@" >/dev/null 2>&1; then ok "$what"; else bad "$what"; fi; }
has() { grep -qF -- "$2" <<<"$1"; }
free_port() { "$NODE" -e 'const s=require("net").createServer().listen(0,"127.0.0.1",()=>{console.log(s.address().port);s.close()})'; }
wait_port() { for _ in $(seq 1 100); do curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$1/mcp" -X GET && return 0; sleep 0.2; done; return 1; }

# ---- pack and unpack ---------------------------------------------------------------------------------------------
say "== npm pack"
mkdir -p "$T/pack" "$T/unpacked"
( cd "$PKG" && npm pack --pack-destination "$T/pack" --silent ) >"$T/pack.out" 2>&1 || { cat "$T/pack.out"; exit 2; }
TGZ=$(ls "$T"/pack/*.tgz)
tar xzf "$TGZ" -C "$T/unpacked"
UP="$T/unpacked/package"
say "tarball: $TGZ ($(tar tzf "$TGZ" | wc -l | tr -d ' ') files)"
check "tarball has no test/ directory" bash -c "! tar tzf '$TGZ' | grep -q '^package/test/'"
for f in package.json README.md LICENSE .mcp.json .codex-mcp.json codex-models.direct.json hooks/hooks.json hooks/on-compact.mjs manifest.generated.json .claude-plugin/plugin.json .claude-plugin/marketplace.json .codex-plugin/plugin.json opencode/plugin.js skills/ni-rua-robot/SKILL.md skills/ni-rua-experiments/SKILL.md bin/cli.mjs; do
  check "tarball has $f" test -f "$UP/$f"
done
CLI="$UP/bin/cli.mjs"

# ---- servers -----------------------------------------------------------------------------------------------------
start_mcp() {   # port -> starts the MCP server double; sets MCP_KIND
  if [ -n "$SPIKE" ] && [ -d "$SPIKE/scripts" ] && command -v mamba >/dev/null; then
    echo '{"slow_s": 3}' > "$T/control.json"
    ( cd "$REPO" && mamba run -n ni-rua python "$SPIKE/scripts/mcp_server_spike.py" --catalogue "$SPIKE/data/x5.json" --port "$1" \
        --log "$T/mcp-$1.jsonl" --control "$T/control.json" --images "$SPIKE/data/img" >"$T/mcp-$1.out" 2>&1 & )
    MCP_KIND="NiRuaMcpServer (spike fixture, X5 catalogue)"
  else
    "$NODE" "$HERE/fake-mcp-server.mjs" "$1" >"$T/mcp-$1.out" 2>&1 & PIDS+=($!)
    MCP_KIND="fake Node MCP server"
  fi
  wait_port "$1" || { say "MCP server on $1 did not start"; cat "$T/mcp-$1.out"; return 1; }
}
stop_mcp() { pkill -f "mcp_server_spike.py --catalogue .* --port $1" 2>/dev/null; pkill -f "fake-mcp-server.mjs $1" 2>/dev/null; sleep 0.5; }
start_model() {   # port logfile
  [ -n "$SPIKE" ] && [ -f "$SPIKE/scripts/fake_model.py" ] || return 1
  ( cd "$REPO" && mamba run -n ni-rua python "$SPIKE/scripts/fake_model.py" "$1" "$2" >"$T/model-$1.out" 2>&1 & )
  sleep 3
}
stop_model() { pkill -f "fake_model.py $1" 2>/dev/null; sleep 0.3; }

# ---- home snapshots ----------------------------------------------------------------------------------------------
# Host-written session bookkeeping (Claude's .claude.json / backups / projects, Codex sessions and logs, OpenCode's
# XDG data and cache) is not ours and not compared; plugin directories, settings and config files are.
snapshot() { ( cd "$1" && find . -type f ! -path './work/*' ! -path './.claude/.claude.json' ! -path './.claude/backups/*' \
    ! -path './.claude/projects/*' ! -path './.claude/plugins/store/*' ! -path './.claude/sessions/*' ! -path './.claude/cache/*' \
    ! -path './.cache/claude-cli-nodejs/*' \
    ! -path './codex/skills/.system/*' ! -path './codex/.sandbox_migration' ! -path './codex/installation_id' ! -path './codex/mcp-oauth-locks/*' \
    ! -path './codex/thread-writer-locks/*' ! -path './codex/sessions/*' ! -path './codex/log/*' ! -path './codex/*.sqlite*' ! -path './codex/.tmp/*' ! -path './codex/shell_snapshots/*' \
    ! -path './Library/*' ! -path './xdg/data/*' ! -path './xdg/cache/*' ! -path './xdg/state/*' -exec shasum {} + 2>/dev/null | sort -k2 ); }

# Host-owned residue that stays after a correct uninstall (the host's own bookkeeping, not ours). Anything else that
# differs from the snapshot taken before the install fails the check.
RESIDUE='^\./\.claude/(settings\.json|plugins/(installed_plugins\.json|known_marketplaces\.json|cache/ni_rua/.*))$'
finish_uninstall() {   # label home before-file
  snapshot "$2" > "$T/$1.after"
  local changed bad_paths=""
  changed=$(diff "$3" "$T/$1.after" | grep '^[<>]' | awk '{print $3}' | sort -u)
  for f in $changed; do
    if grep -qF " $f" "$3" && ! grep -qF " $f" "$T/$1.after" && ! [[ $f =~ $RESIDUE ]]; then bad_paths="$bad_paths $f(removed)"
    elif ! [[ $f =~ $RESIDUE ]]; then bad_paths="$bad_paths $f"; fi
  done
  if [ -z "$changed" ]; then ok "uninstall: home identical to before install"
  elif [ -z "$bad_paths" ]; then ok "uninstall: home differs from before install only by host-owned residue ($(echo $changed | wc -w | tr -d ' ') files under .claude/plugins and settings.json)"
  else bad "uninstall: unexpected differences from before install:$bad_paths"; fi
}

# ---- one host section --------------------------------------------------------------------------------------------
run_host() {
  local label=$1 host=$2 bin=$3
  say; say "== $label ($host) $bin"
  if [ ! -x "$bin" ]; then skip "$label: binary not found"; return; fi
  local version; version=$("$bin" --version 2>&1 | head -1); say "version: $version"
  local H="$T/$label/home"; mkdir -p "$H/work"
  local MPORT; MPORT=$(free_port); local FPORT; FPORT=$(free_port)
  local MODEL_LOG="$T/$label/model.jsonl"
  start_mcp "$MPORT" || { bad "$label: MCP server"; return; }
  say "MCP server: $MCP_KIND on $MPORT"
  local HAVE_MODEL=0
  start_model "$FPORT" "$MODEL_LOG" && HAVE_MODEL=1
  case $host in
    claude) cc_section "$label" "$bin" "$H" "$MPORT" "$FPORT" "$MODEL_LOG" "$HAVE_MODEL" ;;
    codex) cx_section "$label" "$bin" "$H" "$MPORT" "$FPORT" "$MODEL_LOG" "$HAVE_MODEL" ;;
    opencode) oc_section "$label" "$bin" "$H" "$MPORT" "$FPORT" "$MODEL_LOG" "$HAVE_MODEL" ;;
  esac
  stop_mcp "$MPORT"; stop_model "$FPORT"
}

cc_section() {
  local label=$1 bin=$2 H=$3 MPORT=$4 FPORT=$5 LOG=$6 HAVE=$7
  cc() { env -i PATH=/usr/bin:/bin HOME="$H" CLAUDE_CONFIG_DIR="$H/.claude" ANTHROPIC_BASE_URL="http://127.0.0.1:$FPORT" \
      ANTHROPIC_API_KEY=sk-ant-fake TERM=dumb CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 \
      DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 "$@"; }
  mkdir -p "$H/.claude"; snapshot "$H" > "$T/$label.before"
  local DUR="$H/.local/share/ni-rua/agent-plugin/$VERSION"
  # first host runs through npx from the tarball (the documented way); the rest call the unpacked cli directly
  local out rc
  if [ "$label" = claude-pinned ]; then
    out=$(cd "$H/work" && cc env PATH="$(dirname "$NODE"):/usr/bin:/bin" npm_config_cache="$T/$label/npm-cache" npx --yes --package "$TGZ" ni-rua-plugin install claude --claude-bin "$bin" --port "$MPORT" 2>&1)
    rc=$?; say "(installed with: npx --package <tarball> ni-rua-plugin install claude)"
  else
    out=$(cd "$H/work" && cc "$NODE" "$CLI" install claude --claude-bin "$bin" --port "$MPORT" 2>&1)
    rc=$?
  fi
  say "$out" | sed 's/^/    | /'
  [ $rc -eq 0 ] && ok "install exit 0" || { bad "install exit $rc"; return; }
  check "durable copy exists" test -f "$DUR/package.json"
  check "claude plugin validate passes" cc "$bin" plugin validate "$DUR"
  local list; list=$(cd "$H/work" && cc "$bin" plugin list --json)
  has "$list" '"id": "ni_rua@ni_rua"' && ok "plugin listed (ni_rua@ni_rua)" || bad "plugin listed"
  local details; details=$(cd "$H/work" && cc "$bin" plugin details ni_rua@ni_rua 2>&1)
  has "$details" "Skills (2)  ni-rua-experiments, ni-rua-robot" && ok "skills visible: ni-rua-experiments, ni-rua-robot" || bad "skills visible"
  local mcp; mcp=$(cd "$H/work" && cc "$bin" mcp list 2>&1)
  has "$mcp" "plugin:ni_rua:ni_rua: http://127.0.0.1:$MPORT/mcp (HTTP) - ✔ Connected" && ok "claude mcp list: plugin:ni_rua:ni_rua Connected on --port $MPORT" || { bad "claude mcp list connected"; say "$mcp" | sed 's/^/        /'; }
  local S="$H/.claude/settings.json"
  check "settings: four env keys + allow entry" "$NODE" -e '
    const s = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    const want = ["MAX_MCP_OUTPUT_TOKENS","MCP_TOOL_TIMEOUT","CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS","CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH"];
    process.exit(want.every((k) => s.env[k]) && !s.env.CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT && s.permissions.allow.includes("mcp__plugin_ni_rua_ni_rua") ? 0 : 1)' "$S"
  local doc; doc=$(cc "$NODE" "$CLI" doctor --port "$MPORT" --claude-bin "$bin" --codex-bin /nonexistent --opencode-bin /nonexistent 2>&1)
  has "$doc" "[ok] gateway on 127.0.0.1:$MPORT" && ok "doctor sees the gateway" || bad "doctor sees the gateway"
  has "$doc" "claude: ni_rua@ni_rua listed" && ok "doctor sees the plugin" || bad "doctor sees the plugin"
  if [ "$HAVE" = 1 ]; then
    ( cd "$H/work" && cc perl -e 'alarm 120; exec @ARGV' "$bin" -p "hello" >"$T/$label/claude-turn.out" 2>&1 </dev/null )
    local body; body=$(cat "$LOG" 2>/dev/null)
    has "$body" "ni_rua:ni-rua-robot" && ok "model request lists skill ni_rua:ni-rua-robot" || bad "model request lists skill ni_rua:ni-rua-robot"
    has "$body" "ni_rua:ni-rua-experiments" && ok "model request lists skill ni_rua:ni-rua-experiments" || bad "model request lists skill ni_rua:ni-rua-experiments"
    has "$body" "mcp__plugin_ni_rua_ni_rua__robot_observe" && ok "model request has the MCP tools (mcp__plugin_ni_rua_ni_rua__*)" || bad "model request has the MCP tools"
  else skip "model-level skill/tool visibility (no fake model fixture)"; fi
  out=$(cd "$H/work" && cc "$NODE" "$CLI" uninstall claude --claude-bin "$bin" 2>&1); say "$out" | sed 's/^/    | /'
  check "durable copy removed" test ! -e "$H/.local/share/ni-rua/agent-plugin"
  check "settings carry no NI-RUA env/allow" bash -c "! grep -q 'MAX_MCP_OUTPUT_TOKENS\|mcp__plugin_ni_rua' '$S'"
  mcp=$(cd "$H/work" && cc "$bin" plugin list 2>&1); has "$mcp" "ni_rua@ni_rua" && bad "plugin gone after uninstall" || ok "plugin gone after uninstall"
  finish_uninstall "$label" "$H" "$T/$label.before"
}

cx_section() {
  local label=$1 bin=$2 H=$3 MPORT=$4 FPORT=$5 LOG=$6 HAVE=$7
  cx() { env -i PATH="$(dirname "$NODE"):/usr/bin:/bin" HOME="$H" CODEX_HOME="$H/codex" FAKE_API_KEY=sk-fake TERM=dumb "$@"; }
  mkdir -p "$H/codex"
  printf 'model = "fake-model"\nmodel_provider = "fake"\n\n[model_providers.fake]\nname = "fake"\nbase_url = "http://127.0.0.1:%s/v1"\nwire_api = "responses"\nenv_key = "FAKE_API_KEY"\n' "$FPORT" > "$H/codex/config.toml"
  snapshot "$H" > "$T/$label.before"
  local DUR="$H/.local/share/ni-rua/agent-plugin/$VERSION"
  local out; out=$(cd "$H/work" && cx "$NODE" "$CLI" install codex --codex-bin "$bin" --port "$MPORT" 2>&1); local rc=$?
  say "$out" | sed 's/^/    | /'
  [ $rc -eq 0 ] && ok "install exit 0" || { bad "install exit $rc"; return; }
  local list; list=$(cx "$bin" plugin list --json)
  has "$list" '"pluginId": "ni_rua@ni_rua"' && ok "plugin listed (ni_rua@ni_rua)" || bad "plugin listed"
  local get; get=$(cx "$bin" mcp get ni_rua --json 2>&1)
  has "$get" "\"url\": \"http://127.0.0.1:$MPORT/mcp\"" && has "$get" '"tool_timeout_sec": 3600.0' && has "$get" '"startup_timeout_sec": 30.0' \
    && ok "codex mcp get ni_rua: streamable_http url on --port $MPORT, tool_timeout 3600, startup 30" || { bad "codex mcp get ni_rua"; say "$get" | sed 's/^/        /'; }
  local CACHE="$H/codex/plugins/cache/ni_rua/ni_rua/$VERSION"
  check "skills in the Codex cache" test -f "$CACHE/skills/ni-rua-robot/SKILL.md" -a -f "$CACHE/skills/ni-rua-experiments/SKILL.md"
  check "cached .codex-mcp.json has output_token_limit and the port" bash -c "grep -q output_token_limit '$CACHE/.codex-mcp.json' && grep -q ':$MPORT/mcp' '$CACHE/.codex-mcp.json'"
  local doc; doc=$(cx "$NODE" "$CLI" doctor --port "$MPORT" --claude-bin /nonexistent --codex-bin "$bin" --opencode-bin /nonexistent 2>&1)
  has "$doc" "codex: ni_rua@ni_rua listed" && ok "doctor sees the plugin" || bad "doctor sees the plugin"
  if [ "$HAVE" = 1 ]; then
    ( cd "$H/work" && cx perl -e 'alarm 120; exec @ARGV' "$bin" exec --json --skip-git-repo-check "hello" >"$T/$label/codex-turn.out" 2>&1 </dev/null )
    local body; body=$(cat "$LOG" 2>/dev/null)
    has "$body" "ni_rua:ni-rua-robot" && ok "model request lists skill ni_rua:ni-rua-robot" || bad "model request lists skill ni_rua:ni-rua-robot"
    has "$body" "ni_rua:ni-rua-experiments" && ok "model request lists skill ni_rua:ni-rua-experiments" || bad "model request lists skill ni_rua:ni-rua-experiments"
    has "$body" "mcp__ni_rua" && ok "model request has the mcp__ni_rua tool namespace" || bad "model request has the mcp__ni_rua namespace"
    # a 120,000-character tool result (spike fixture: history_list_actions) must reach the model whole; without the
    # per-tool output_token_limit Codex cuts it to about 12,000 characters
    local script; script=$("$NODE" -e 'console.log("SCRIPT="+Buffer.from(JSON.stringify([[{name:"history_list_actions",namespace:"mcp__ni_rua",input:{}}]])).toString("base64url"))')
    ( cd "$H/work" && cx perl -e 'alarm 120; exec @ARGV' "$bin" exec --json --skip-git-repo-check "long result. $script" >"$T/$label/codex-long.out" 2>&1 </dev/null )
    local longest; longest=$("$NODE" - "$LOG" <<'JS'
const fs = require("fs");
const reqs = fs.readFileSync(process.argv[2], "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((r) => (r.path || "").endsWith("/responses"));
let best = 0;
for (const item of reqs.length ? reqs[reqs.length - 1].body.input : []) {
  if (item.type !== "function_call_output") continue;
  const parts = typeof item.output === "string" ? [item.output] : item.output.filter((c) => c.type === "input_text").map((c) => c.text);
  for (const p of parts) best = Math.max(best, p.length);
}
console.log(best);
JS
)
    [ "${longest:-0}" -ge 120000 ] && ok "120,000-character result delivered whole (longest output item $longest chars)" || bad "120,000-character result delivered whole (longest output item ${longest:-0} chars)"
  else skip "model-level skill/tool visibility (no fake model fixture)"; fi
  out=$(cd "$H/work" && cx "$NODE" "$CLI" uninstall codex --codex-bin "$bin" 2>&1); say "$out" | sed 's/^/    | /'
  check "durable copy removed" test ! -e "$H/.local/share/ni-rua/agent-plugin"
  list=$(cx "$bin" plugin list 2>&1); has "$list" "ni_rua@ni_rua" && bad "plugin gone after uninstall" || ok "plugin gone after uninstall"
  check "config.toml has no ni_rua sections" bash -c "! grep -q ni_rua '$H/codex/config.toml'"
  finish_uninstall "$label" "$H" "$T/$label.before"
}

oc_section() {
  local label=$1 bin=$2 H=$3 MPORT=$4 FPORT=$5 LOG=$6 HAVE=$7
  oc() { env -i PATH=/usr/bin:/bin HOME="$H" XDG_CONFIG_HOME="$H/xdg/config" XDG_DATA_HOME="$H/xdg/data" XDG_CACHE_HOME="$H/xdg/cache" \
      XDG_STATE_HOME="$H/xdg/state" OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_PASSWORD=check TERM=dumb "$@"; }
  local CONF="$H/xdg/config/opencode/opencode.json"; mkdir -p "$(dirname "$CONF")"
  "$NODE" -e '
    const [file, port] = process.argv.slice(1);
    require("fs").writeFileSync(file, JSON.stringify({ model: "fake/fake-model", providers: { fake: { name: "fake", package: "@opencode/ai/providers/anthropic-compatible",
      settings: { baseURL: `http://127.0.0.1:${port}/v1`, apiKey: "sk-fake" }, models: { "fake-model": { name: "fake-model",
      capabilities: { tools: true, input: ["text", "image"], output: ["text"] }, limit: { context: 200000, output: 32000 } } } } } }, null, 2));' "$CONF" "$FPORT"
  snapshot "$H" > "$T/$label.before"
  local DUR="$H/.local/share/ni-rua/agent-plugin/$VERSION"
  local out; out=$(cd "$H/work" && oc "$NODE" "$CLI" install opencode --opencode-bin "$bin" --port "$MPORT" 2>&1); local rc=$?
  say "$out" | sed 's/^/    | /'
  [ $rc -eq 0 ] && ok "install exit 0" || { bad "install exit $rc"; return; }
  check "config has plugins entry, tool_output and media.image" "$NODE" -e '
    const c = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
    process.exit(c.plugins.includes(process.argv[2]) && c.tool_output.max_lines === 100000 && c.tool_output.max_bytes === 16000000 && c.media?.image?.auto_resize === false && c.media?.image?.max_base64_bytes === 1073741824 && c.model === "fake/fake-model" ? 0 : 1)' "$CONF" "$DUR"
  local SPORT; SPORT=$(free_port)
  ( cd "$H/work" && oc "$bin" serve --port "$SPORT" --hostname 127.0.0.1 >"$T/$label/serve.log" 2>&1 & )
  sleep 6
  local auth; auth=$(printf 'opencode:check' | base64)
  api() { curl -s -H "Authorization: Basic $auth" -H 'content-type: application/json' "$@"; }
  api "http://127.0.0.1:$SPORT/api/mcp" >/dev/null      # plugins load per location, lazily at first use
  api -X POST "http://127.0.0.1:$SPORT/api/location/reload" >/dev/null
  for _ in $(seq 1 30); do has "$(api "http://127.0.0.1:$SPORT/api/mcp")" '"name":"ni_rua"' && break; sleep 1; done
  local plug; plug=$(api "http://127.0.0.1:$SPORT/api/plugin"); has "$plug" "\"id\":\"ni_rua\"" && ok "plugin loaded (GET /api/plugin)" || bad "plugin loaded"
  local mcp; mcp=$(api "http://127.0.0.1:$SPORT/api/mcp"); has "$mcp" '"name":"ni_rua","status":{"status":"connected"}' && ok "GET /api/mcp: ni_rua connected on --port $MPORT" || { bad "GET /api/mcp connected"; say "$mcp" | sed 's/^/        /'; }
  local skills; skills=$(api "http://127.0.0.1:$SPORT/api/skill")
  has "$skills" '"id":"ni-rua-robot"' && has "$skills" '"id":"ni-rua-experiments"' && ok "GET /api/skill lists ni-rua-robot and ni-rua-experiments" || bad "GET /api/skill lists both skills"
  if [ "$HAVE" = 1 ]; then
    local sid; sid=$(api -X POST -d '{"title":"check"}' "http://127.0.0.1:$SPORT/api/session" | "$NODE" -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).data.id))')
    api -X POST -d '{"text":"hello"}' "http://127.0.0.1:$SPORT/api/session/$sid/prompt" >/dev/null; sleep 8
    local body; body=$(cat "$LOG" 2>/dev/null)
    has "$body" "<name>ni-rua-robot</name>" && ok "system prompt lists skill ni-rua-robot" || bad "system prompt lists skill ni-rua-robot"
    has "$body" "<name>ni-rua-experiments</name>" && ok "system prompt lists skill ni-rua-experiments" || bad "system prompt lists skill ni-rua-experiments"
    has "$body" "ni_rua_robot_observe" && ok "model request has the MCP tools (ni_rua_*)" || bad "model request has the MCP tools"
  else skip "model-level skill/tool visibility (no fake model fixture)"; fi
  pkill -f "opencode serve --port $SPORT" 2>/dev/null; sleep 1
  local doc; doc=$(oc "$NODE" "$CLI" doctor --port "$MPORT" --claude-bin /nonexistent --codex-bin /nonexistent --opencode-bin "$bin" 2>&1)
  has "$doc" "opencode: config" && ok "doctor sees the config entry" || bad "doctor sees the config entry"
  out=$(cd "$H/work" && oc "$NODE" "$CLI" uninstall opencode 2>&1); say "$out" | sed 's/^/    | /'
  check "durable copy removed" test ! -e "$H/.local/share/ni-rua/agent-plugin"
  check "config restored to the seed (no plugins/tool_output/media)" "$NODE" -e '
    const c = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")); process.exit(c.plugins === undefined && c.tool_output === undefined && c.media === undefined ? 0 : 1)' "$CONF"
  # opencode.json is compared as JSON value above (the installer rewrites it with a trailing newline); here: nothing of
  # ours is left (OpenCode's own server writes databases and logs under XDG_*, which are not compared)
  check "no agent-plugin directory or installed.json left" bash -c "test -z \"\$(find '$H' -path '$H/xdg/data' -prune -o \( -name agent-plugin -o -name installed.json \) -print)\""
}

for item in $WANT; do
  case $item in
    claude-pinned) run_host claude-pinned claude "$CLAUDE_PINNED" ;;
    codex-pinned) run_host codex-pinned codex "$CODEX_PINNED" ;;
    claude-user) run_host claude-user claude "$CLAUDE_USER" ;;
    codex-user) run_host codex-user codex "$CODEX_USER" ;;
    opencode-pinned) run_host opencode-pinned opencode "$OPENCODE_PINNED" ;;
    *) say "unknown target $item" ;;
  esac
done
say; say "== summary: $PASS passed, $FAIL failed, $SKIP skipped (work dir $T)"
[ "$FAIL" -eq 0 ]
