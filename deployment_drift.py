#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare BoTTube's OpenAPI, Flask source, and an optional live deployment.

The Flask inventory is extracted from Python's AST.  Production modules are
never imported, because importing bottube_server.py creates directories and
initializes several database-backed integrations.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"})
SAFE_LIVE_METHODS = frozenset({"GET", "HEAD"})
MISSING_IN_CODE = 1
MISSING_IN_SPEC = 2
LIVE_UNAVAILABLE = 4
STALE_ALLOWANCE = 8
CONFIG_ERROR = 64

_FLASK_PARAMETER = re.compile(r"<(?:[^:<>]+:)?([^<>]+)>")
_OPENAPI_PARAMETER = re.compile(r"{([^{}]+)}")


class ConfigError(ValueError):
    """Raised for invalid sentinel input or unsupported source syntax."""


@dataclass(frozen=True, order=True)
class Operation:
    """A normalized HTTP method and route template."""

    method: str
    path: str

    def __post_init__(self) -> None:
        method = self.method.upper()
        if method not in HTTP_METHODS:
            raise ConfigError(f"unsupported HTTP method: {self.method}")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "path", normalize_path(self.path))

    @classmethod
    def parse(cls, value: str) -> "Operation":
        try:
            method, path = value.strip().split(None, 1)
        except ValueError as exc:
            raise ConfigError(f"operation must be 'METHOD /path': {value!r}") from exc
        return cls(method, path)

    def label(self) -> str:
        return f"{self.method} {self.path}"

    def as_dict(self) -> dict[str, str]:
        return {"method": self.method, "path": self.path}


def normalize_path(path: str) -> str:
    """Normalize Flask converters to OpenAPI parameter syntax."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ConfigError(f"route path must start with '/': {path!r}")
    if "?" in path or "#" in path or "\n" in path or "\r" in path:
        raise ConfigError(f"route path must not contain query, fragment, or newline data: {path!r}")
    return _FLASK_PARAMETER.sub(r"{\1}", path)


def _literal_string(node: ast.AST, context: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise ConfigError(f"{context} must be a string literal")


def _literal_methods(node: ast.AST, context: str) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        raise ConfigError(f"{context} methods must be a literal list, tuple, or set")
    methods = set()
    for item in node.elts:
        methods.add(_literal_string(item, context).upper())
    unsupported = methods - HTTP_METHODS
    if unsupported:
        raise ConfigError(f"{context} uses unsupported methods: {', '.join(sorted(unsupported))}")
    return methods


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _call_owner(call: ast.Call) -> tuple[str, str] | None:
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return None
    return call.func.value.id, call.func.attr


def _blueprint_prefixes(tree: ast.AST, source: Path) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        func_name = value.func.id if isinstance(value.func, ast.Name) else None
        if func_name != "Blueprint":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        prefix_node = _keyword(value, "url_prefix")
        prefix = "" if prefix_node is None else _literal_string(prefix_node, f"{source}: Blueprint url_prefix")
        for target in targets:
            if isinstance(target, ast.Name):
                prefixes[target.id] = prefix
    return prefixes


def _join_route(prefix: str, route: str) -> str:
    if not prefix:
        return route
    if route == "/":
        return prefix.rstrip("/") + "/"
    return prefix.rstrip("/") + "/" + route.lstrip("/")


def extract_flask_operations(source_paths: Sequence[Path], app_names: Iterable[str] = ("app",)) -> set[Operation]:
    """Extract Flask decorator and add_url_rule registrations without imports."""
    operations: set[Operation] = set()
    configured_app_names = set(app_names)

    for source in sorted(source_paths):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ConfigError(f"could not parse application source {source}: {exc}") from exc

        prefixes = _blueprint_prefixes(tree, source)
        route_owners = configured_app_names | set(prefixes)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    owner = _call_owner(decorator)
                    if owner is None or owner[0] not in route_owners:
                        continue
                    object_name, decorator_name = owner
                    if decorator_name not in {"route", "get", "post", "put", "patch", "delete"}:
                        continue
                    if not decorator.args:
                        raise ConfigError(f"{source}:{decorator.lineno}: route has no path")
                    route = _literal_string(decorator.args[0], f"{source}:{decorator.lineno}: route path")
                    if decorator_name == "route":
                        methods_node = _keyword(decorator, "methods")
                        methods = {"GET"} if methods_node is None else _literal_methods(
                            methods_node, f"{source}:{decorator.lineno}: route"
                        )
                    else:
                        methods = {decorator_name.upper()}
                    full_path = _join_route(prefixes.get(object_name, ""), route)
                    operations.update(Operation(method, full_path) for method in methods)

            if not isinstance(node, ast.Call):
                continue
            owner = _call_owner(node)
            if owner is None or owner[0] not in route_owners or owner[1] != "add_url_rule":
                continue
            if not node.args:
                raise ConfigError(f"{source}:{node.lineno}: add_url_rule has no path")
            route = _literal_string(node.args[0], f"{source}:{node.lineno}: add_url_rule path")
            methods_node = _keyword(node, "methods")
            methods = {"GET"} if methods_node is None else _literal_methods(
                methods_node, f"{source}:{node.lineno}: add_url_rule"
            )
            full_path = _join_route(prefixes.get(owner[0], ""), route)
            operations.update(Operation(method, full_path) for method in methods)

    return operations


def load_openapi_operations(path: Path) -> set[Operation]:
    """Load operations from an OpenAPI 3 document."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by the CI install contract
        raise ConfigError("PyYAML is required to read openapi.yaml (pip install PyYAML)") from exc

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not read OpenAPI document {path}: {exc}") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("paths"), Mapping):
        raise ConfigError(f"OpenAPI document {path} must contain a paths mapping")

    operations: set[Operation] = set()
    for route, path_item in document["paths"].items():
        if not isinstance(route, str) or not isinstance(path_item, Mapping):
            raise ConfigError(f"invalid OpenAPI path item: {route!r}")
        for method in path_item:
            if isinstance(method, str) and method.upper() in HTTP_METHODS:
                operations.add(Operation(method, route))
    return operations


def _resolve_sources(repo_root: Path, patterns: Sequence[str]) -> list[Path]:
    if not patterns:
        raise ConfigError("application_sources must contain at least one source glob")
    matches: set[Path] = set()
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ConfigError("application_sources entries must be non-empty strings")
        current = {path.resolve() for path in repo_root.glob(pattern) if path.is_file()}
        if not current:
            raise ConfigError(f"application source pattern matched no files: {pattern}")
        for path in current:
            try:
                path.relative_to(repo_root)
            except ValueError as exc:
                raise ConfigError(f"application source is outside repo root: {path}") from exc
        matches.update(current)
    return sorted(matches)


def _operation_list(values: Any, context: str) -> set[Operation]:
    if values is None:
        return set()
    if not isinstance(values, list):
        raise ConfigError(f"{context} must be a list")
    operations = set()
    for value in values:
        if isinstance(value, str):
            operations.add(Operation.parse(value))
        elif isinstance(value, Mapping):
            try:
                operations.add(Operation(str(value["method"]), str(value["path"])))
            except KeyError as exc:
                raise ConfigError(f"{context} entries require method and path") from exc
        else:
            raise ConfigError(f"{context} entries must be strings or objects")
    return operations


def _path_matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _operation_dicts(operations: Iterable[Operation]) -> list[dict[str, str]]:
    return [operation.as_dict() for operation in sorted(operations)]


def _parse_fixtures(values: Sequence[str]) -> dict[str, str]:
    fixtures: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ConfigError(f"fixture must be NAME=VALUE: {value!r}")
        name, fixture = value.split("=", 1)
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigError(f"invalid fixture name: {name!r}")
        if not fixture:
            raise ConfigError(f"fixture value must not be empty: {name}")
        fixtures[name] = fixture
    return fixtures


def _timeout_seconds(value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigError("timeout must be a positive number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("timeout must be a positive number") from exc
    if timeout <= 0:
        raise ConfigError("timeout must be a positive number")
    return timeout


def render_path(path: str, fixtures: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Render an OpenAPI route template with URL-encoded fixture values."""
    missing = sorted({name for name in _OPENAPI_PARAMETER.findall(path) if name not in fixtures})
    if missing:
        return None, missing

    def replace(match: re.Match[str]) -> str:
        return urllib.parse.quote(str(fixtures[match.group(1)]), safe="")

    return _OPENAPI_PARAMETER.sub(replace, path), []


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, ANN001
        return None


def request_head(url: str, timeout: float) -> int:
    """Make one credential-free HEAD request without following redirects."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "*/*", "User-Agent": "bottube-deployment-drift/1"},
        method="HEAD",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def validate_live_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("live base URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError("live base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigError("live base URL must not contain a query string or fragment")
    return base_url.rstrip("/")


def probe_live(
    operations: Iterable[Operation],
    base_url: str,
    fixtures: Mapping[str, Any],
    timeout: float,
    requester: Callable[[str, float], int] = request_head,
) -> list[dict[str, Any]]:
    """Probe only safe route operations, using HEAD on the wire."""
    base_url = validate_live_base_url(base_url)
    results: list[dict[str, Any]] = []
    for operation in sorted(set(operations)):
        if operation.method not in SAFE_LIVE_METHODS:
            raise ConfigError(f"live probes may only target GET or HEAD: {operation.label()}")
        request_path, missing = render_path(operation.path, fixtures)
        result: dict[str, Any] = {
            "available": False,
            "method": operation.method,
            "path": operation.path,
            "reason": "",
            "request_method": "HEAD",
            "request_path": request_path,
            "status": None,
        }
        if missing:
            result["reason"] = "missing_fixture:" + ",".join(missing)
            results.append(result)
            continue

        try:
            status = int(requester(base_url + str(request_path), timeout))
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            result["reason"] = "network_error:" + type(exc).__name__
            results.append(result)
            continue
        result["status"] = status
        if status in {404, 405} or status >= 500:
            result["reason"] = f"http_status:{status}"
        else:
            result["available"] = True
            result["reason"] = "route_present"
        results.append(result)
    return results


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read sentinel config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("sentinel config root must be a JSON object")
    return value


def build_report(
    repo_root: Path,
    config: Mapping[str, Any],
    *,
    live_enabled: bool = False,
    live_base_url: str | None = None,
    fixture_overrides: Mapping[str, str] | None = None,
    requester: Callable[[str, float], int] = request_head,
) -> dict[str, Any]:
    """Build a deterministic drift report from validated inputs."""
    openapi_name = config.get("openapi", "openapi.yaml")
    if not isinstance(openapi_name, str):
        raise ConfigError("openapi must be a path string")
    openapi_path = (repo_root / openapi_name).resolve()
    try:
        openapi_path.relative_to(repo_root)
    except ValueError as exc:
        raise ConfigError("openapi path must stay inside the repo root") from exc

    source_patterns = config.get("application_sources", ["bottube_server.py"])
    if not isinstance(source_patterns, list):
        raise ConfigError("application_sources must be a list")
    source_paths = _resolve_sources(repo_root, source_patterns)
    app_names = config.get("application_names", ["app"])
    if not isinstance(app_names, list) or not all(isinstance(name, str) for name in app_names):
        raise ConfigError("application_names must be a list of strings")

    spec_operations = load_openapi_operations(openapi_path)
    code_operations = extract_flask_operations(source_paths, app_names)
    canaries = _operation_list(config.get("canaries", []), "canaries")
    unsafe_canaries = canaries - {op for op in canaries if op.method in SAFE_LIVE_METHODS}
    if unsafe_canaries:
        labels = ", ".join(operation.label() for operation in sorted(unsafe_canaries))
        raise ConfigError(f"canaries must be safe GET or HEAD routes: {labels}")

    expected_in_code = spec_operations | canaries
    missing_in_code = expected_in_code - code_operations

    patterns = config.get("missing_in_spec_patterns", ["*"])
    if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) for item in patterns):
        raise ConfigError("missing_in_spec_patterns must be a non-empty list of glob strings")
    scoped_code = {operation for operation in code_operations if _path_matches(operation.path, patterns)}
    missing_in_spec = scoped_code - spec_operations - canaries

    allowed_config = config.get("allowed_drift", {})
    if not isinstance(allowed_config, Mapping):
        raise ConfigError("allowed_drift must be an object")
    allowed_missing_in_code = _operation_list(
        allowed_config.get("missing_in_code", []), "allowed_drift.missing_in_code"
    )
    allowed_missing_in_spec = _operation_list(
        allowed_config.get("missing_in_spec", []), "allowed_drift.missing_in_spec"
    )
    blocking_missing_in_code = missing_in_code - allowed_missing_in_code
    blocking_missing_in_spec = missing_in_spec - allowed_missing_in_spec
    stale_allowances = (allowed_missing_in_code - missing_in_code) | (allowed_missing_in_spec - missing_in_spec)

    config_fixtures = config.get("fixtures", {})
    if not isinstance(config_fixtures, Mapping):
        raise ConfigError("fixtures must be an object")
    fixtures = {str(key): str(value) for key, value in config_fixtures.items()}
    fixtures.update(fixture_overrides or {})
    timeout = _timeout_seconds(config.get("timeout", 5))

    if live_enabled and not live_base_url:
        raise ConfigError("--live requires --live-base-url")
    if not live_enabled and live_base_url:
        raise ConfigError("--live-base-url is inert without explicit --live")

    live_results: list[dict[str, Any]] = []
    normalized_live_base: str | None = None
    if live_enabled:
        normalized_live_base = validate_live_base_url(str(live_base_url))
        probe_openapi = config.get("live_probe_openapi_reads", True)
        if not isinstance(probe_openapi, bool):
            raise ConfigError("live_probe_openapi_reads must be true or false")
        live_operations = set(canaries)
        if probe_openapi:
            live_operations.update(op for op in spec_operations if op.method in SAFE_LIVE_METHODS)
        live_results = probe_live(live_operations, normalized_live_base, fixtures, timeout, requester)

    live_unavailable = [result for result in live_results if not result["available"]]
    exit_code = 0
    if blocking_missing_in_code:
        exit_code |= MISSING_IN_CODE
    if blocking_missing_in_spec:
        exit_code |= MISSING_IN_SPEC
    if live_unavailable:
        exit_code |= LIVE_UNAVAILABLE
    if stale_allowances:
        exit_code |= STALE_ALLOWANCE

    relative_sources = [str(path.relative_to(repo_root)) for path in source_paths]
    report = {
        "allowed": {
            "missing_in_code": _operation_dicts(missing_in_code & allowed_missing_in_code),
            "missing_in_spec": _operation_dicts(missing_in_spec & allowed_missing_in_spec),
        },
        "blocking": {
            "live_unavailable": live_unavailable,
            "missing_in_code": _operation_dicts(blocking_missing_in_code),
            "missing_in_spec": _operation_dicts(blocking_missing_in_spec),
        },
        "drift": {
            "live_unavailable": live_unavailable,
            "missing_in_code": _operation_dicts(missing_in_code),
            "missing_in_spec": _operation_dicts(missing_in_spec),
        },
        "exit_code": exit_code,
        "inventory": {
            "application_operations": len(code_operations),
            "application_sources": relative_sources,
            "canary_operations": len(canaries),
            "openapi": str(openapi_path.relative_to(repo_root)),
            "openapi_operations": len(spec_operations),
        },
        "live": {
            "base_url": normalized_live_base,
            "enabled": live_enabled,
            "results": live_results,
        },
        "stale_allowances": _operation_dicts(stale_allowances),
        "status": "pass" if exit_code == 0 else "fail",
    }
    return report


def format_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _operation_from_dict(value: Mapping[str, Any]) -> Operation:
    return Operation(str(value["method"]), str(value["path"]))


def format_text(report: Mapping[str, Any]) -> str:
    """Render a stable, human-readable report."""
    inventory = report["inventory"]
    live = report["live"]
    lines = [
        "BoTTube deployment drift sentinel",
        f"OpenAPI: {inventory['openapi']} ({inventory['openapi_operations']} operations)",
        f"Application: {inventory['application_operations']} operations from "
        + ", ".join(inventory["application_sources"]),
        f"Canaries: {inventory['canary_operations']} operations",
        "Live: " + (f"enabled ({live['base_url']}, HEAD only)" if live["enabled"] else "disabled"),
    ]

    allowed_code = {_operation_from_dict(item) for item in report["allowed"]["missing_in_code"]}
    allowed_spec = {_operation_from_dict(item) for item in report["allowed"]["missing_in_spec"]}
    for key, title, allowed in (
        ("missing_in_code", "Missing in code", allowed_code),
        ("missing_in_spec", "Missing in spec", allowed_spec),
    ):
        values = [_operation_from_dict(item) for item in report["drift"][key]]
        blocking = len(report["blocking"][key])
        lines.append(f"{title}: {len(values)} ({blocking} blocking)")
        for operation in values:
            marker = "known" if operation in allowed else "blocking"
            lines.append(f"  [{marker}] {operation.label()}")

    unavailable = report["drift"]["live_unavailable"]
    lines.append(f"Live unavailable: {len(unavailable)} ({len(unavailable)} blocking)")
    for result in unavailable:
        destination = result["request_path"] or result["path"]
        lines.append(
            f"  [blocking] {result['method']} {result['path']} -> HEAD {destination}: {result['reason']}"
        )

    stale = [_operation_from_dict(item) for item in report["stale_allowances"]]
    lines.append(f"Stale allowances: {len(stale)}")
    for operation in stale:
        lines.append(f"  [blocking] {operation.label()}")
    lines.append(f"Status: {str(report['status']).upper()}")
    lines.append(f"Exit code: {report['exit_code']}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="repository root (default: current directory)")
    parser.add_argument("--config", default="deployment-drift.json", help="JSON policy path relative to repo root")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="report output format")
    parser.add_argument("--fixture", action="append", default=[], metavar="NAME=VALUE", help="override a path fixture")
    parser.add_argument("--live", action="store_true", help="explicitly enable credential-free HEAD probes")
    parser.add_argument("--live-base-url", help="base URL to probe; requires --live")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve()
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config = load_config(config_path)
        report = build_report(
            repo_root,
            config,
            live_enabled=args.live,
            live_base_url=args.live_base_url,
            fixture_overrides=_parse_fixtures(args.fixture),
        )
    except ConfigError as exc:
        print(f"deployment-drift: configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    output = format_json(report) if args.format == "json" else format_text(report)
    sys.stdout.write(output)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
