"""RunManager — launches and tracks analysis pipeline runs.

Runs spawn the ``fmriflow run <config.yaml>`` CLI as a detached subprocess
(``start_new_session=True``) with stdout+stderr captured to a log file
under ``~/.fmriflow/runs/{run_id}/stdout.log`` and a sidecar
``state.json``. They survive server restarts: on startup the manager
scans the registry and reattaches any live pipeline PIDs.

Tradeoff: structured stage events that used to flow via UICaptureProxy
are not emitted to the WebSocket during detached runs (the subprocess
can't reach the parent's in-memory queue). The live log stream from the
tailer is still there, and on completion the stage timeline is loaded
back from ``{output_dir}/run_summary.json``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess as _subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fmriflow.core import paths
from fmriflow.server.services.run_registry import RunRegistry, RunStateFile

logger = logging.getLogger(__name__)


def _display_name(config: dict) -> str:
    """Best human-readable name for an in-flight row.

    Group YAMLs lack a top-level ``experiment:`` — the relevant name is
    ``group:``. Study YAMLs use ``study:``. Subject YAMLs use the
    normal ``experiment:`` field. Falls back to '' when none is set.
    """
    if not isinstance(config, dict):
        return ''
    if isinstance(config.get('study'), str) and config['study']:
        return config['study']
    if isinstance(config.get('group'), str) and config['group']:
        return config['group']
    return str(config.get('experiment') or '')


def _experiment_from_state(state: RunStateFile) -> str:
    """Display name for a registry-only (finished, no in-memory handle) run.

    Prefer the value persisted in ``state.params['experiment']`` — for
    runs registered after the ``_display_name`` fix it's already the
    group/study/experiment name. For older entries persisted before the
    fix (where group/study runs got ``None``), lazily load the YAML the
    state points at and recompute. Returns ``''`` if nothing usable
    survives.
    """
    params = state.params or {}
    persisted = params.get('experiment')
    if isinstance(persisted, str) and persisted:
        return persisted
    cfg = _load_state_config(state)
    if cfg:
        return _display_name(cfg)
    return ''


def _load_state_config(state: RunStateFile) -> dict | None:
    """Lazy YAML load for the config a registry entry points at."""
    params = state.params or {}
    cfg_path = params.get('config_path') or state.config_path
    if not cfg_path:
        return None
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    return cfg if isinstance(cfg, dict) else None


def discover_subject_run_summaries(
    registry: "RunRegistry",
) -> list[Path]:
    """Locate every subject-scope ``run_summary.json`` we can see.

    Two sources are merged and deduped by the run-dir's realpath:

    1. **Default root** — ``paths.results_root().rglob('run_summary.json')``.
       The legacy behavior; finds anything under ``$FMRIFLOW_HOME/data/results/``.
    2. **Run registry** — every ``kind='run'`` entry that is *not* a
       group/study run. Its ``output_dir`` is the run dir itself
       (subject runs land flat: ``<base>/run_<stamp>_<id>/run_summary.json``),
       which catches runs whose ``reporting.output_dir`` points outside
       ``$FMRIFLOW_HOME/data/results/``.

    Returns the ``run_summary.json`` paths so callers can hydrate
    :class:`RunSummary` without re-discovering them.
    """
    seen: set[str] = set()
    out: list[Path] = []

    def add(summary_path: Path) -> None:
        try:
            real = str(summary_path.parent.resolve(strict=False))
        except Exception:
            real = str(summary_path.parent)
        if real in seen:
            return
        seen.add(real)
        out.append(summary_path)

    # Source 1 — filesystem default tree(s): primary + read-only extras.
    for root in paths.result_roots():
        try:
            for summary_path in root.rglob('run_summary.json'):
                add(summary_path)
        except OSError:
            # An extra root on an offline mount shouldn't break discovery.
            logger.warning("Skipping unreadable results root: %s", root)

    # Source 2 — registry entries whose output_dir we know.
    for state in registry.list_all():
        if state.kind != 'run':
            continue
        is_group, is_study = kind_from_state(state)
        if is_group or is_study:
            continue
        params = state.params or {}
        output_dir = params.get('output_dir')
        if not output_dir:
            continue
        # Defensive: registry entries written before the tilde-expanduser
        # fix may still hold an unexpanded ``~/...`` path.
        run_dir = Path(os.path.expanduser(os.path.expandvars(str(output_dir))))
        summary_path = run_dir / 'run_summary.json'
        if summary_path.is_file():
            add(summary_path)

    return out


def discover_group_run_dirs(
    registry: "RunRegistry", *, name: str | None = None,
) -> list[tuple[str, str, Path]]:
    """Locate every ``(group_name, run_id, run_dir)`` we can see.

    Two sources are merged and deduped by the run-dir's realpath:

    1. **Default root** — ``paths.group_runs_root()/<group_name>/<run_id>/``.
       Includes legacy and symlinked layouts. This is what
       ``/group-runs`` saw before the registry plumbing landed.
    2. **Run registry** — every ``kind='run'`` entry with
       ``is_group=True``. Its ``output_dir`` is the *parent* (the
       group's directory), so we scan it for any
       ``<subdir>/group_summary.json``. This catches runs whose
       ``output_dir`` lives outside ``$FMRIFLOW_HOME``.

    Optional ``name`` filters both sources to one group.
    """
    return _discover_run_dirs(
        registry,
        default_roots=paths.group_run_roots(),
        summary_name='group_summary.json',
        kind='group',
        name=name,
    )


def discover_study_run_dirs(
    registry: "RunRegistry", *, name: str | None = None,
) -> list[tuple[str, str, Path]]:
    """Study analogue of :func:`discover_group_run_dirs`."""
    return _discover_run_dirs(
        registry,
        default_roots=paths.study_run_roots(),
        summary_name='study_summary.json',
        kind='study',
        name=name,
    )


def _discover_run_dirs(
    registry: "RunRegistry",
    *,
    default_roots: list[Path],
    summary_name: str,
    kind: str,
    name: str | None,
) -> list[tuple[str, str, Path]]:
    seen: set[str] = set()
    out: list[tuple[str, str, Path]] = []

    def add(group_or_study: str, run_id: str, run_dir: Path) -> None:
        try:
            real = str(run_dir.resolve(strict=False))
        except Exception:
            real = str(run_dir)
        if real in seen:
            return
        seen.add(real)
        out.append((group_or_study, run_id, run_dir))

    # Source 1 — default root(s): primary + read-only extras. Earlier
    # roots win on collisions (dedupe-by-realpath), so the primary's copy
    # of a same-named run takes precedence.
    for default_root in default_roots:
        try:
            if not default_root.is_dir():
                continue
            tops = sorted(default_root.iterdir())
        except OSError:
            logger.warning("Skipping unreadable run root: %s", default_root)
            continue
        for top in tops:
            if not top.is_dir():
                continue
            if name is not None and top.name != name:
                continue
            # Pre-run-id layout: <name>/group_summary.json directly.
            if (top / summary_name).is_file():
                add(top.name, '', top)
            for child in sorted(top.iterdir()):
                if not child.is_dir() or child.name == 'latest':
                    continue
                if (child / summary_name).is_file():
                    add(top.name, child.name, child)

    # Source 2 — registry entries whose output_dir we know.
    for state in registry.list_all():
        if state.kind != 'run':
            continue
        is_group, is_study = kind_from_state(state)
        if kind == 'group' and not is_group:
            continue
        if kind == 'study' and not is_study:
            continue
        params = state.params or {}
        output_dir = params.get('output_dir')
        if not output_dir:
            continue
        parent = Path(output_dir)
        if not parent.is_dir():
            continue
        # output_dir is the *parent* (group/study directory); the
        # orchestrator creates a timestamped subdir under it.
        group_name = _experiment_from_state(state) or parent.name
        if name is not None and group_name != name:
            continue
        # Pre-run-id layout.
        if (parent / summary_name).is_file():
            add(group_name, '', parent)
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name == 'latest':
                continue
            if (child / summary_name).is_file():
                add(group_name, child.name, child)

    return out


def resolve_group_run_dir(
    registry: "RunRegistry", name: str, run_id: str,
    root_id: str | None = None,
) -> Path | None:
    """Return the on-disk dir for a ``(name, run_id)`` pair or ``None``.

    Discovery is in precedence order (primary root first), so without a
    ``root_id`` the primary's copy wins on a cross-root collision. Pass
    ``root_id`` to disambiguate to a specific root.
    """
    for gname, rid, run_dir in discover_group_run_dirs(registry, name=name):
        if gname == name and rid == run_id:
            if root_id is not None and paths.root_id_for_path(run_dir) != root_id:
                continue
            return run_dir
    return None


def resolve_study_run_dir(
    registry: "RunRegistry", name: str, run_id: str,
    root_id: str | None = None,
) -> Path | None:
    """Study analogue of :func:`resolve_group_run_dir`."""
    for sname, rid, run_dir in discover_study_run_dirs(registry, name=name):
        if sname == name and rid == run_id:
            if root_id is not None and paths.root_id_for_path(run_dir) != root_id:
                continue
            return run_dir
    return None


def kind_from_state(state: RunStateFile) -> tuple[bool, bool]:
    """``(is_group, is_study)`` for a registry entry.

    Prefer the explicit flags persisted in ``params`` (new runs). Fall
    back to inspecting the YAML the state points at (old runs that
    pre-date the flag-persistence fix). Defaults to ``(False, False)``.
    """
    params = state.params or {}
    if 'is_group' in params or 'is_study' in params:
        return bool(params.get('is_group')), bool(params.get('is_study'))
    cfg = _load_state_config(state)
    if not cfg:
        return False, False
    is_study = (
        isinstance(cfg.get('study'), str)
        and isinstance(cfg.get('groups'), list)
    )
    is_group = (not is_study) and (
        isinstance(cfg.get('group'), str)
        and isinstance(cfg.get('subjects'), list)
    )
    return is_group, is_study


def _apply_per_run_output_dir(config: dict, run_id: str) -> dict:
    """Rewrite ``config['reporting']['output_dir']`` to a per-run
    subdirectory so repeat runs of the same experiment don't clobber
    each other's ``run_summary.json`` / artifacts.

    The suffix is ``run_<YYYYmmdd-HHMMSS>_<run_id>`` — sortable and
    human-scannable. Returns a shallow copy of config (original is
    left intact for callers that hold a reference).

    ``~`` and ``$VAR`` in the configured base path are expanded — a
    literal tilde would otherwise become a directory named ``~``
    under CWD (Python's ``Path()`` doesn't expanduser by default).
    """
    from datetime import datetime

    out = dict(config)
    reporting = dict(out.get('reporting') or {})
    base = reporting.get('output_dir') or './results'
    base = os.path.expanduser(os.path.expandvars(str(base)))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    reporting['output_dir'] = str(Path(base) / f"run_{stamp}_{run_id}")
    out['reporting'] = reporting
    return out


@dataclass
class RunHandle:
    """A single analysis pipeline run.

    Detach-reattach bookkeeping lives alongside the legacy fields; the
    WebSocket endpoint treats ``log`` events the same way the preproc /
    convert / autoflatten streams do.
    """

    run_id: str
    config: dict
    config_path: str | None = None
    status: str = 'pending'              # pending | running | done | failed | cancelled | lost
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    _pending: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Detach-reattach bookkeeping
    pid: int | None = None
    pgid: int | None = None
    log_path: str | None = None
    events_path: str | None = None
    output_dir: str | None = None
    is_reattached: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0

    # Path to a temp YAML we wrote for subprocess consumption (cleanup after run).
    _temp_config_path: str | None = None
    # True when this is a group-scope run (spawned via `fmriflow run-group`).
    # Drives both subprocess dispatch and how the finalize step looks
    # for the per-invocation group_summary.json on disk.
    is_group: bool = False
    # True when this is a study-scope run (spawned via `fmriflow run-study`).
    # Mutually exclusive with is_group at construction time.
    is_study: bool = False

    def push_event(self, event: dict) -> None:
        event.setdefault('timestamp', time.time())
        with self._lock:
            self.events.append(event)
            self._pending.append(event)

    def drain_events(self) -> list[dict]:
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    def to_summary(self) -> dict:
        return {
            'run_id': self.run_id,
            'status': self.status,
            'pid': self.pid,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'is_reattached': self.is_reattached,
            'error': self.error,
            'config_path': self.config_path,
            'output_dir': self.output_dir,
            'log_path': self.log_path,
            'events_path': self.events_path,
        }


class RunManager:
    """Manages background pipeline runs as detached subprocesses."""

    def __init__(self, registry: RunRegistry | None = None):
        self.active_runs: dict[str, RunHandle] = {}
        self.registry = registry or RunRegistry()
        try:
            self._reattach_active_runs()
        except Exception:
            logger.warning("Failed to scan run registry on startup", exc_info=True)

    # ── Launch ──────────────────────────────────────────────────────

    def start_run(self, config: dict) -> str:
        """Launch a pipeline run from a config dict.

        The dict is written to a temp YAML so the CLI subprocess can
        consume it; the temp file is cleaned up when the run finishes
        (or the server dies — temp files are tracked in ``TMPDIR``).
        """
        if not isinstance(config, dict) or not config:
            raise ValueError("start_run requires a non-empty config dict")

        run_id = uuid.uuid4().hex[:12]
        config = _apply_per_run_output_dir(config, run_id)

        # Write dict → temp YAML that the CLI can load.
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix=f"_{run_id}.yaml", delete=False,
        )
        yaml.safe_dump(config, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()

        handle = self._register_handle(
            run_id=run_id,
            config=config,
            config_path=tmp.name,
            temp_config_path=tmp.name,
        )
        self._spawn_and_track(handle)
        logger.info("Started run %s (temp config %s)", run_id, tmp.name)
        return run_id

    def start_run_from_config(
        self,
        config_path: str,
        overrides: dict | None = None,
    ) -> str:
        """Launch a pipeline run from a YAML config file.

        Detects the config kind from top-level fields:

          * ``study:`` + ``groups:``       → spawn ``fmriflow run-study``
          * ``group:`` + ``subjects:``     → spawn ``fmriflow run-group``
          * otherwise                       → spawn ``fmriflow run``

        Subject runs get a per-run ``reporting.output_dir`` suffix so
        repeat runs don't clobber each other. Group and study runs
        already isolate per-run via the orchestrator's own
        ``<name>/<run_id>/`` layout, so no rewrite is needed.
        """
        run_id = uuid.uuid4().hex[:12]

        with open(config_path) as f:
            base = yaml.safe_load(f) or {}
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    base[k] = v

        is_study = (
            isinstance(base.get('study'), str)
            and isinstance(base.get('groups'), list)
        )
        is_group = (not is_study) and (
            isinstance(base.get('group'), str)
            and isinstance(base.get('subjects'), list)
        )

        if is_study or is_group:
            config = base
        else:
            config = _apply_per_run_output_dir(base, run_id)

        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix=f"_{run_id}.yaml", delete=False,
        )
        yaml.safe_dump(config, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()
        effective_path = tmp.name
        temp_path = tmp.name

        handle = self._register_handle(
            run_id=run_id,
            config=config,
            config_path=effective_path,
            temp_config_path=temp_path,
            is_group=is_group,
            is_study=is_study,
        )
        self._spawn_and_track(handle)
        logger.info(
            "Started %s run %s from config %s",
            "study" if is_study else ("group" if is_group else "subject"),
            run_id, config_path,
        )
        return run_id

    # ── Registry + spawn helpers ────────────────────────────────────

    def _register_handle(
        self,
        *,
        run_id: str,
        config: dict,
        config_path: str,
        temp_config_path: str | None,
        is_group: bool = False,
        is_study: bool = False,
    ) -> RunHandle:
        now = time.time()
        if is_study:
            # StudyOrchestrator lands runs under
            # study_runs/<study_name>/<run_id>/; same pattern as group
            # — surface the parent and resolve the real run dir later.
            from fmriflow.core import paths
            study_name = config.get('study') or ''
            output_dir = (
                config.get('output_dir')
                or str(paths.study_runs_root() / study_name)
            )
        elif is_group:
            # GroupOrchestrator lands runs under
            # group_runs/<group_name>/<run_id>/; we won't know the exact
            # timestamp until the child process creates it, so just
            # surface the parent here. _finalize_from_output will
            # resolve the real run dir from the latest summary written.
            from fmriflow.core import paths
            group_name = config.get('group') or ''
            output_dir = (
                config.get('output_dir')
                or str(paths.group_runs_root() / group_name)
            )
        else:
            output_dir = (
                (config.get('reporting') or {}).get('output_dir') or './results'
            )

        handle = RunHandle(
            run_id=run_id,
            config=config,
            config_path=config_path,
            status='running',
            started_at=now,
            output_dir=output_dir,
            _temp_config_path=temp_config_path,
            is_group=is_group,
            is_study=is_study,
        )

        state = RunStateFile(
            run_id=run_id,
            kind='run',
            backend='pipeline',
            subject=str(config.get('subject', '')),
            status='running',
            started_at=now,
            config_path=config_path,
            params={
                'config_path': config_path,
                'output_dir': output_dir,
                # _display_name covers group/study YAMLs that don't carry
                # a top-level `experiment:` — falls back to `group:` /
                # `study:` so the registry-persisted name is always the
                # one the dashboard wants to show.
                'experiment': _display_name(config),
                # Scope flags so registry-only consumers (group/study run
                # listings) can identify the kind without re-parsing the
                # YAML on every call.
                'is_group': bool(is_group),
                'is_study': bool(is_study),
            },
        )
        self.registry.register(state)
        handle.log_path = state.stdout_log
        self.active_runs[run_id] = handle
        return handle

    def _spawn_and_track(self, handle: RunHandle) -> None:
        thread = threading.Thread(
            target=self._execute,
            args=(handle,),
            daemon=True,
            name=f"run-{handle.run_id}",
        )
        thread.start()

    def _execute(self, handle: RunHandle) -> None:
        """Spawn the CLI as a detached child and drive its log."""
        log_path = Path(handle.log_path) if handle.log_path else None
        events_path = (log_path.parent / "events.jsonl") if log_path else None

        try:
            sub = (
                "run-study" if handle.is_study
                else "run-group" if handle.is_group
                else "run"
            )
            cmd = [
                sys.executable, "-u", "-m", "fmriflow.cli",
                sub, handle.config_path,
            ]
            logger.info("Running pipeline: %s", " ".join(cmd))

            if handle.is_study:
                name_for_log = handle.config.get('study', '?')
            elif handle.is_group:
                name_for_log = handle.config.get('group', '?')
            else:
                name_for_log = handle.config.get('experiment', '?')
            handle.push_event({
                'event': 'started',
                'message': f"Starting {sub} for {name_for_log}",
                'is_group': handle.is_group,
                'is_study': handle.is_study,
            })

            # Pass an events file so the subprocess can emit per-stage
            # transitions (stimuli/responses/features/prepare/model/
            # analyze/report) for the workflow graph and UI.
            child_env = os.environ.copy()
            if events_path is not None:
                events_path.touch()
                child_env["FMRIFLOW_EVENTS_FILE"] = str(events_path)
                handle.events_path = str(events_path)
                self._persist_state(handle)

            log_fh = None
            tailer = None
            events_tailer = None
            try:
                if log_path is not None:
                    log_fh = open(log_path, "w", buffering=1)
                    proc = _subprocess.Popen(
                        cmd,
                        stdout=log_fh,
                        stderr=_subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                        env=child_env,
                    )
                else:
                    proc = _subprocess.Popen(
                        cmd,
                        stdout=_subprocess.DEVNULL,
                        stderr=_subprocess.STDOUT,
                        text=True,
                        env=child_env,
                    )

                handle.pid = proc.pid
                try:
                    handle.pgid = os.getpgid(proc.pid)
                except OSError:
                    handle.pgid = proc.pid
                self._persist_state(handle)

                proc_done = lambda: proc.poll() is not None
                if log_path is not None:
                    tailer = _RunLogTailer(
                        log_path, handle, stop_when=proc_done,
                    )
                    tailer.start()
                if events_path is not None:
                    events_tailer = _RunEventsTailer(
                        events_path, handle, stop_when=proc_done,
                    )
                    events_tailer.start()

                proc.wait()
                # Drain both tailers before finalizing: the final status ends the
                # dashboard stream, and the last log lines hold the traceback.
                if tailer is not None:
                    tailer.stop_and_join(timeout=10.0)
                if events_tailer is not None:
                    events_tailer.stop_and_join(timeout=10.0)
                self._finalize_from_output(handle, proc.returncode)
            finally:
                if tailer is not None:
                    tailer.stop_and_join()
                if events_tailer is not None:
                    events_tailer.stop_and_join()
                if log_fh is not None:
                    log_fh.close()

        except Exception as e:
            import traceback as _tb
            tb_text = _tb.format_exc()
            handle.status = 'failed'
            handle.error = f"{type(e).__name__}: {e}"
            handle.finished_at = time.time()
            handle.push_event({
                'event': 'run_failed',
                'error': handle.error,
                'traceback': tb_text,
                'elapsed': handle.finished_at - handle.started_at,
            })
            if handle.log_path:
                try:
                    with open(handle.log_path, 'a') as _lf:
                        _lf.write('\n\n=== wrapper traceback ===\n')
                        _lf.write(tb_text)
                except Exception:
                    pass
            logger.error("Run %s failed: %s", handle.run_id, e, exc_info=True)

        finally:
            self._persist_state(handle)
            # Clean up the temp YAML if we wrote one.
            if handle._temp_config_path:
                try:
                    os.unlink(handle._temp_config_path)
                except Exception:
                    pass

    def _finalize_from_output(self, handle: RunHandle, returncode: int) -> None:
        """Inspect the run's summary file to determine final status.

        Subject runs write ``<output_dir>/run_summary.json``; group / study
        runs write ``<output_dir>/<run_id>/<scope>_summary.json``. The
        actual path resolution and status-derivation lives in
        :func:`_apply_summary_to_handle` so the reattached monitor uses
        the same logic.
        """
        _apply_summary_to_handle(handle, returncode=returncode)

    # ── Registry / reattach / cancel ────────────────────────────────

    def _persist_state(self, handle: RunHandle) -> None:
        state = RunStateFile(
            run_id=handle.run_id,
            kind='run',
            backend='pipeline',
            subject=str(handle.config.get('subject', '')),
            status=handle.status,
            pid=handle.pid,
            pgid=handle.pgid,
            started_at=handle.started_at,
            finished_at=handle.finished_at,
            stdout_log=handle.log_path or '',
            config_path=handle.config_path,
            params={
                'config_path': handle.config_path,
                'output_dir': handle.output_dir,
                'experiment': _display_name(handle.config),
                'events_path': handle.events_path,
                'is_group': bool(handle.is_group),
                'is_study': bool(handle.is_study),
            },
            error=handle.error,
        )
        self.registry.update(state)

        from fmriflow.triage.service import trigger_on_failure
        trigger_on_failure(
            run_id=handle.run_id,
            kind='run',
            status=handle.status,
            state=state.to_dict(),
            run_dir=self.registry.run_dir(handle.run_id),
        )

    def _reattach_active_runs(self) -> None:
        for state in self.registry.list_active():
            if state.kind != 'run':
                continue
            if not RunRegistry.pid_alive(state.pid):
                self.registry.mark_lost(state, 'server_lost_track')
                continue

            params = state.params or {}
            config = {}
            cfg_path = params.get('config_path') or state.config_path
            if cfg_path and Path(cfg_path).is_file():
                try:
                    with open(cfg_path) as f:
                        config = yaml.safe_load(f) or {}
                except Exception:
                    config = {}

            # Re-detect kind from the YAML the subprocess is reading
            # from. Without this, reattached group/study runs look like
            # subject runs (is_group/is_study default to False) and the
            # in-flight graph endpoint silently falls into the subject
            # branch, returning a near-empty graph.
            re_is_study = (
                isinstance(config.get('study'), str)
                and isinstance(config.get('groups'), list)
            )
            re_is_group = (not re_is_study) and (
                isinstance(config.get('group'), str)
                and isinstance(config.get('subjects'), list)
            )
            handle = RunHandle(
                run_id=state.run_id,
                config=config,
                config_path=cfg_path,
                status='running',
                started_at=state.started_at,
                pid=state.pid,
                pgid=state.pgid,
                log_path=state.stdout_log,
                events_path=params.get('events_path'),
                output_dir=params.get('output_dir'),
                is_reattached=True,
                is_group=re_is_group,
                is_study=re_is_study,
            )
            self.active_runs[state.run_id] = handle

            monitor = _RunReattachedMonitor(handle, self, state)
            thread = threading.Thread(
                target=monitor.run, daemon=True, name=f"reattach-run-{state.run_id}",
            )
            thread.start()
            logger.info(
                "Reattached to pipeline run %s (pid=%s, experiment=%s)",
                state.run_id, state.pid, (params.get('experiment') or '?'),
            )

    def list_runs(self, include_finished: bool = True) -> list[dict]:
        out: dict[str, dict] = {}
        for handle in self.active_runs.values():
            row = handle.to_summary()
            # For group/study runs there's no top-level `experiment:`
            # in the YAML — the meaningful name lives in `group:` /
            # `study:`. Prefer those so the in-flight card shows
            # something useful instead of "(no experiment)".
            row['experiment'] = _display_name(handle.config)
            row['subject'] = handle.config.get('subject', '')
            row['is_group'] = bool(handle.is_group)
            row['is_study'] = bool(handle.is_study)
            out[handle.run_id] = row
        if include_finished:
            for state in self.registry.list_all():
                if state.kind != 'run' or state.run_id in out:
                    continue
                params = state.params or {}
                out[state.run_id] = {
                    'run_id': state.run_id,
                    'status': state.status,
                    'pid': state.pid,
                    'started_at': state.started_at,
                    'finished_at': state.finished_at,
                    'is_reattached': False,
                    'error': state.error,
                    'config_path': state.config_path,
                    'output_dir': params.get('output_dir'),
                    'log_path': state.stdout_log,
                    'experiment': _experiment_from_state(state),
                    'subject': state.subject,
                }
        return sorted(out.values(), key=lambda r: r.get('started_at') or 0, reverse=True)

    def get_run_live(self, run_id: str) -> dict | None:
        handle = self.active_runs.get(run_id)
        if handle is not None:
            summary = handle.to_summary()
            summary['experiment'] = _display_name(handle.config)
            summary['subject'] = handle.config.get('subject', '')
            summary['is_group'] = bool(handle.is_group)
            summary['is_study'] = bool(handle.is_study)
        else:
            state = self.registry.load(run_id)
            if state is None or state.kind != 'run':
                return None
            params = state.params or {}
            summary = {
                'run_id': state.run_id,
                'status': state.status,
                'pid': state.pid,
                'started_at': state.started_at,
                'finished_at': state.finished_at,
                'is_reattached': False,
                'error': state.error,
                'config_path': state.config_path,
                'output_dir': params.get('output_dir'),
                'log_path': state.stdout_log,
                'events_path': params.get('events_path'),
                'experiment': _experiment_from_state(state),
                'subject': state.subject,
            }
        log_path = summary.get('log_path')
        summary['log_tail'] = _read_tail(log_path, n=200) if log_path else ''
        summary['inner_stages'] = _parse_events_file(summary.get('events_path'))
        return summary

    def delete_run(self, run_id: str) -> dict:
        """Delete a finished analysis run.

        Refuses while running. Removes the registry dir and the
        run's per-run output subdir (which has been scoped to
        ``run_<ts>_<id>/`` since the per-run output-dir fix).
        """
        import shutil

        handle = self.active_runs.get(run_id)
        status = handle.status if handle else None
        state = self.registry.load(run_id)
        if status is None and state is not None:
            status = state.status
        if status == 'running':
            return {'deleted': False, 'reason': 'run is still running; cancel first'}
        if state is None and handle is None:
            return {'deleted': False, 'reason': 'run not found'}

        removed: list[str] = []
        params = (state.params if state else (handle.config if handle else {})) or {}
        output_dir = params.get('output_dir') or (handle.output_dir if handle else None)
        if output_dir:
            od = Path(output_dir)
            # Only nuke if it looks like the per-run subdir pattern
            # `run_<timestamp>_<id>`. Otherwise bail — user may have
            # hand-edited the config to a shared dir.
            if od.is_dir() and od.name.startswith(f'run_'):
                try:
                    shutil.rmtree(od)
                    removed.append(str(od))
                except OSError as e:
                    logger.warning('Could not remove %s: %s', od, e)

        self.active_runs.pop(run_id, None)
        existed = self.registry.delete(run_id)
        if not existed and not removed:
            return {'deleted': False, 'reason': 'nothing to delete'}
        return {'deleted': True, 'removed_paths': removed}

    def cancel_run(self, run_id: str) -> dict:
        handle = self.active_runs.get(run_id)
        if handle is None:
            return {'cancelled': False, 'reason': 'run not found in active set'}
        if handle.status != 'running':
            return {'cancelled': False, 'reason': f'status is {handle.status}'}
        pgid = handle.pgid or handle.pid
        if not pgid:
            return {'cancelled': False, 'reason': 'no pid recorded'}

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            handle.status = 'failed'
            handle.error = 'process already gone'
            self._persist_state(handle)
            return {'cancelled': True, 'reason': 'process already exited'}
        except Exception as e:
            return {'cancelled': False, 'reason': str(e)}

        def _grace_kill():
            time.sleep(5)
            if RunRegistry.pid_alive(handle.pid):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
        threading.Thread(target=_grace_kill, daemon=True).start()

        handle.status = 'cancelled'
        handle.finished_at = time.time()
        handle.push_event({'event': 'cancelled', 'message': 'SIGTERM sent'})
        self._persist_state(handle)
        return {'cancelled': True}

    # ── Legacy API (kept so existing callers keep working) ──────────

    def get_status(self, run_id: str) -> dict | None:
        handle = self.active_runs.get(run_id)
        if handle is None:
            return None
        new_events = handle.drain_events()
        return {
            'run_id': handle.run_id,
            'status': handle.status,
            'error': handle.error,
            'new_events': new_events,
            'all_events': handle.events,
        }

    def cleanup(self, max_age_s: float = 3600) -> None:
        to_remove = []
        for run_id, handle in self.active_runs.items():
            if handle.status in ('done', 'failed', 'cancelled', 'lost'):
                to_remove.append(run_id)
        for run_id in to_remove:
            del self.active_runs[run_id]


# ── Log tailer + reattached monitor ─────────────────────────────────────


class _RunLogTailer(threading.Thread):
    """Reads new lines from the pipeline log file and pushes them as events."""

    def __init__(
        self,
        log_path: Path,
        handle: RunHandle,
        stop_when,
        poll_interval: float = 0.5,
    ):
        super().__init__(daemon=True, name=f"run-tail-{handle.run_id}")
        self.log_path = log_path
        self.handle = handle
        self.stop_when = stop_when
        self.poll_interval = poll_interval
        self._stop_flag = threading.Event()

    def run(self) -> None:
        deadline = time.time() + 5
        while not self.log_path.is_file() and time.time() < deadline:
            time.sleep(0.1)
        if not self.log_path.is_file():
            return
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                while True:
                    line = f.readline()
                    if line:
                        self._emit(line.rstrip("\n"))
                        continue
                    if self._stop_flag.is_set() or self.stop_when():
                        tail = f.read()
                        if tail:
                            for ln in tail.splitlines():
                                self._emit(ln)
                        return
                    time.sleep(self.poll_interval)
        except Exception:
            logger.warning("Run log tailer crashed for %s", self.handle.run_id, exc_info=True)

    def _emit(self, line: str) -> None:
        self.handle.push_event({"event": "log", "message": line})

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self._stop_flag.set()
        self.join(timeout=timeout)


class _RunEventsTailer(threading.Thread):
    """Tails the subprocess's events.jsonl and pushes structured stage
    events into handle.events so the WebSocket can deliver them.

    Replaces the in-process UICaptureProxy path that is no longer
    reachable once the pipeline runs in a detached subprocess.
    """

    def __init__(
        self,
        events_path: Path,
        handle: "RunHandle",
        stop_when,
        poll_interval: float = 0.3,
    ):
        super().__init__(daemon=True, name=f"run-events-{handle.run_id}")
        self.events_path = events_path
        self.handle = handle
        self.stop_when = stop_when
        self.poll_interval = poll_interval
        self._stop_flag = threading.Event()

    def run(self) -> None:
        deadline = time.time() + 5
        while not self.events_path.is_file() and time.time() < deadline:
            time.sleep(0.1)
        if not self.events_path.is_file():
            return
        try:
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                while True:
                    line = f.readline()
                    if line:
                        self._emit(line.rstrip("\n"))
                        continue
                    if self._stop_flag.is_set() or self.stop_when():
                        tail = f.read()
                        if tail:
                            for ln in tail.splitlines():
                                self._emit(ln)
                        return
                    time.sleep(self.poll_interval)
        except Exception:
            logger.warning("Run events tailer crashed for %s", self.handle.run_id, exc_info=True)

    def _emit(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            ev = json.loads(line)
        except Exception:
            return
        if not isinstance(ev, dict) or "event" not in ev:
            return
        self.handle.push_event(ev)

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self._stop_flag.set()
        self.join(timeout=timeout)


class _RunReattachedMonitor:
    """Watches a reattached pipeline PID and tails its log file."""

    def __init__(
        self,
        handle: RunHandle,
        manager: "RunManager",
        state: RunStateFile,
    ):
        self.handle = handle
        self.manager = manager
        self.state = state

    def run(self) -> None:
        log_path = Path(self.handle.log_path) if self.handle.log_path else None
        events_path = Path(self.handle.events_path) if self.handle.events_path else None
        proc_dead = threading.Event()

        def stop_when() -> bool:
            if not RunRegistry.pid_alive(self.handle.pid):
                proc_dead.set()
                return True
            return False

        tailer = None
        events_tailer = None
        if log_path and log_path.is_file():
            tailer = _RunLogTailer(log_path, self.handle, stop_when=stop_when)
            tailer.start()
        if events_path and events_path.is_file():
            events_tailer = _RunEventsTailer(events_path, self.handle, stop_when=stop_when)
            events_tailer.start()

        while RunRegistry.pid_alive(self.handle.pid):
            time.sleep(1.0)
        proc_dead.set()

        if tailer is not None:
            tailer.stop_and_join()
        if events_tailer is not None:
            events_tailer.stop_and_join()

        self._finalize()

    def _finalize(self) -> None:
        # Reattach has no returncode — the subprocess was already detached
        # when the server came up. Derive ok/failed from the on-disk
        # summary's stage records via the shared helper, which also
        # handles group_summary.json / study_summary.json (the previous
        # implementation only looked for ``run_summary.json`` and
        # mis-reported every finished group/study run as "subprocess
        # exited without a run_summary.json").
        _apply_summary_to_handle(self.handle, returncode=None)
        self.manager._persist_state(self.handle)


def _read_tail(path: str | None, n: int = 200) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.is_file():
            return ""
        lines = p.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _resolve_summary_path(handle: RunHandle) -> tuple[Path | None, str]:
    """Locate the on-disk summary for a finished run, regardless of scope.

    Group / study orchestrators write their summary under a timestamped
    subdir of ``output_dir`` (``<output_dir>/<run_id>/(group|study)_summary.json``);
    subject runs write ``<output_dir>/run_summary.json`` directly. Returns
    the matched path (or ``None`` if missing) and the JSON key under which
    that summary stores its stage records.
    """
    if handle.is_study and handle.output_dir:
        parent = Path(handle.output_dir)
        if parent.is_dir():
            # _apply_per_run_output_dir writes ``run_<stamp>_<run_id>/`` so
            # the handle's own run_id always suffixes its directory name.
            # Prefer that exact match; fall back to newest-mtime for legacy
            # layouts where the suffix convention may not apply.
            exact = sorted(parent.glob(f'run_*_{handle.run_id}'))
            for p in exact:
                if (p / 'study_summary.json').is_file():
                    return p / 'study_summary.json', 'study_stages'
            candidates = [
                p for p in parent.iterdir()
                if p.is_dir() and p.name != 'latest'
                and (p / 'study_summary.json').is_file()
            ]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0] / 'study_summary.json', 'study_stages'
        return None, 'study_stages'
    if handle.is_group and handle.output_dir:
        parent = Path(handle.output_dir)
        if parent.is_dir():
            exact = sorted(parent.glob(f'run_*_{handle.run_id}'))
            for p in exact:
                if (p / 'group_summary.json').is_file():
                    return p / 'group_summary.json', 'group_stages'
            candidates = [
                p for p in parent.iterdir()
                if p.is_dir() and p.name != 'latest'
                and (p / 'group_summary.json').is_file()
            ]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0] / 'group_summary.json', 'group_stages'
        return None, 'group_stages'
    if handle.output_dir:
        path = Path(handle.output_dir) / 'run_summary.json'
        return (path if path.is_file() else None), 'stages'
    return None, 'stages'


def _apply_summary_to_handle(
    handle: RunHandle, *, returncode: int | None = None,
) -> None:
    """Drive a finished run's status / error / final event from the summary.

    Used by both the foreground monitor (which knows the subprocess's
    ``returncode``) and the reattached monitor (which doesn't — the
    process was already detached when the server came up, so we infer
    success from the recorded stage statuses).

    When ``returncode`` is ``None``, the run is considered successful
    iff a summary exists and every recorded stage's status is one of
    ``ok``/``warning``/``skipped``. Otherwise ``returncode == 0`` plus a
    present summary is required.
    """
    summary_path, stages_key = _resolve_summary_path(handle)
    summary: dict | None = None
    if summary_path and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = None

    now = time.time()
    handle.finished_at = now

    if returncode is None:
        if summary is None:
            ok = False
        else:
            stages = summary.get(stages_key, [])
            ok = bool(stages) and all(
                s.get('status') in ('ok', 'warning', 'skipped') for s in stages
            )
    else:
        ok = (returncode == 0 and summary is not None)

    if ok:
        handle.status = 'done'
        total_elapsed = (summary.get('total_elapsed_s', now - handle.started_at)
                         if summary else now - handle.started_at)
        handle.push_event({
            'event': 'run_done',
            'total_elapsed': total_elapsed,
            'summary_path': str(summary_path) if summary_path else None,
        })
        return

    handle.status = 'failed'
    summary_name = summary_path.name if summary_path else (
        'study_summary.json' if handle.is_study
        else 'group_summary.json' if handle.is_group
        else 'run_summary.json'
    )
    if summary is None and returncode == 0:
        handle.error = f'pipeline exited 0 but produced no {summary_name}'
    elif summary is None and returncode is not None:
        handle.error = _exit_code_message(returncode)
    elif summary is None:
        handle.error = f'subprocess exited without a {summary_name}'
    else:
        stages = summary.get(stages_key, [])
        failed_stage = next(
            (s for s in stages if s.get('status') == 'failed'),
            None,
        )
        if failed_stage:
            handle.error = (
                f"{failed_stage.get('name')}: "
                f"{failed_stage.get('detail') or 'stage failed'}"
            )
        elif returncode is not None:
            handle.error = _exit_code_message(returncode)
        else:
            handle.error = 'pipeline ended in an unknown state'

    run_error, node_errors = _failure_events(getattr(handle, 'events_path', None))
    if run_error and (summary is None or not handle.error):
        handle.error = run_error.get('error') or handle.error
    if node_errors and handle.error:
        first = node_errors[0]
        handle.error = f"{handle.error} (earlier failure: {first['node']}: {first['error']})"

    handle.push_event({
        'event': 'run_failed',
        'error': handle.error,
        'elapsed': now - handle.started_at,
        'log_tail': _read_tail(handle.log_path, n=200),
        'log_path': handle.log_path,
        'traceback': (run_error or {}).get('traceback'),
        'node_errors': node_errors,
    })


def _exit_code_message(returncode: int) -> str:
    """Explain a non-zero exit code, naming the signal when the process was killed."""
    import signal as _signal
    signum = -returncode if returncode < 0 else (returncode - 128 if returncode > 128 else None)
    if signum:
        try:
            name = _signal.Signals(signum).name
        except ValueError:
            name = f"signal {signum}"
        if signum == _signal.SIGKILL:
            return (f"pipeline was killed by {name} (exit code {returncode}), most likely out of memory: "
                    "the kernel stops the process without a Python error. Reduce memory use, e.g. "
                    "model.params.solver_params: {n_targets_batch: 10000, n_alphas_batch: 5} for himalaya models")
        return f"pipeline was killed by {name} (exit code {returncode})"
    return f"pipeline exited with code {returncode}"


def _failure_events(events_path: str | None) -> tuple[dict | None, list[dict]]:
    """The last ``run_error`` event and every ``node_fail`` from a run's events.jsonl."""
    run_error: dict | None = None
    node_errors: list[dict] = []
    if not events_path or not Path(events_path).is_file():
        return run_error, node_errors
    try:
        with open(events_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get('event') == 'run_error':
                    run_error = ev
                elif ev.get('event') == 'node_fail':
                    node_errors.append({'node': ev.get('node_id') or ev.get('name') or '?',
                                        'error': ev.get('error') or 'failed'})
    except OSError:
        pass
    return run_error, node_errors


# Known pipeline stages, in pipeline execution order.
_ANALYSIS_STAGES = (
    'stimuli', 'responses', 'features',
    'prepare', 'model', 'analyze', 'report',
)


def _parse_events_file(path: str | None) -> list[dict]:
    """Parse the pipeline subprocess's events.jsonl into a stage list.

    Each line is one JSON event:
      - stage_start {stage, t}
      - stage_done  {stage, t, elapsed, detail}
      - stage_fail  {stage, t, elapsed, error}
      - stage_warn  {stage, t, elapsed, detail}

    Returns the stages in the order they first appeared, each with
    its current status ('running' / 'ok' / 'warning' / 'failed'),
    started_at, finished_at, elapsed, detail, and error fields.
    Returns [] when there's no events file yet (e.g. subprocess
    just started, or this isn't an analysis run).
    """
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []

    # Preserve insertion order via dict + list to handle the unusual
    # case where a stage appears twice (e.g. resume).
    order: list[str] = []
    by_name: dict[str, dict] = {}

    try:
        raw = p.read_text(errors='replace')
    except Exception:
        return []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        event = ev.get('event')
        stage = ev.get('stage')
        if not stage or event not in (
            'stage_start', 'stage_done', 'stage_fail', 'stage_warn',
        ):
            continue

        if stage not in by_name:
            by_name[stage] = {
                'stage': stage,
                'status': 'pending',
                'started_at': 0.0,
                'finished_at': 0.0,
                'elapsed': 0.0,
                'detail': '',
                'error': None,
            }
            order.append(stage)

        slot = by_name[stage]
        t = ev.get('t', 0.0)
        if event == 'stage_start':
            slot['status'] = 'running'
            slot['started_at'] = t
        elif event == 'stage_done':
            slot['status'] = 'ok'
            slot['finished_at'] = t
            slot['elapsed'] = ev.get('elapsed', 0.0)
            slot['detail'] = ev.get('detail', '')
        elif event == 'stage_warn':
            slot['status'] = 'warning'
            slot['finished_at'] = t
            slot['elapsed'] = ev.get('elapsed', 0.0)
            slot['detail'] = ev.get('detail', '')
        elif event == 'stage_fail':
            slot['status'] = 'failed'
            slot['finished_at'] = t
            slot['elapsed'] = ev.get('elapsed', 0.0)
            slot['error'] = ev.get('error', '')

    return [by_name[s] for s in order]
