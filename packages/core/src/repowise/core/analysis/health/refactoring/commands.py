"""The command that runs a plan's tests, shared by plan validation and staged-plan batches."""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from ....ingestion.package_roots import owning_package
from ....ingestion.workspace_members import (
    _toml,
    declared_workspace_members,
    manifest_package_name,
)
from ...test_selection import JS_SUFFIXES, PYTHON_SUFFIXES

#: Each JS/TS test file's package: the directory (``""`` is the repository
#: root) whose package.json declares a ``test`` script, or None when no
#: package.json above the file does. A file the map does not hold is None too.
JsPackages = Mapping[str, str | None]

#: Each test file of a language whose runner works per package (Go, Rust, JVM,
#: Ruby), mapped to ``[run, arg]``: the command that runs its package from the
#: repository root, and the argument narrowing it to the file's own target, or
#: None to run the whole package. A file the map does not hold gets no command.
TestRuns = Mapping[str, Sequence[str | None]]


def _file(test: str) -> str:
    return test.split("::", 1)[0]


def _is_js(test: str) -> bool:
    return _file(test).lower().endswith(JS_SUFFIXES)


def _arg(path: str) -> str:
    # Double quotes hold a space in sh, PowerShell and cmd alike. Ceiling: a
    # path with ``"``, ``$`` or a backtick is left as is; upgrade by quoting
    # per shell once a plan names its shell.
    return f'"{path}"' if re.search(r"\s", path) and not re.search(r'["$`]', path) else path


def _declares_test(manifest: Path) -> bool:
    try:
        scripts = json.loads(manifest.read_text(encoding="utf-8")).get("scripts")
    except (OSError, ValueError, AttributeError):
        return False
    script = scripts.get("test") if isinstance(scripts, dict) else None
    # npm init's placeholder exits 1 without running anything.
    return isinstance(script, str) and bool(script) and "no test specified" not in script


def js_test_packages(
    root: Path, manifests: Mapping[str, frozenset[str]], tests: Iterable[str]
) -> dict[str, str | None]:
    """The package that runs each JS/TS test in *tests*: the nearest package
    root (*manifests*, from :func:`scan_package_manifests`) whose package.json
    declares a test script. Blocking file reads: an async caller uses a thread.
    """
    declares: dict[str, bool] = {}

    def accepts(directory: str) -> bool:
        if directory not in declares:
            declares[directory] = _declares_test(root / directory / "package.json")
        return declares[directory]

    return {
        path: owning_package(path, manifests, "package.json", accepts)
        for path in {_file(test) for test in tests if _is_js(test)}
    }


def plan_packages(tests: Iterable[str], packages: JsPackages | None) -> dict[str, str | None]:
    """The part of *packages* a plan's *tests* use, stored on the plan so a
    later read (batches, a served plan) needs no checkout. Empty without a map."""
    if packages is None:
        return {}
    return {path: packages.get(path) for path in {_file(test) for test in tests if _is_js(test)}}


def _relative(path: str, owner: str) -> str:
    return path[len(owner) + 1 :] if owner else path


@dataclass
class _Checkout:
    """One repository's package manifests, each extra file read at most once."""

    root: Path
    manifests: Mapping[str, frozenset[str]]
    _memo: dict[tuple[Hashable, ...], Any] = field(default_factory=dict)

    def memo(self, read: Callable[[Path], Any], path: Path) -> Any:
        """*read* of *path*, once per checkout: every test of a package asks again."""
        key = (read, path)
        if key not in self._memo:
            self._memo[key] = read(path)
        return self._memo[key]

    def has(self, directory: str, name: str) -> bool:
        return self.memo(Path.is_file, self.root / directory / name)

    def owner(
        self, path: str, name: str, accepts: Callable[[str], bool] | None = None
    ) -> str | None:
        # The scan leaves the root manifest out, so the default checks the disk.
        return owning_package(
            path, self.manifests, name, accepts or (lambda directory: self.has(directory, name))
        )

    @cached_property
    def cargo_members(self) -> set[str]:
        return declared_workspace_members(self.root).get("Cargo.toml", set())

    @cached_property
    def maven_reactor(self) -> frozenset[str]:
        """Module directories the root POM reaches through ``<modules>``."""
        from ....ingestion.external_systems.maven_model import load_maven_reactor
        from ....workspace.maven_dependencies import _declared_reactor_projects

        poms = [self.root / "pom.xml"] + [
            self.root / directory / "pom.xml"
            for directory, names in self.manifests.items()
            if "pom.xml" in names
        ]
        projects = load_maven_reactor(self.root, poms).projects
        if not any(project.manifest == "pom.xml" for project in projects):
            return frozenset()
        return frozenset(
            posixpath.dirname(project.manifest)
            for project in _declared_reactor_projects(projects, ())
        )

    @cached_property
    def gradle_projects(self) -> dict[str, str]:
        """Each project directory the root settings file includes, to its project path."""
        from ....ingestion.resolvers.jvm_gradle import _parse_settings

        for name in ("settings.gradle.kts", "settings.gradle"):
            if self.has("", name):
                modules, overrides = _parse_settings(self.root / name)
                projects: dict[str, str] = {}
                for module in modules:
                    directory = overrides.get(module) or module.replace(":", "/")
                    projects[posixpath.normpath(directory)] = f":{module}"
                return projects
        return {}


Run = tuple[str, str | None]


def _go_run(repo: _Checkout, path: str) -> Run | None:
    """``go test`` on the test's package directory, inside the module owning it."""
    module = repo.owner(path, "go.mod")
    if module is None:
        return None
    package = posixpath.dirname(_relative(path, module))
    run = f"go -C {_arg(module)} test" if module else "go test"
    return run, _arg(f"./{package}" if package else ".")


def _rust_test_target(data: Mapping[str, Any], rel: str) -> str | None:
    """The integration-test target whose root is *rel*, by the crate's *data*."""
    declared = data.get("test")
    for target in declared if isinstance(declared, list) else ():
        name = target.get("name") if isinstance(target, dict) else None
        path = isinstance(name, str) and (target.get("path") or f"tests/{name}.rs")
        if isinstance(path, str) and posixpath.normpath(path) == rel:
            return name
    package = data.get("package")
    if isinstance(package, dict) and package.get("autotests") is False:
        return None
    parts = rel.split("/")
    if len(parts) == 2 and parts[0] == "tests" and parts[1].endswith(".rs"):
        return parts[1][:-3]
    return parts[1] if len(parts) == 3 and parts[0] == "tests" and parts[2] == "main.rs" else None


def _rust_run(repo: _Checkout, path: str) -> Run | None:
    """``cargo test`` on the crate owning the test; a virtual workspace root is no crate."""

    def name(directory: str) -> str | None:
        return repo.memo(manifest_package_name, repo.root / directory / "Cargo.toml")

    crate = repo.owner(path, "Cargo.toml", lambda directory: name(directory) is not None)
    if crate is None:
        return None
    # ``-p`` finds only crates of the workspace at the root; any other needs its manifest.
    run = (
        f"cargo test -p {name(crate)}"
        if not crate or crate in repo.cargo_members
        else f"cargo test --manifest-path {_arg(crate + '/Cargo.toml')}"
    )
    data = repo.memo(_toml, repo.root / crate / "Cargo.toml")
    target = _rust_test_target(data, _relative(path, crate))
    return run, f"--test {target}" if target else None


def _maven_run(repo: _Checkout, module: str) -> Run:
    mvn = "./mvnw" if repo.has("", "mvnw") else "mvn"
    if not module:
        return f"{mvn} test", None
    if module in repo.maven_reactor:
        # ``-am`` builds the modules it depends on; bare ``-pl`` would look for
        # them in the local repository and fail on a fresh checkout.
        return f"{mvn} -pl {_arg(module)} -am test", None
    return f"{mvn} -f {_arg(module + '/pom.xml')} test", None


def _gradle_run(repo: _Checkout, module: str) -> Run | None:
    gradle = "./gradlew" if repo.has("", "gradlew") else "gradle"
    if not module:
        return f"{gradle} test", None
    project = repo.gradle_projects.get(module)
    return (f"{gradle} {project}:test", None) if project else None


_GRADLE_SCRIPTS = ("build.gradle.kts", "build.gradle")


def _jvm_run(repo: _Checkout, path: str) -> Run | None:
    """Maven or Gradle on the deepest module owning the test."""
    maven = repo.owner(path, "pom.xml")
    gradle = max(
        (owner for name in _GRADLE_SCRIPTS if (owner := repo.owner(path, name)) is not None),
        key=len,
        default=None,
    )
    # Both own the same directory when a repo ships both builds; Maven then runs.
    if maven is not None and (gradle is None or len(maven) >= len(gradle)):
        return _maven_run(repo, maven)
    return _gradle_run(repo, gradle) if gradle is not None else None


_RAKE_TEST = re.compile(r"""TestTask|\btask\s*\(?\s*:?['"]?test\b""")


def _declares_rake_test(rakefile: Path) -> bool:
    try:
        return bool(_RAKE_TEST.search(rakefile.read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return False


def _ruby_run(repo: _Checkout, path: str) -> Run | None:
    """RSpec for a file under its Gemfile directory's ``spec/``, else the Rakefile's test task."""
    owner = repo.owner(path, "Gemfile")
    if owner is None:
        return None
    rel = _relative(path, owner)
    # Ceiling: ``&&`` fails in Windows PowerShell 5.1, as quoting does in _arg.
    cd = f"cd {_arg(owner)} && " if owner else ""
    if rel.startswith("spec/"):
        # A spec helper is no example file, so the whole suite runs.
        return f"{cd}bundle exec rspec", _arg(rel) if rel.endswith("_spec.rb") else None
    if repo.memo(_declares_rake_test, repo.root / owner / "Rakefile"):
        return f"{cd}bundle exec rake test", None
    return None


#: Test-file suffixes and the function naming the run, per language whose
#: runner works per package.
_RUNNERS: tuple[tuple[tuple[str, ...], Callable[[_Checkout, str], Run | None]], ...] = (
    ((".go",), _go_run),
    ((".rs",), _rust_run),
    ((".java", ".kt"), _jvm_run),
    ((".rb",), _ruby_run),
)


def runs_for_tests(
    root: Path, manifests: Mapping[str, frozenset[str]], tests: Iterable[str]
) -> dict[str, list[str | None]]:
    """The run (:data:`TestRuns`) of each test in *tests* whose language is in
    :data:`_RUNNERS`, from the package roots in *manifests*. A test whose runner
    or manifest the repository lacks is left out. Blocking file reads.
    """
    repo = _Checkout(root, manifests)
    runs: dict[str, list[str | None]] = {}
    for path in sorted({_file(test) for test in tests}):
        lowered = path.lower()
        runner = next((run for suffixes, run in _RUNNERS if lowered.endswith(suffixes)), None)
        if runner is not None and (found := runner(repo, path)) is not None:
            runs[path] = list(found)
    return runs


def plan_runs(tests: Iterable[str], runs: TestRuns | None) -> dict[str, list[str | None]]:
    """The part of *runs* a plan's *tests* use, stored like :func:`plan_packages`."""
    if runs is None:
        return {}
    return {path: list(runs[path]) for path in {_file(test) for test in tests} if path in runs}


def _package_commands(tests: list[str], runs: TestRuns) -> list[str]:
    """One command per run naming every target its tests need; a test that
    needs its whole package widens the run to the whole package."""
    groups: dict[str, set[str] | None] = {}
    for path in {_file(test) for test in tests}:
        if path not in runs:
            continue
        run, arg = runs[path]
        if arg is None:
            groups[run] = None
        elif (args := groups.setdefault(run, set())) is not None:
            args.add(arg)
    return [
        " ".join([run, *sorted(args or ())]) for run, args in sorted(groups.items())
    ]


def _js_commands(tests: list[str], packages: JsPackages | None) -> list[str]:
    """``npm test`` from the package each test belongs to; a test no package
    can run gets no command. Without *packages* every test runs from the root."""
    groups: dict[str, list[str]] = {}
    for test in tests:
        owner = packages.get(_file(test)) if packages is not None else ""
        if owner is not None:
            groups.setdefault(owner, []).append(test[len(owner) + 1 :] if owner else test)
    return [
        ("npm test -- " if not owner else f"npm --prefix {_arg(owner)} test -- ")
        + " ".join(map(_arg, paths))
        for owner, paths in sorted(groups.items())
    ]


def commands_for_tests(
    tests: list[str],
    *,
    total: int | None = None,
    packages: JsPackages | None = None,
    runs: TestRuns | None = None,
) -> list[str]:
    """A command that runs at least everything the plan says validates it.

    When the displayed list is capped, enumerating only the shown tests would
    read as a complete validation run while silently skipping the rest, so the
    selection widens to the files those tests live in: bounded by file count
    rather than test count, and never narrower than the evidence.

    An empty list means the plan has no command to suggest. Every command names
    its tests: with none found, a bare ``pytest`` or ``npm test`` would run a
    whole suite as if it guarded the change. A JS/TS test runs from the package
    whose ``test`` script owns it (*packages*), with its path relative to that
    package; a test no package.json can run gets no command. A Go, Rust, JVM or
    Ruby test runs its package, or the narrower target *runs* names for it. Any
    other test, or one *runs* does not hold, gets none in place of a guess that
    would fail when run.
    """
    if total is not None and total > len(tests):
        tests = sorted({_file(test) for test in tests})
    python_tests = sorted(test for test in tests if _file(test).endswith(PYTHON_SUFFIXES))
    js_tests = sorted(test for test in tests if _is_js(test))
    commands: list[str] = []
    if python_tests:
        commands.append("pytest " + " ".join(map(_arg, python_tests)))
    commands.extend(_js_commands(js_tests, packages))
    if runs:
        commands.extend(_package_commands(tests, runs))
    return commands


__all__ = [
    "JsPackages",
    "TestRuns",
    "commands_for_tests",
    "js_test_packages",
    "plan_packages",
    "plan_runs",
    "runs_for_tests",
]
