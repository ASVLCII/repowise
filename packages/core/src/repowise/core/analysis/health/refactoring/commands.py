"""The command that runs a plan's tests, shared by plan validation and staged-plan batches."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from ....ingestion.package_roots import module_for, owning_package
from ....ingestion.resolvers.dotnet.msbuild import parse_csproj
from ...test_selection import JS_SUFFIXES, PYTHON_SUFFIXES

#: Each test file's runner location, for JS/TS and C# alike; plans store it as
#: ``js_packages``. A JS/TS test's value is the directory
#: (``""`` is the repository root) whose package.json declares a ``test``
#: script; a C# test's is the project or solution ``dotnet test`` takes. None
#: when nothing can run the file; a file the map does not hold is None too.
JsPackages = Mapping[str, str | None]

_SOLUTIONS = (".sln", ".slnx")


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


def _is_dotnet(test: str) -> bool:
    return _file(test).lower().endswith(".cs")


#: NuGet ids that make a project a test project: the test SDK and each
#: framework's runner. Ceiling: a project that gets them only from shared
#: build files (``Directory.Build.*``, imported targets) and does not set
#: ``IsTestProject`` itself is not seen as one; its tests fall back to the
#: solution.
_TEST_PACKAGES = frozenset(
    {
        "microsoft.net.test.sdk",
        "mstest",
        "mstest.testadapter",
        "nunit3testadapter",
        "tunit",
        "xunit",
        "xunit.runner.visualstudio",
        "xunit.v3",
        "xunit.v3.mtp-v1",
        "xunit.v3.mtp-v2",
    }
)


def _project_graph(root: Path, projects: list[str]) -> tuple[dict[str, set[str]], set[str]]:
    """Over *projects* (repo-relative ``.csproj`` paths): each referenced one
    mapped to the projects whose ``ProjectReference`` names it, and the test
    projects, which say so or declare a test package. An app or library that
    nothing references is not one. Ceiling: one parse per project, per call."""
    by_path = {(root / project).resolve(): project for project in projects}
    referrers: dict[str, set[str]] = {}
    test_projects: set[str] = set()
    for project in projects:
        parsed = parse_csproj(root / project)
        if parsed is None:
            continue
        packages = {package.lower() for package in parsed.package_references}
        if parsed.is_test_project or not _TEST_PACKAGES.isdisjoint(packages):
            test_projects.add(project)
        for target in parsed.project_references:
            if target in by_path:
                referrers.setdefault(by_path[target], set()).add(project)
    return referrers, test_projects


def _root_files(root: Path) -> tuple[list[str], str | None]:
    """The ``.csproj`` files at the repository root, which the package scan
    leaves out, and its solution: the only one, or the one named after the
    checkout; None when there is none or the choice is a guess."""
    try:
        names = sorted(p.name for p in root.iterdir())
    except OSError:
        return [], None
    projects = [name for name in names if name.lower().endswith(".csproj")]
    solutions = [name for name in names if Path(name).suffix.lower() in _SOLUTIONS]
    if len(solutions) > 1:
        solutions = [name for name in solutions if Path(name).stem.lower() == root.name.lower()]
    return projects, solutions[0] if len(solutions) == 1 else None


def _test_project(
    path: str,
    by_dir: Mapping[str, list[str]],
    referrers: Mapping[str, set[str]],
    test_projects: set[str],
) -> str | None:
    """The test project that runs C# file *path*: the one project in its
    directory (the root's when no deeper one holds it) when that is a test
    project, else the only test project that references it (*path* is in a
    test helper library); None otherwise."""
    owner = module_for(path, by_dir.keys())
    projects = by_dir.get(owner, []) if owner in by_dir else by_dir.get("", [])
    if len(projects) != 1:
        return None
    if projects[0] in test_projects:
        return projects[0]
    users = [user for user in referrers.get(projects[0], ()) if user in test_projects]
    return users[0] if len(users) == 1 else None


def dotnet_test_projects(
    root: Path, manifests: Mapping[str, frozenset[str]], tests: Iterable[str]
) -> dict[str, str | None]:
    """The project ``dotnet test`` runs each C# test in *tests* through, from
    the ``.csproj`` names *manifests* (:func:`scan_package_manifests`) records;
    the root solution when no single test project runs the file, else None.
    Blocking file reads, and none without a C# test: an async caller uses a
    thread."""
    files = {_file(test) for test in tests if _is_dotnet(test)}
    if not files:
        return {}
    root_projects, solution = _root_files(root)
    by_dir: dict[str, list[str]] = {"": root_projects} if root_projects else {}
    for directory, names in manifests.items():
        for name in sorted(names):
            if name.lower().endswith(".csproj"):
                by_dir.setdefault(directory, []).append(f"{directory}/{name}")
    referrers, test_projects = _project_graph(root, [p for ps in by_dir.values() for p in ps])
    return {
        path: _test_project(path, by_dir, referrers, test_projects) or solution for path in files
    }


def packages_for_tests(
    root: Path, manifests: Mapping[str, frozenset[str]], tests: Iterable[str]
) -> dict[str, str | None]:
    """Where each test in *tests* runs from (:data:`JsPackages`), every language."""
    tests = list(tests)
    return {
        **js_test_packages(root, manifests, tests),
        **dotnet_test_projects(root, manifests, tests),
    }


def plan_packages(tests: Iterable[str], packages: JsPackages | None) -> dict[str, str | None]:
    """The part of *packages* a plan's *tests* use, stored on the plan so a
    later read (batches, a served plan) needs no checkout. Empty without a map."""
    if packages is None:
        return {}
    files = {_file(test) for test in tests}
    return {path: packages.get(path) for path in files if _is_js(path) or path in packages}


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


def _dotnet_filter(test: str) -> str | None:
    """``Class.Method`` for a ``file::...::Class::Method`` id: a substring of
    the test's fully qualified name, so a namespace or an outer class in front
    still matches. None for a file, or a name the filter syntax would misread."""
    name = ".".join(test.split("::")[1:][-2:])
    return name if re.fullmatch(r"\w+(\.\w+)?", name) else None


def _filter_arg(names: set[str]) -> str:
    """``--filter`` for *names*, with coverage collection off: a coverage gate
    (coverlet's threshold) fails any filtered run. Double quotes keep ``|``
    from every shell."""
    joined = "|".join(sorted(names))
    return (f'--filter "{joined}"' if len(names) > 1 else f"--filter {joined}") + (
        " -p:CollectCoverage=false"
    )


def _dotnet_commands(tests: list[str], packages: JsPackages | None) -> list[str]:
    """``dotnet test`` per project, filtered to the named tests; the whole
    project when any of its tests is a file. A test no project runs gets none."""
    groups: dict[str, set[str | None]] = {}
    for test in tests:
        project = packages.get(_file(test)) if packages is not None else None
        if project is not None:
            groups.setdefault(project, set()).add(_dotnet_filter(test))
    commands = []
    for project, names in sorted(groups.items()):
        command = f"dotnet test {_arg(project)}"
        if None not in names:
            command += " " + _filter_arg({name for name in names if name is not None})
        commands.append(command)
    return commands


def commands_for_tests(
    tests: list[str], *, total: int | None = None, packages: JsPackages | None = None
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
    package; a test no package.json can run gets no command. A C# test runs
    through the project or solution *packages* names for it. A language other
    than these gets none in place of a guess that would fail when run.
    """
    if total is not None and total > len(tests):
        tests = sorted({_file(test) for test in tests})
    python_tests = sorted(test for test in tests if _file(test).endswith(PYTHON_SUFFIXES))
    js_tests = sorted(test for test in tests if _is_js(test))
    commands: list[str] = []
    if python_tests:
        commands.append("pytest " + " ".join(map(_arg, python_tests)))
    commands.extend(_js_commands(js_tests, packages))
    commands.extend(_dotnet_commands([test for test in tests if _is_dotnet(test)], packages))
    return commands


__all__ = [
    "JsPackages",
    "commands_for_tests",
    "dotnet_test_projects",
    "js_test_packages",
    "packages_for_tests",
    "plan_packages",
]
