"""The docs follow DOCS-GUIDE.md's rules and stay in step with the code: every
page has the guide's header, every page link and anchor resolves, and the
reference pages list every public function, config field and CLI command."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

import mayaku
from mayaku.cli.main import app
from mayaku.config import KeypointConfig, MayakuConfig

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
FENCE = re.compile(r"```.*?```", re.S)

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="docs/ not in this checkout")


def prose(page: Path) -> str:
    return FENCE.sub("", page.read_text())


def header(page: Path) -> dict[str, str]:
    m = re.match(r"# .+\n\n<!--\n(.*?)\n-->\n", page.read_text(), re.S)
    assert m, f"{page} must start with '# Title' and an HTML comment header"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def anchors(page: Path) -> set[str]:
    out: set[str] = set()
    seen: dict[str, int] = {}
    for heading in re.findall(r"^#{2,6} (.+)$", prose(page), re.M):
        slug = re.sub(r"[^a-z0-9\-_]", "", heading.lower().replace(" ", "-"))
        n = seen.get(slug, 0)
        out.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return out


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_page_header(page: Path) -> None:
    fields = header(page)
    is_index = page == DOCS / "index.md"
    assert set(fields) == {"summary", "label", "keywords"} | ({"template"} if is_index else set())
    assert len(fields["summary"]) < 160
    assert 1 <= len(fields["label"].split()) <= 3
    assert fields["keywords"].split(", ")[0] == "mayaku"
    assert len(re.findall(r"^# ", prose(page), re.M)) == 1


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_links_resolve(page: Path) -> None:
    text = prose(page)
    assert not re.findall(r"\]\(/mayaku/", text), "link pages as /docs/...md"
    for target in re.findall(r"\]\((/docs/[^)\s]+|#[^)\s]+)\)", text):
        path, _, anchor = target.partition("#")
        dest = ROOT / path.lstrip("/") if path else page
        assert dest.exists(), target
        if anchor:
            assert anchor in anchors(dest), target


def test_python_api_lists_public_functions() -> None:
    page = (DOCS / "reference" / "python-api.md").read_text()
    for name in mayaku.__all__:
        section = page.split(f"## mayaku.{name}\n", 1)
        assert len(section) == 2, f"mayaku.{name} has no section"
        body = section[1].split("\n## ", 1)[0]
        for param in inspect.signature(getattr(mayaku, name)).parameters:
            assert param in body, f"mayaku.{name}({param}) is not documented"


def config_fields(obj: object, prefix: str = "") -> list[str]:
    if isinstance(obj, BaseModel):
        items = {k: getattr(obj, k) for k in type(obj).model_fields}
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    else:
        return [prefix]
    return [f for k, v in items.items() for f in config_fields(v, f"{prefix}{k}.")]


def test_configuration_lists_every_field() -> None:
    page = (DOCS / "reference" / "configuration.md").read_text()
    fields = [f.rstrip(".") for f in config_fields(MayakuConfig())]
    fields += [f"model.keypoints.{k}" for k in KeypointConfig.model_fields]
    rows = set(re.findall(r"^\| `([\w.]+)` \|", page, re.M))
    # the train.aug table, shared by train.final_aug, lists the short names
    missing = [f for f in fields if f not in rows and f.rsplit(".", 1)[-1] not in rows]
    assert not missing


def test_cli_lists_every_command() -> None:
    page = (DOCS / "reference" / "cli.md").read_text()
    for command in app.registered_commands:
        assert f"## mayaku {command.name}\n" in page


def test_llms_txt_lists_every_page() -> None:
    listed = dict(re.findall(r"^- \[.+?\]\((/docs/[^)]+)\): (.+)$",
                             (DOCS / "llms.txt").read_text(), re.M))
    for page in PAGES:
        assert listed.get(f"/{page.relative_to(ROOT).as_posix()}") == header(page)["summary"], page
    assert len(listed) == len(PAGES)
