from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from baseai.config import Config


@dataclass
class SkillMetadata:
    name: str
    description: str
    version: str = "1.0.0"
    author: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillMetadata":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
        )


@dataclass
class Skill:
    metadata: SkillMetadata
    path: Path


class SkillsManager:
    """Manager for agent skills"""

    SKILL_FILE = "SKILL.md"

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self._all: dict[str, Skill] = {}
        self._disabled: set[str] = set()
        self._sync()

    def _parse(self, skill_dir: Path) -> Skill | None:
        """Parse SKILL.md file in directory and extract metadata."""
        skill_file = skill_dir / self.SKILL_FILE
        if not skill_file.exists():
            return None

        try:
            raw = skill_file.read_text(encoding="utf-8")
        except OSError:
            return None

        if not raw.startswith("---"):
            return None

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return None

        try:
            metadata_dict = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            return None

        if not isinstance(metadata_dict, dict) or not metadata_dict.get("name"):
            return None

        return Skill(
            metadata=SkillMetadata.from_dict(metadata_dict),
            path=skill_dir,
        )

    def _sync(self) -> None:
        """Sync skills: remove deleted, add new skills from directory."""
        if not self.skills_dir.exists():
            self._all.clear()
            return

        current_names: set[str] = {
            d.name
            for d in self.skills_dir.iterdir()
            if d.is_dir() and (d / self.SKILL_FILE).exists()
        }

        for name in list(self._all.keys()):
            if name not in current_names:
                self._all.pop(name, None)

        for name in current_names:
            if name not in self._all:
                skill = self._parse(self.skills_dir / name)
                if skill and skill.metadata.name:
                    self._all[name] = skill

    def reload(self) -> None:
        """Reload skills from directory."""
        self._sync()

    def upload(self, source: Path) -> str | None:
        """Upload skill from source path to skills directory. Returns installed name."""
        if not source.exists() or not source.is_dir():
            return None

        source_skill = self._parse(source)
        if not source_skill:
            return None

        name = source_skill.metadata.name
        target = self.skills_dir / name

        if target.exists():
            suffix = 1
            while (self.skills_dir / f"{name}_{suffix}").exists():
                suffix += 1
            name = f"{name}_{suffix}"
            target = self.skills_dir / name

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._copy_dir(source, target)

        skill = self._parse(target)
        if skill:
            self._all[name] = skill

        return name

    def unload(self, name: str) -> bool:
        """Unload skill by name from skills directory."""
        target = self.skills_dir / name
        if not target.is_dir():
            return False

        self._remove_dir(target)
        self._all.pop(name, None)
        self._disabled.discard(name)
        return True

    @staticmethod
    def _copy_dir(src: Path, dst: Path) -> None:
        """Recursively copy directory contents."""
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            src_item = src / item.name
            dst_item = dst / item.name
            if src_item.is_dir():
                SkillsManager._copy_dir(src_item, dst_item)
            else:
                dst_item.write_bytes(src_item.read_bytes())

    @staticmethod
    def _remove_dir(path: Path) -> None:
        """Recursively remove directory or file."""
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            for item in path.iterdir():
                SkillsManager._remove_dir(item)
            path.rmdir()

    def enable(self, name: str, config: Config) -> bool:
        """Enable a skill by name."""
        if name not in self._all:
            return False
        self._disabled.discard(name)
        return True

    def disable(self, name: str, config: Config) -> bool:
        """Disable a skill by name."""
        if name not in self._all:
            return False
        self._disabled.add(name)
        return True

    def is_enabled(self, name: str) -> bool:
        """Check if skill is enabled."""
        return name in self._all and name not in self._disabled

    def get(self, name: str) -> Skill | None:
        """Get skill by name."""
        return self._all.get(name)

    def list(self) -> list[Skill]:
        """List all loaded skills."""
        return [self._all[name] for name in sorted(self._all.keys())]

    def get_summary(self) -> str:
        """Generate XML summary of enabled skills for agent context."""
        if not self._all:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for name in sorted(self._all.keys()):
            if not self.is_enabled(name):
                continue

            skill = self._all[name]
            name = escape_xml(skill.metadata.name)
            description = escape_xml(skill.metadata.description)

            lines.append("  <skill>")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{description}</description>")
            lines.append(f"    <path>{skill.path / self.SKILL_FILE}</path>")
            lines.append("  </skill>")

        lines.append("</skills>")

        return "\n".join(lines)
