import yaml
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "skills"

class SkillsLoader:
    """Loader for agent skills"""

    def __init__(self, workspace_skills: Path,):
        self.workspace_dir = workspace_skills
        self.builtin_dir = BUILTIN_DIR

    def get_skills_summary(self) -> str:
        """Get a summary of skills including name and description"""
        all_skills = self.list_skills()

        skills = []
        for skill in all_skills:
            skill_metadata = self.get_metadata(skill["name"])
            skills.append(f"-{skill_metadata["name"]}: {skill_metadata["description"]}")

        return "\n".join(skills)

    def list_skills(self) -> list[str]:
        """List all of available skills"""
        skills = []

        # workspace (优先级更高)
        if self.workspace_dir.exists():
            for skill_dir in self.workspace_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "source": "custom"})

        # built-in
        if self.builtin_dir.exists():
            for skill_dir in self.builtin_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "source": "built-in"})

        return skills

    def read_skill(self, name: str) -> str | None:
        """根据名称读取skill"""
        for dir in (self.workspace_dir, self.builtin_dir):
            skill_file =  dir / name / "SKILL.md"
            if skill_file.exists():
                return skill_file.read_text(encoding="utf-8")

        return None

    def get_metadata(self, name: str) -> dict | None:
        """根据名称获取skill的元数据"""
        skill = self.read_skill(name)

        if skill and skill.startswith('---\n'):
            parts = skill.split('---\n', 2)
            yaml_str = parts[1]
            try:
                return yaml.safe_load(yaml_str)
            except yaml.YAMLError:
                return None
        return None
    
    def get_content(self, name: str) -> dict | None:
        """根据名称获取skill的内容"""
        skill = self.read_skill(name)

        if skill and skill.startswith('---\n'):
            parts = skill.split('---\n', 2)
            return parts[2]
        
        return None
    