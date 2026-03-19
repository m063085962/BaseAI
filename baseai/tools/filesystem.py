import difflib
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

def resolve_path(path: str, config: RunnableConfig,) -> Path:
    """根据工作区解析路径（如果是相对路径），并检查路径限制"""
    workspace = config.get("configurable").get("workspace")
    restrict = config.get("configurable").get("restrict_to_workspace", True)
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if restrict:
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            raise PermissionError(f"路径 {path} 在限制路径 {workspace.resolve()} 之外")
    return resolved


class ListDirSchema(BaseModel):
    path: Optional[str] = Field(description="要列出的目录路径")

@tool(args_schema=ListDirSchema)
def list_dir(path: str, config: RunnableConfig) -> str:
    """列出指定目录下的内容"""
    try:
        dir_path = resolve_path(path, config)
        if not dir_path.exists():
            return f"Error: 未找到目录：{path}"
        if not dir_path.is_dir():
            return f"Error: {path}不是一个目录。"

        items = []
        for item in dir_path.iterdir():
            sufix ="/" if item.is_dir() else ""
            items.append(f"{item.name}{sufix}")

        if not items:
            f"目录 '{path}' 为空。"

        return "  ".join(items)
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error listing diretory: {e}"


class ReadFileSchema(BaseModel):
    path: str = Field(description="要读取的文件路径")

@tool(args_schema=ReadFileSchema)
def read_file(path: str, config: RunnableConfig) -> str:
    """读取指定文件的内容"""
    try:
        file_path = resolve_path(path, config)
        if not file_path.exists():
            return f"Error: 未找到文件: {path}"
        if not file_path.is_file():
            return f"Error: {path}不是一个文件"

        content = file_path.read_text(encoding="utf-8")
        return content
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


class WriteFileSchema(BaseModel):
    path: str = Field(description="要写入的文件路径")
    content: str = Field(description="要写入的内容")

@tool(args_schema=WriteFileSchema)
def write_file(
    path: str,
    content: str,
    config: RunnableConfig
) -> str:
    """将内容写入指定文件"""
    try:
        file_path = resolve_path(path, config)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"成功写入 {len(content)} bytes 到文件{path}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


class EditFileSchema(BaseModel):
    path: str = Field(description="要编辑的文件路径")
    old_text: str = Field(description="要查找并替换的确切文本")
    new_text: str = Field(description="要替换成的文本")

@tool
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    config: RunnableConfig
) -> str:
    """通过将旧文本替换为新文本来编辑指定文件。旧文本必须在文件中完全匹配"""
    try:
        file_path = resolve_path(path, config)
        if not file_path.exists():
            return f"Error: 未找到文件：{path}"

        content = file_path.read_text(encoding="utf-8")

        if old_text not in content:
            return _not_found_message(old_text, content, path)

        count = content.count(old_text)
        if count > 1:
            return "Wanning: "

        new_content = content.replace(old_text, new_text, 1)
        file_path.write_text(new_content, encoding="utf-8")

        return f"成功编辑{path}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error editing file: {e}"
    
def _not_found_message(path: str, old_text: str, content: str) -> str:
    """当未找到旧文本时，构建一个有帮助的错误信息"""
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)

    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i

    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines, lines[best_start : best_start + window],
            fromfile="old_text (provided)", tofile=f"{path} (actual, line {best_start + 1})",
            lineterm="",
        ))
        return f"Error：在 {path} 中未找到旧文本。\n最佳匹配（相似度{best_ratio:.0%}位于第{best_start + 1}行：\n{diff}"
    return f"Error：在 {path} 中未找到旧文本，且未发现相似内容。请检查文件内容。"


filesystem_tools = [list_dir, read_file, write_file, edit_file]
