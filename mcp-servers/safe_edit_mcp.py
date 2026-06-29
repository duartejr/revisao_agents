from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Safe Edit MCP")

PM_ALLOWED_WRITE_DIRS = ["management/roadmap", "management/reports"]
SOCIAL_ALLOWED_WRITE_DIRS = ["management/reports/social_assets"]


def _write_within(file_path: str, content: str, allowed_dirs: list[str], agent_label: str) -> str:
    try:
        path = Path(file_path).resolve()
        project_root = Path.cwd().resolve()

        is_allowed = any(
            path.is_relative_to(project_root / allowed_dir) for allowed_dir in allowed_dirs
        )

        if not is_allowed:
            allowed_desc = " or ".join(f"'{d}/'" for d in allowed_dirs)
            return f"🚫 SECURITY ERROR ({agent_label}): This agent can only edit files within {allowed_desc}. Attempt blocked: {file_path}"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ File successfully edited: {file_path}"
    except Exception as e:
        return f"❌ Error editing file: {str(e)}"


@mcp.tool
def safe_edit_file(file_path: str, content: str) -> str:
    """
    Edits a file safely.
    Only allows writing to 'management/roadmap/' and 'management/reports/' directories.
    Any other directory is immediately blocked.

    Args:
        file_path (str): The path to the file to be edited, relative to the project root.
        content (str): The new content to write into the file.

    Returns:
        str: A message indicating success or the reason for failure.
    """
    return _write_within(file_path, content, PM_ALLOWED_WRITE_DIRS, "Delivery Manager")


@mcp.tool
def safe_edit_social_asset_file(file_path: str, content: str) -> str:
    """
    Edits a file safely.
    Only allows writing to 'management/reports/social_assets/'.
    Any other directory is immediately blocked.

    Args:
        file_path (str): The path to the file to be edited, relative to the project root.
        content (str): The new content to write into the file.

    Returns:
        str: A message indicating success or the reason for failure.
    """
    return _write_within(file_path, content, SOCIAL_ALLOWED_WRITE_DIRS, "Social Media")


if __name__ == "__main__":
    mcp.run(transport="stdio")
