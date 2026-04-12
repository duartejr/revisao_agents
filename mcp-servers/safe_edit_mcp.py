from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Safe Edit MCP - Delivery Manager")

ALLOWED_WRITE_DIRS = ["roadmap", "reports"]


@mcp.tool
def safe_edit_file(file_path: str, content: str) -> str:
    """
    Edits a file safely.
    Only allows writing to 'roadmap/' and 'reports/' directories.
    Any other directory is immediately blocked.

    Args:
        file_path (str): The path to the file to be edited, relative to the project root.
        content (str): The new content to write into the file.

    Returns:
        str: A message indicating success or the reason for failure.
    """
    try:
        path = Path(file_path).resolve()
        project_root = Path.cwd().resolve()

        # Verify if the file is within the allowed directories
        is_allowed = any(
            path.is_relative_to(project_root / allowed_dir) for allowed_dir in ALLOWED_WRITE_DIRS
        )

        if not is_allowed:
            return f"🚫 SECURITY ERROR (Delivery Manager): This agent can only edit files within 'roadmap/' or 'reports/'. Attempt blocked: {file_path}"

        # Create directories if necessary and write the file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ File successfully edited: {file_path}"
    except Exception as e:
        return f"❌ Error editing file: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
