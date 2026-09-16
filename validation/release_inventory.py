"""发布文件边界；排除文件只留在本地，不删除或移动。"""

from pathlib import Path

EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv",
    ".idea", ".vscode", ".ruff_cache", ".mypy_cache", ".tmp_jt_cache",
    "outputs", "dataset", "tmp",
}
EXCLUDED_FILES = {
    "b_board/starter_code/tests/_dcd_oracle_numpy.py",
    "b_board/starter_code/tests/test_dcd_oracle_equivalence.py",
    "b_board/starter_code/tests/test_dcd_official_jittor.py",
    "b_board/starter_code/src/model/pdlts_light/losses/dcd_official.py",
    "a_board/starter_code/src/model/pdlts_light/losses/dcd_official.py",
}
MANIFEST_PATH = "manifests/release_file_manifest.json"
STATIC_REPORT_PATH = "validation/reports/release_static_validation.json"


def release_files(root):
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if EXCLUDED_DIRS.intersection(relative.parts) or relative.as_posix() in EXCLUDED_FILES:
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not accepted: " + relative.as_posix())
        if path.is_file():
            if path.suffix in {".pyc", ".pyo", ".zip", ".tar", ".gz", ".7z", ".log"} or path.name.startswith(".env"):
                raise ValueError("Unexpected generated or private artifact: " + relative.as_posix())
            yield path
