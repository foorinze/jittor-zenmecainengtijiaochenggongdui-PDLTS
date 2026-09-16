"""检查验证报告并生成文件清单，可选生成压缩包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from release_inventory import EXCLUDED_FILES, MANIFEST_PATH, STATIC_REPORT_PATH, release_files


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/release_file_manifest.json"
REQUIRED_REPORTS = [
    "validation/reports/release_static_validation.json",
    "validation/reports/release_evaluator_validation.json",
    "validation/reports/release_evaluator_pcu_validation.json",
    "validation/reports/release_entrypoint_validation.json",
    "validation/reports/release_environment_validation.json",
    "validation/reports/release_model_validation.json",
]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.version):
        parser.error("version must contain lowercase ASCII letters, digits, underscores or hyphens")
    output = args.output_dir.resolve() if args.output_dir else None
    if not (args.manifest_only or args.verify_only) and output is None:
        parser.error("--output-dir is required when creating an archive")
    if output is not None and (output == ROOT or ROOT in output.parents):
        parser.error("output directory must be outside the public source tree")
    files = list(release_files(ROOT))
    expected_audit = {p.relative_to(ROOT).as_posix() for p in files}
    expected_audit -= {MANIFEST_PATH, STATIC_REPORT_PATH}
    for report in REQUIRED_REPORTS:
        data = json.loads((ROOT / report).read_text(encoding="utf-8"))
        if not data.get("passed"):
            raise SystemExit("Release validation failed: " + report)
        if report == STATIC_REPORT_PATH:
            if {entry["path"] for entry in data["audited_files"]} != expected_audit:
                raise SystemExit("Release file set changed after static validation")
            if data.get("excluded_files") != sorted(EXCLUDED_FILES):
                raise SystemExit("Release exclusions changed after static validation")
        for checked in data.get("audited_files", []) + data.get("sources", []):
            if digest((ROOT / checked["path"]).read_bytes()) != checked["sha256"]:
                raise SystemExit("File changed after validation: " + checked["path"])
        if "evaluator" in report:
            for board in data["boards"]:
                if digest((ROOT / board["evaluator"]).read_bytes()) != board["sha256"]:
                    raise SystemExit("Evaluator changed after validation: " + board["evaluator"])
    records = []
    contents = {}
    for path in files:
        relative = path.relative_to(ROOT)
        if path == MANIFEST:
            continue
        body = path.read_bytes()
        records.append({"path": relative.as_posix(), "bytes": len(body), "sha256": digest(body)})
        contents[relative.as_posix()] = body
    manifest = {"schema_version": 1, "version": args.version, "status": "validated_source",
                "self_excluded": MANIFEST_PATH, "excluded_files": sorted(EXCLUDED_FILES), "files": records}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.verify_only:
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit("Current files differ from frozen release manifest")
        print("PASS: frozen file manifest matches", len(records), "files")
        return 0
    if args.manifest_only:
        MANIFEST.write_bytes(manifest_bytes)
        print("PASS: file manifest updated;", len(records), "files; no archive created")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (args.version + ".zip")
    if archive.exists():
        raise SystemExit("Archive already exists; verify it or choose a new version")
    MANIFEST.write_bytes(manifest_bytes)
    contents["manifests/release_file_manifest.json"] = manifest_bytes
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, body in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, body)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != len(contents):
            raise SystemExit("Archive integrity validation failed")
        for name, body in contents.items():
            if bundle.read(name) != body:
                raise SystemExit("Archive differs from source: " + name)
    result = {"version": args.version, "archive": archive.name, "files": len(contents),
              "bytes": archive.stat().st_size, "sha256": digest(archive.read_bytes()),
              "manifest_sha256": digest(manifest_bytes), "published": False}
    (output / (args.version + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
