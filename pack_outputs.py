import argparse
import zipfile
from pathlib import Path


DEFAULT_ZIP_NAME = "outputs.zip"


def find_project_root():
    return Path(__file__).resolve().parent


def collect_result_dirs(root):
    return sorted(
        path
        for path in root.glob("prog*/results")
        if path.is_dir()
    )


def pack_outputs(root, zip_path):
    root = Path(root).resolve()
    zip_path = Path(zip_path).resolve()
    result_dirs = collect_result_dirs(root)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result_dir in result_dirs:
            for file_path in sorted(result_dir.rglob("*")):
                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(root)
                archive_path = Path("outputs") / relative_path
                zf.write(file_path, arcname=archive_path.as_posix())
                file_count += 1

    print(f"Packed {file_count} file(s) from {len(result_dirs)} result folder(s).")
    print(f"Created: {zip_path}")
    return zip_path


def parse_args():
    root = find_project_root()
    parser = argparse.ArgumentParser(
        description="Pack all prog*/results folders into outputs.zip."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=root,
        help="Project root directory. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=root / DEFAULT_ZIP_NAME,
        help="Output zip path. Defaults to outputs.zip in the project root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack_outputs(args.root, args.zip)
