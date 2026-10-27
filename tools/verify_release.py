"""Verify this frozen release without loading models or importing the runtime."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT):
            errors.append(f"Invalid manifest path: {row['path']}")
            continue
        if not path.is_file():
            errors.append(f"Missing: {row['path']}")
            continue
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            errors.append(f"Changed: {row['path']}")
    if errors:
        print('\n'.join(errors))
        return 1
    print(f"Verified {len(manifest['files'])} release files. Model/environment data are external.")
    print('This checks file integrity, not the scientific correctness of the analyses.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
