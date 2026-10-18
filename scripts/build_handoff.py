"""Package the offline analysis source for a teammate; no caches or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(output):
    root = Path(__file__).resolve().parents[1]
    files = [root / 'README.md', root / 'pyproject.toml']
    for folder, pattern in [('floaterfinder', '*.py'), ('docs', '*.md'),
                            ('tests', '*.py'), ('examples', '*.json'),
                            ('examples', '*.py'), ('scripts', '*.py'), ('configs', '*.json')]:
        files.extend(sorted((root / folder).glob(pattern)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, 'FloaterFinder/' + str(path.relative_to(root)))
        archive.writestr('FloaterFinder/SOURCE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(build(parser.parse_args().output))
