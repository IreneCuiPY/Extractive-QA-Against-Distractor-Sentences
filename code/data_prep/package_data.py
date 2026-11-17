"""Package the complete prepared data; verify the archive by reading it back."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
TARGET = ROOT / 'milestones' / 'm1' / 'data.tar.gz'
# The M1 archive contents. data/README.md and data/raw/ are not packaged.
MEMBERS = ['data.md', 'source_manifest.json', 'split_manifest.json', 'validation.json',
           'train/clean.jsonl', 'dev/clean.jsonl', 'test/clean.jsonl',
           'test/addsent.jsonl', 'test/addonesent.jsonl']

def main():
    missing = [name for name in MEMBERS if not (DATA/name).is_file()]
    if missing:
        # Without this check, a fresh clone would overwrite data.tar.gz without the training split.
        raise FileNotFoundError(f'Missing prepared files {missing}. Extract milestones/m1/data.tar.gz '
                                'or run code/data_prep/prepare_data.py first.')
    files = sorted(DATA/name for name in MEMBERS)
    target = TARGET
    # Fixed metadata gives byte-identical archives for unchanged inputs.
    with target.open('wb') as output:
        with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for path in files:
                    info = tar.gettarinfo(str(path), arcname='data/'+str(path.relative_to(DATA)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mode = 0o644
                    info.mtime = 0
                    with path.open('rb') as source:
                        tar.addfile(info,source)
    expected = {'data/'+str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(target,'r:gz') as tar:
        assert set(tar.getnames()) == set(expected)
        for member in tar.getmembers():
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[member.name]
    result = {'archive':'data.tar.gz','bytes':target.stat().st_size,
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'members':len(files),'all_member_hashes_verified':True}
    (ROOT/'references/package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
