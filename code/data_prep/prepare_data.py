"""Build article-disjoint QA splits using only the downloaded, pinned sources.

No model is trained or scored. Test access is limited to structural validation.
Run: python3 code/data_prep/prepare_data.py from the repository root.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
SEED = '5300-m1-2026'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_source(name):
    path = DATA / 'raw' / name
    manifest = json.loads((ROOT / 'references/source_manifest.json').read_text())
    # Match by file name: the M1 manifest records the old source_data/ location.
    expected = next(x['sha256'] for x in manifest if Path(x['file']).name == name)
    assert digest(path) == expected, f'Source changed: {name}'
    return json.loads(path.read_text(encoding='utf-8'))

def flatten(source, variant):
    rows = []
    for article in source['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qid = qa['id']
                rows.append(dict(id=qid, original_id=qid.split('-')[0],
                                 title=article['title'], context=paragraph['context'],
                                 question=qa['question'], answers=qa['answers'],
                                 variant=variant if '-' in qid else 'clean',
                                 source='squad_v1.1' if variant == 'clean' else 'jia_liang_2017'))
    return rows

def validate_rows(rows):
    assert len({r['id'] for r in rows}) == len(rows)
    for row in rows:
        assert row['answers'] and row['context'] and row['question']
        for answer in row['answers']:
            start, text = answer['answer_start'], answer['text']
            assert isinstance(start, int) and start >= 0 and text
            assert row['context'][start:start+len(text)] == text, row['id']

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def write_rows(name, rows):
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    return {'file': name, 'rows': len(rows),
            'unique_original_questions': len({r['original_id'] for r in rows}),
            'articles': len({r['title'] for r in rows}),
            'bytes': path.stat().st_size, 'sha256': digest(path)}

def main():
    original_train = flatten(read_source('train-v1.1.json'), 'clean')
    clean_test = flatten(read_source('dev-v1.1.json'), 'clean')
    titles = sorted({r['title'] for r in original_train},
                    key=lambda t: (hashlib.sha256((SEED+'\n'+t).encode()).hexdigest(), t))
    # About 10% of source training articles become our development partition.
    n_dev = round(0.10 * len(titles))
    dev_titles = set(titles[:n_dev])
    train = [r for r in original_train if r['title'] not in dev_titles]
    dev = [r for r in original_train if r['title'] in dev_titles]
    assert len(train)+len(dev) == len(original_train) == 87599
    assert len(clean_test) == 10570
    clean_by_id = {r['id']: r for r in clean_test}
    attack_rows, cohort_sets, attacked_sets = {}, {}, {}
    source_counts = {}
    for variant in ('addsent', 'addonesent'):
        raw = flatten(read_source(variant+'.json'), variant)
        validate_rows(raw)
        cohort_sets[variant] = {r['original_id'] for r in raw}
        attack_rows[variant] = [r for r in raw if r['variant'] != 'clean']
        attacked_sets[variant] = {r['original_id'] for r in attack_rows[variant]}
        source_counts[variant] = {'raw_rows':len(raw), 'clean_rows':sum(r['variant']=='clean' for r in raw)}
        for row in raw:
            original = clean_by_id[row['original_id']]
            assert row['title'] == original['title']
            assert row['question'] == original['question']
            assert row['answers'] == original['answers']
            if row['variant'] == 'clean':
                assert row['context'] == original['context']
            else:
                assert row['context'].startswith(original['context']+' ')
                assert len(row['context']) > len(original['context'])+1
    assert cohort_sets['addsent'] == cohort_sets['addonesent']
    assert attacked_sets['addsent'] == attacked_sets['addonesent']
    assert len(cohort_sets['addsent']) == 1000
    assert len(attacked_sets['addsent']) == 787
    sent_by_id = {r['id']:r for r in attack_rows['addsent']}
    for r in attack_rows['addonesent']:
        assert r['context'] == sent_by_id[r['id']]['context']

    split_rows = {'train':train, 'dev':dev,
                  'test':clean_test+attack_rows['addsent']+attack_rows['addonesent']}
    overlaps = {}
    for left, right in [('train','dev'),('train','test'),('dev','test')]:
        overlaps[left+'__'+right] = {}
        for field in ('title','original_id','context'):
            def values(rows):
                if field == 'context':
                    return {' '.join(r[field].split()) for r in rows}
                return {r[field] for r in rows}
            count = len(values(split_rows[left]) & values(split_rows[right]))
            overlaps[left+'__'+right][field] = count
            assert count == 0, (left,right,field,count)

    files = []
    for name, rows in [('train/clean.jsonl',train),('dev/clean.jsonl',dev),
                       ('test/clean.jsonl',clean_test),
                       ('test/addsent.jsonl',attack_rows['addsent']),
                       ('test/addonesent.jsonl',attack_rows['addonesent'])]:
        validate_rows(rows)
        files.append(write_rows(name,rows))
    split_manifest = {'seed':SEED,
        'algorithm':'Sort source training article titles by SHA256(seed + newline + title); first 44 titles are dev; rest train.',
        'train_articles':sorted({r['title'] for r in train}),
        'dev_articles':sorted(dev_titles),
        'test_articles':sorted({r['title'] for r in clean_test}),
        'adversarial_cohort_ids':sorted(cohort_sets['addsent']),
        'adversarial_attack_available_ids':sorted(attacked_sets['addsent']),
        'adversarial_no_attack_ids':sorted(cohort_sets['addsent']-attacked_sets['addsent'])}
    write_json(DATA/'split_manifest.json',split_manifest)
    write_json(DATA/'source_manifest.json',json.loads((ROOT/'references/source_manifest.json').read_text()))
    validation = {'files':files,'upstream_adversarial_counts':source_counts,
                  'cross_split_overlap_counts':overlaps,
                  'all_answer_offsets_valid':True,
                  'all_adversarial_questions_and_answers_match_clean':True,
                  'all_adversarial_contexts_preserve_original_as_prefix':True,
                  'addonesent_is_subset_of_addsent_candidates':True,
                  'no_training_or_inference_performed':True}
    write_json(DATA/'validation.json',validation)
    # Actual training example for documentation, selected by ID for reproducibility.
    example = min(train, key=lambda r:(len(r['context']),r['id']))
    write_json(ROOT/'references/training_example.json',example)
    print(json.dumps(validation,indent=2))

if __name__ == '__main__':
    main()
