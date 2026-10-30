"""Prepare public confirmation source identity and duplicate ledgers, never models.

Usage: .venv/bin/python experiments/v02/data/prepare_sources.py
Requires the two official archives at paths below; archives are disposable caches.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import csv, hashlib, io, json, sys
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p): return sha(p.read_bytes())
def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
def pixel_hash(row): return sha(np.asarray(row, dtype=np.uint8).tobytes())

archives = {'optical': Path('/private/tmp/modelchord-v02-optical.zip'), 'youtube': Path('/private/tmp/modelchord-v02-youtube.zip')}
raw = OUT / 'raw'; raw.mkdir(exist_ok=True)
with ZipFile(archives['optical']) as z:
    for name in ['optdigits.tra', 'optdigits.tes', 'optdigits.names', 'readme.txt']:
        (raw/name).write_bytes(z.read(name))
tra = np.loadtxt(raw/'optdigits.tra', delimiter=',', dtype=np.int64)
tes = np.loadtxt(raw/'optdigits.tes', delimiter=',', dtype=np.int64)
old = load_digits()
assert tra.shape == (3823,65) and tes.shape == (1797,65)
assert np.array_equal(tes[:,:64], old.data) and np.array_equal(tes[:,64],old.target)
assert np.all((tra[:,:64]>=0)&(tra[:,:64]<=16)) and np.all((tra[:,64]>=0)&(tra[:,64]<=9))
old_hashes = {pixel_hash(x) for x in old.data}
seen = {}; eligible = []; excluded=[]; records=[]
for i,row in enumerate(tra):
    h=pixel_hash(row[:64]); reason=None
    if h in old_hashes: reason='pixel_hash_matches_all_1797_old_source'
    elif h in seen: reason='within_confirmation_exact_pixel_duplicate'
    if h in seen and int(row[-1])!=seen[h][1]: raise ValueError('conflicting duplicate digit labels')
    record={'row_index_zero_based':i,'pixel_sha256_uint8_rowmajor':h,'label':int(row[-1]),'eligible':reason is None}
    if reason: record['exclusion']=reason;excluded.append(record)
    else: eligible.append(i)
    seen.setdefault(h,(i,int(row[-1])));records.append(record)
write('vision-confirmation-index.json',{'dataset':'UCI Optical Recognition of Handwritten Digits original TRAIN file used only as NEW confirmation','schema_version':1,'raw_file':'raw/optdigits.tra','rows':len(tra),'eligible_row_indices':eligible,'excluded':excluded,'records':records,'hash_representation':'SHA256 of exactly 64 uint8 pixel counts row-major, excludes label; equivalent original representations compared before normalization','old_source_all_1797_equality_to_official_test':True,'old_source_all_1797_unique_pixel_hashes':len(old_hashes),'counts':{'eligible':len(eligible),'excluded':len(excluded),'eligible_class_counts':dict(sorted(Counter(int(tra[i,-1]) for i in eligible).items()))},'model_inference_performed':False,'split_status':'entire eligible cohort reserved confirmation; never train/tune on it'})
# Text: compare only hash IDs from v1 ledger; raw SMS content is not opened.
old_text_ledger=json.loads((ROOT/'study/text-split.json').read_text())
old_text_hashes={h for ids in old_text_ledger['ids'].values() for h in ids}
text_records=[]; groups=defaultdict(list); per_file={}; raw_files=[]
with ZipFile(archives['youtube']) as z:
    for name in sorted(n for n in z.namelist() if n.startswith('Youtube') and n.endswith('.csv')):
        b=z.read(name); (raw/name).write_bytes(b); raw_files.append(name)
        rows=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
        counts=Counter(int(r['CLASS']) for r in rows)
        per_file[name]={'rows':len(rows),'class_counts':dict(sorted(counts.items()))}
        for i,row in enumerate(rows):
            assert row['CLASS'] in ('0','1') and row['CONTENT']
            h=sha(row['CONTENT'].encode('utf-8'))
            record={'source_file':name,'row_index_zero_based':i,'text_sha256_utf8':h,'label':int(row['CLASS'])}
            groups[h].append(record);text_records.append(record)
text_eligible=[]; excluded_text=[]
for h,rs in groups.items():
    labels={r['label'] for r in rs}
    for i,r in enumerate(rs):
        reason='conflicting_exact_text_labels' if len(labels)>1 else ('text_hash_matches_v1_balanced_1306' if h in old_text_hashes else ('within_transfer_exact_text_duplicate' if i else None))
        r['eligible']=reason is None
        if reason: r['exclusion']=reason;excluded_text.append(r)
        else: text_eligible.append(r)
write('text-transfer-index.json',{'dataset':'UCI YouTube Spam Collection','schema_version':1,'role':'optional new distribution-shifted transfer confirmation; NOT new SMS confirmation','labels':{'0':'ham','1':'spam'},'raw_files':raw_files,'source_rows':len(text_records),'per_file':per_file,'eligible_records':text_eligible,'excluded_records':excluded_text,'counts':{'eligible':len(text_eligible),'excluded':len(excluded_text),'eligible_class_counts':dict(sorted(Counter(r['label'] for r in text_eligible).items()))},'deduplication':'Exact original Unicode text UTF-8 SHA256; drop all rows of conflicting-label groups; keep first sorted-file/row occurrence otherwise','old_source_comparison':'All 1306 balanced SMS source hashes from study/text-split.json, all partitions; not the remaining unused ham messages; raw SMS not opened','old_source_ledger_sha256':file_sha(ROOT/'study/text-split.json'),'old_source_hash_count':len(old_text_hashes),'model_inference_performed':False,'split_status':'not split; no model metrics; freeze use/protocol before any inference'})
manifest={'prepared_utc':'2026-10-29','model_inference_performed':False,'downloads':{},'files':{},'sources':{'optical':{'landing_url':'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','doi':'10.24432/C50P49','attribution':'Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository.','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','license_evidence':'Official UCI landing page License section checked 2026-10-29','writer_split_evidence':'Official UCI landing page Additional Information and raw/optdigits.names section 4: original TRAIN 30 writers, original TEST different 13 writers; individual writer IDs unavailable'},'youtube':{'landing_url':'https://archive.ics.uci.edu/dataset/380/youtube+spam+collection','doi':'10.24432/C58885','attribution':'Alberto, T. and Lochter, J. (2015). YouTube Spam Collection. UCI Machine Learning Repository.','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','license_evidence':'Official UCI landing page License section checked 2026-10-29','author_laboratory_url':'https://lasid.dcomp.ufscar.br/youtube-spam-collection/'}}}
for key,p in archives.items():
    manifest['downloads'][key]={'url':{'optical':'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip','youtube':'https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip'}[key],'archive_sha256':file_sha(p),'archive_bytes':p.stat().st_size,'retained_archive':False}
for p in sorted(raw.iterdir()): manifest['files'][str(p.relative_to(OUT))]={'sha256':file_sha(p),'bytes':p.stat().st_size}
for name in ['vision-confirmation-index.json','text-transfer-index.json','prepare_sources.py']:
    p=OUT/name;manifest['files'][name]={'sha256':file_sha(p),'bytes':p.stat().st_size}
write('source-manifest.json',manifest)
print(json.dumps({'vision':{'eligible':len(eligible),'excluded':len(excluded)},'text_transfer':{'eligible':len(text_eligible),'excluded':len(excluded_text)},'archives':manifest['downloads']},indent=2))
