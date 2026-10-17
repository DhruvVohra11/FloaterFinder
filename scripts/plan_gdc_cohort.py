"""Fetch public GDC file metadata only; never downloads slides or calls models."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request

PROJECTS = {'TCGA-PRAD':'prostate', 'TCGA-COAD':'colon', 'TCGA-LUAD':'lung'}


def fetch(project):
    filters = {'op':'and','content':[
        {'op':'in','content':{'field':'cases.project.project_id','value':[project]}},
        {'op':'in','content':{'field':'data_type','value':['Slide Image']}},
        {'op':'in','content':{'field':'experimental_strategy','value':['Diagnostic Slide']}},
        {'op':'in','content':{'field':'access','value':['open']}}]}
    fields = 'file_id,file_name,file_size,md5sum,state,access,cases.case_id,cases.submitter_id,cases.project.project_id'
    query = urllib.parse.urlencode({'filters':json.dumps(filters),'fields':fields,
                                   'size':200,'sort':'file_id:asc','format':'JSON'})
    with urllib.request.urlopen('https://api.gdc.cancer.gov/files?' + query, timeout=30) as r:
        return json.load(r)['data']


def select_cases(hits, count, seed):
    hits = list(hits)
    random.Random(seed).shuffle(hits)
    selected, seen = [], set()
    for hit in hits:
        cases = hit.get('cases', [])
        if len(cases) != 1 or not cases[0].get('case_id') or hit.get('access') != 'open':
            continue
        patient = cases[0]['case_id']
        if patient not in seen:
            selected.append(hit)
            seen.add(patient)
        if len(selected) == count:
            return selected
    raise ValueError(f'Only {len(selected)} unique eligible patients in metadata page; requested {count}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--per-project', type=int, default=6)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    if not 6 <= args.per_project <= 100:
        p.error('Use 6–100 slides per project for this pilot planner')
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Use a new output directory')
    # Complete read-only queries before creating the output folder.
    results = {project: fetch(project) for project in PROJECTS}
    rows, downloads = [], []
    for project, response in results.items():
        selected = select_cases(response['hits'],args.per_project,args.seed)
        # At least two distinct patients per split and organ, permitting same-organ donors.
        for i, hit in enumerate(selected):
            case = hit['cases'][0]
            split = ['development','calibration','test'][i % 3]
            rows.append({'file_id':hit['file_id'],'file_name':hit['file_name'],
                'patient_id':case['case_id'],'patient_submitter_id':case.get('submitter_id',''),
                'organ_from_project':PROJECTS[project], 'project':project, 'split':split,
                'file_size_bytes':hit['file_size'],'md5':hit['md5sum'],
                'review_status':'unreviewed'})
            downloads.append({'id':hit['file_id'],'filename':hit['file_name'],
                              'md5':hit['md5sum'],'size':hit['file_size'],'state':hit['state']})
    root.mkdir(parents=True)
    for filename, values in [('cohort.csv',rows),('gdc-manifest.tsv',downloads)]:
        with (root/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0]),delimiter='\t' if filename.endswith('.tsv') else ',')
            writer.writeheader()
            writer.writerows(values)
    info = {'created_at':datetime.now(timezone.utc).isoformat(),'slides':len(rows),
        'distinct_patients':len({r['patient_id'] for r in rows}),
        'estimated_download_bytes':sum(r['file_size_bytes'] for r in rows),
        'selection':'Seeded convenience sample from first 200 file-ID-sorted eligible files per project; one file per case',
        'limitations':['Metadata only; no slide images downloaded or visually reviewed.',
                      'Project organ is provisional specimen context, not a tile annotation or disease label.',
                      'Review physical scale, tissue quality, specimen identity and normal/rare regions before use.']}
    (root/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    (root/'metadata.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__ == '__main__':
    main()
