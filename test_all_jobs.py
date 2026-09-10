import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from candidate_ranker.run_ranking import rank_candidates
from candidate_ranker.config import DEFAULT_SCORING_WEIGHTS, DEFAULT_TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def test_all_jobs(jobs_path='job_postings.json', top_k=5, skip_llm=True):
    if not os.path.exists(jobs_path):
        print(f"Hata: {jobs_path} bulunamadi.")
        return

    with open(jobs_path, 'r', encoding='utf-8') as f:
        jobs = json.load(f)

    print('=' * 70)
    print(f'Toplam {len(jobs)} is ilani test ediliyor...')
    print('=' * 70)

    summary = []

    for job in jobs:
        jid = job.get('id')
        title = job.get('title')
        lang = job.get('language')
        jd_text = job.get('raw_text') or job.get('description', '')

        print(f'\n>>> [{jid}/11] {title} ({lang})')

        try:
            res = rank_candidates(
                jd_text=jd_text,
                top_k=top_k,
                weights=DEFAULT_SCORING_WEIGHTS,
                json_only=True,
                skip_llm=skip_llm,
            )

            top_candidates = res.get('top_candidates', [])
            print(f'    Eslesen Aday Sayisi: {len(top_candidates)}')
            for c in top_candidates[:3]:
                cname = c.get('candidate_name') or c.get('candidate_id')
                cscore = c.get('final_score', 0)
                rank = c.get('rank')
                print(f'     * Rank {rank}: {cname} (Skor: {cscore})')

            summary.append({
                'job_id': jid,
                'job_title': title,
                'language': lang,
                'total_matched': len(top_candidates),
                'top_3': [
                    {
                        'rank': c.get('rank'),
                        'name': c.get('candidate_name'),
                        'score': c.get('final_score')
                    }
                    for c in top_candidates[:3]
                ]
            })
        except Exception as e:
            print(f'    [HATA]: {e}')

    os.makedirs('ranking_outputs', exist_ok=True)
    with open('ranking_outputs/all_jobs_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print('Tamamlandi! Ozet: ranking_outputs/all_jobs_summary.json')
    print('=' * 70)

if __name__ == '__main__':
    test_all_jobs()
