# MCore LA0 / LA1: 300-step comparison

Both cases completed all 300 steps with checkpoint 300 and successful Slurm exits.
No missing metric steps, nonfinite KL/loss/reward/gradient metrics, sample-count
violations, or lag violations were found. After warmup LR stays 1e-6, with
32 valid responses per update and lag 0 / 0–1 respectively.

| Last 100 updates | LA0 | LA1 |
|---|---:|---:|
| Mean gen_kl_error | 0.000936859 | 0.001064585 |
| Mean reward | 0.61906 | 0.59969 |
| Mean response tokens | 1142.68 | 1179.65 |
| Mean step time, excluding first update after resume | 131.55 s | 68.45 s |

LA1 mean gen KL is about 13.6% higher over
updates 201–300. Neither run shows runaway KL; LA1 global maximum is 0.001704.
LA1's late reward is close to LA0's (0.600 vs 0.619); one seed with eight
prompt groups per update cannot establish statistical equivalence.
The step-time ratio is 1.92x, using the same update interval and excluding
resume-cold updates. It includes normal checkpoint overhead and excludes
allocation startup and queue waiting. This is observed elapsed step time,
not a measurement attributing every gain solely to GPU overlap.

The paired configurations use the same model, batch, length, precision and
parallel layout. Both use logprob_chunk_size=512 from checkpoint 10 onward
and CPU staging of portable optimizer checkpoint state after the failed
initial attempts. Original logs remain preserved; retries are deduplicated
by update number with the latest attempt winning. See README.md for the
OOM fixes and backend-equivalence limitations.

Online validation remains disabled as requested; the validation accuracy
panel has no data. Reward above is a training metric.

Final local plot: results/comparison.png (both MCore cases through step 300,
plus historical AutoModel references). Reproducible scalar checks and
window summaries: completion-check.json; source logs: slurm/submitted.tsv.
The full result artifacts and checkpoints remain local rather than in Git.

The unused LA0 continuation 21326850 was cancelled. LA1 continuations
21326853/54 were already cancelled. Old AutoModel 900-step jobs
21315214/16/17 remain held. The stop-10min-watch marker disables future
conversation wakeups. Removal of the crontab block could not be verified: a
prior automatic safety review rejected crontab access as a persistence action;
no alternate access path was used.
