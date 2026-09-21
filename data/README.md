# Data Directory

Place local datasets here. Dataset files are intentionally ignored by git.

Raw downloads in `raw/` can be shared by both experiments. Prepared inputs must
be associated with the experiment that defines their preprocessing and splits.

`processed/glaucoma_rerun_v1/` was produced by the earlier reconstructed draft;
it is not the original GitHub fold data or verified IEEE-paper preprocessing.
The GitHub snapshot still expects its historical `iu_10fold` inputs, but the
separate `experiments/github_original/run.sh` adapter reads this reconstructed
fixed split directly. It does not fabricate historical fold files.
