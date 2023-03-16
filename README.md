First, create directories with the names "checkpoints", "data", and "comb3_cap_data", and download their data from the following Google Drive link:
https://drive.google.com/drive/folders/15YDeik6gIQrL3A1ow--9a86QTjZ513gR

Then, use `sh ./run.sh` to run all experiments. Results will be dumped in `comb*.log`. The experiments include trial 0 to 9. Trial 10 to 19 are used for validation. The best hyperparameters seached on validation set are hard-coded in `comb*_experiment.py`. Only datasets with $n=150, 750, 7500$ are included due to storage limitation of Google drive.
