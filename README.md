# Sign-Language-To-Speech

## Prepare data

```
python load_dataset.py
```

Re-run this if the job dies due to rate limit errors

```
python prepare_dataset.py
```

## Run train job

```
python train.py
```

or if using SLURM

```
sbatch train_job.sh
```

## Inference

```
python inference.py
```
