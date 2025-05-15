cd ../
device=0
dataset=SCEDC
model=SMASH
seq_len=150
catalog_path=../../Datasets/SCEDC/SCEDC_catalog.csv
Mcut=2.0
auxiliary_start=1981-01-01:00:00:00
train_nll_start=1985-01-01:00:00:00
val_nll_start=2005-01-01:00:00:00
test_nll_start=2014-01-01:00:00:00
test_nll_end=2020-01-01:00:00:00
marked_output=1
sigma_time=0.2
sigma_loc=0.25
samplingsteps=2000
langevin_step=0.005
log_normalization=1
cond_dim=16
seed=2
loss_lambda=0.5

save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}_seq_len_${seq_len}_marked_output_${marked_output}/

python app.py --day_number $1 --batch_size $2 --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --save_path ${save_path} --seed ${seed} --dim 3 --log_normalization ${log_normalization} --dataset ${dataset} --mode sample --model ${model} --langevin_step ${langevin_step} --samplingsteps ${samplingsteps} --n_samples 1 --per_step 250  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output} --weight_path ${save_path}model_best.pkl


