max_steps=10
num_rounds=200
checkpoint_step=50
batch_size=2
gradient_accumulation_steps=4
seq_length=1024
num_clients=38
sample_clients=4
lora_r=16
lora_alpha=32   # twice of lora_r
lr=2e-5

local_data_dir=data/Fed-Aya/aya_38c_25k.json   # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=FedAya
dataset_sample=25000
model_name_or_path="../Qwen2.5-0.5B-Instruct/"
output_dir=./models/FLora-FedAya-Qwen/

fed_alg=flora

python main_sft_flora.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --output_dir $output_dir \
 --template "alpaca" \
 --local_data_dir $local_data_dir \
 --gradient_checkpointing \
 --checkpoint_step $checkpoint_step
