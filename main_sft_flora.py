import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, LoraConfig, PeftModel
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import torch_npu
import logging
from time import sleep

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate,script_args.max_steps)
save_config(script_args, fed_args)
print(script_args, fed_args)
logging.basicConfig(
    filename=os.path.join(script_args.output_dir, 'fed_local_sft.log'),
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
logger.debug("Test")

# ===== Load the dataset =====
if fed_args.fed_alg.startswith('local'):
    dataset = get_local_datasets(script_args.local_data_dir)
else:
    dataset = get_fed_datasets(script_args.local_data_dir)


# ===== FLoRA parameters =====
if fed_args.fed_alg == 'flora':
    stacking = True
    heter = True # False True
else:
    stacking = False
    heter = False

if heter:
    rank_8_clients = [7,17,19,29,30,31]
    rank_16_clients = [8,24,27]
    local_ranks = [8 if i in rank_8_clients else 16 if i in rank_16_clients else 4 for i in range(fed_args.num_clients)]
else:
    local_ranks = [script_args.peft_lora_r] * fed_args.num_clients

# local_ranks = [8] * fed_args.num_clients # [64, 32, 16, 16, 8, 8, 4, 4, 4, 4]
lora_target_modules = ["q_proj", "v_proj"]

# ===== Split the dataset into clients =====
print("===== Split the dataset into clients =====")
local_datasets = dataset[:fed_args.num_clients]
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(sample_num_list)
print(len(sample_num_list))

# ===== Get model config =====
print("===== Get model config =====")
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
model.enable_input_require_grads()
if stacking == False:
    config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=script_args.model_name_or_path,
    )
    global_model = get_peft_model(model, config)
    global_dict = copy.deepcopy(get_peft_model_state_dict(global_model))
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
else:
    global_dict = None
    proxy_dict, opt_proxy_dict = None, None
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = None, [None] * fed_args.num_clients, None
    
ddp = False
if not ddp and torch_npu.npu.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

# ===== Define the global and local models =====
print("===== Define the global and local models =====")
local_dict_list = [None for i in range(fed_args.num_clients)]

# ===== Define the tokenizer =====
print("===== Define the tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", model_max_length=script_args.seq_length)
if script_args.multi_turn_task:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    model.resize_token_embeddings(len(tokenizer))
else:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token, script_args)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

if script_args.multi_turn_task:
    data_collator_list, sample_num_list = get_multi_turn_dataset(fed_args.fed_alg, script_args.local_data_dir, tokenizer)
    print("multi-turn")
    print(len(sample_num_list))
    print(sample_num_list)
   
# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")

    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        print(f"========Client {client} Training Started=========")
        if stacking:
            config = LoraConfig(
                r=local_ranks[client],
                lora_alpha=2*local_ranks[client],
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                base_model_name_or_path=script_args.model_name_or_path,
            )
            if isinstance(model, PeftModel):
                model = model.base_model
            model_client = get_peft_model(model, config)
        else:
            set_peft_model_state_dict(global_model, global_dict)
            model_client = global_model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr,script_args.max_steps)

        # ===== Train local model on the client side =====
        if script_args.multi_turn_task:
            client_data_collator = data_collator_list[client]
            print('client',client,'data number',len(client_data_collator))
            trainer = get_fed_local_sft_trainer(
                model=model_client,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=client_data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
        else:
            trainer = get_fed_local_sft_trainer(
                model=model_client,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )

        results = trainer.train()
        print("========Training finished=========")
        training_loss[client].append(results.training_loss)
        logger.debug(f"Round: {round}, Client: {client}, FIM Trace: {trainer.fim_trace}")

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model_client))   # copy is needed!
        del model_client

    if stacking == True:
        config = LoraConfig(
            r=sum([local_ranks[i] for i in clients_this_round]),
            lora_alpha=script_args.peft_lora_alpha*fed_args.num_clients,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            base_model_name_or_path=script_args.model_name_or_path,
        )
        global_model = get_peft_model(model, config)
        global_dict = copy.deepcopy(get_peft_model_state_dict(global_model))
        proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
        global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, script_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict), \
        stacking=stacking, heter=heter, local_ranks=local_ranks
    )
    set_peft_model_state_dict(global_model, global_dict)
    if stacking:
        model = global_model.merge_and_unload()
    else:
        model = copy.deepcopy(global_model)
        model = model.merge_and_unload().base_model
        

    # ===== Save the model =====
    if (round+1) % fed_args.checkpoint_step == 0:
        save_path = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        # if stacking:
        #     model.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        # else:
        #     trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))