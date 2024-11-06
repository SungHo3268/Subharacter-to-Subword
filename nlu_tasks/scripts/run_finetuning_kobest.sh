task=$1

# task=KB_BoolQ KB_COPA KB_WiC KB_HellaSwag KB_SentiNeg
remain_lang=ko_punc
data_remove=false
do_hangeulize=false

for seed in 1
do
    for base_lr in 1e-02
    do
        # ## LoRA ##
        # python nlu_tasks/scripts/run_gpt_finetuning.py mode=nlu_ft task=$task \
        #     seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
        #     optim.grad_acc=1 optim.base_lr=$base_lr data.max_length=256 logging.log_steps=10000 \
        #     model.hf_model=True model.name=skt/kogpt2-base-v2 \
        #     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        #     model.set_sub2=false model.sub2.do_combination=false model.sub2.add_lora=false

        # ## sub2 ##
        # python nlu_tasks/scripts/run_gpt_finetuning.py mode=nlu_ft task=$task \
        #     seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
        #     optim.grad_acc=1 optim.base_lr=$base_lr data.max_length=256 logging.log_steps=10000 \
        #     model.hf_model=True model.name=skt/kogpt2-base-v2 \
        #     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        #     model.set_sub2=true model.sub2.do_combination=true model.sub2.combination.combination_type=gru \
        #     model.sub2.tok_type=jamo_var model.sub2.sub2_max_length=2048 model.sub2.add_lora=false

        ## Linear Pool ##
        python nlu_tasks/scripts/run_kobest_debugging.py mode=nlu_ft task=$task \
            seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
            optim.grad_acc=1 optim.base_lr=$base_lr data.max_length=256 \
            model.hf_model=True model.name=skt/kogpt2-base-v2 \
            model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
            model.set_sub2=true model.sub2.do_combination=false model.sub2.reducer=linear_pool \
            model.sub2.add_lora=true model.sub2.tok_type=jamo_var model.sub2.max_length=2048 \
            optim.warmup_ratio=0

        # ## Attention Pool ##
        # python nlu_tasks/scripts/run_gpt_finetuning.py mode=nlu_ft task=$task \
        #     seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
        #     optim.grad_acc=1 optim.base_lr=$base_lr data.max_length=256 \
        #     model.hf_model=True model.name=skt/kogpt2-base-v2 \
        #     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        #     model.set_sub2=true model.sub2.do_combination=false model.sub2.reducer=attention_pool \
        #     model.sub2.add_lora=true model.sub2.tok_type=jamo_var model.sub2.max_length=256
    done
done
