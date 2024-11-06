task=$1
base_lrs=$2


model=skt/kogpt2-base-v2

remain_lang=ko_punc
data_remove=false
do_hangeulize=false

for seed in 1 2 3
do
    if [[ $task==KorSTS ]];
    then
        remain_lang=ko_en_punc
        data_remove=true
        do_hangeulize=false
    fi

    ## LoRA ##
    python nlu_tasks/scripts/run_finetuning_lrs2.py mode=nlu_ft task=$task \
        seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
        optim.grad_acc=1 optim.base_lrs=$base_lrs data.max_length=256 logging.log_steps=10000 \
        model.hf_model=True model.name=skt/kogpt2-base-v2 \
        model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        model.set_kombo=false model.kombo.do_combination=false model.kombo.add_lora=false

    ## KOMBO ##
    python nlu_tasks/scripts/run_finetuning_lrs2.py mode=nlu_ft task=$task \
        seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
        optim.grad_acc=1 optim.base_lrs=$base_lrs data.max_length=256 logging.log_steps=10000 \
        model.hf_model=True model.name=skt/kogpt2-base-v2 \
        model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        model.set_kombo=true model.kombo.do_combination=true model.kombo.combination.combination_type=gru \
        model.kombo.tok_type=jamo_var model.kombo.kombo_max_length=2048 model.kombo.add_lora=false
done
