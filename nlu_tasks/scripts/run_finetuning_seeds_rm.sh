lr=$1

for task in PAWS_X
do
    ## LoRA ##
    python nlu_tasks/scripts/run_finetuning_seeds.py mode=nlu_ft task=$task \
        seeds="1 2 3" data.remain_lang=ko_punc data.do_hangeulize=false data.data_remove=true \
        optim.grad_acc=1 optim.base_lr=$lr data.max_length=256 \
        model.hf_model=True model.name=skt/kogpt2-base-v2 \
        model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        model.set_kombo=false model.kombo.do_combination=false \
        model.kombo.add_lora=false

    ## KOMBO(+LoRA) ##
    python nlu_tasks/scripts/run_finetuning_seeds.py mode=nlu_ft task=$task \
        seeds="1 2 3" data.remain_lang=ko_punc data.do_hangeulize=false data.data_remove=true \
        optim.grad_acc=1 optim.base_lr=$lr data.max_length=256 \
        model.hf_model=True model.name=skt/kogpt2-base-v2 \
        model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
        model.set_kombo=true model.kombo.do_combination=true model.kombo.combination.combination_type=gru \
        model.kombo.add_lora=true model.kombo.lora.r=32 model.kombo.lora.alpha=128 model.kombo.kombo_max_length=2048

done