task=$1


for remain_lang in ko_en_punc ko_punc
do
    for do_hangeulize in true
    do
        for data_remove in true false
        do
            for seed in 1 2 3
            do
                ## KOMBO ##
                python nlu_tasks/scripts/run_finetuning_lrs.py mode=nlu_ft task=$task \
                    seed=$seed data.remain_lang=$remain_lang data.do_hangeulize=$do_hangeulize data.data_remove=$data_remove \
                    optim.grad_acc=1 optim.base_lrs="1e-02 2e-02" data.max_length=256 logging.log_steps=10000 \
                    model.hf_model=True model.name=skt/kogpt2-base-v2 \
                    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
                    model.set_kombo=false model.kombo.add_lora=false
            done
        done
    done
done