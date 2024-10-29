## KOMBO ##
python analysis/cross_attn.py mode=nlu_ft task=KorSTS \
    data.remain_lang=ko_punc data.do_hangeulize=false data.data_remove=false \
    model.hf_model=True model.name=skt/kogpt2-base-v2 \
    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
    model.set_kombo=true model.kombo.do_combination=true model.kombo.combination.combination_type=gru model.kombo.tok_type=jamo_var model.kombo.add_lora=false \
    logging.log_steps=10000 seed=1 \
    model.ckpt_dir=/data3/user21/KOMBO_Generation/logs/skt_kogpt2-base-v2/kombo/nlu_tasks_lrs/KorSTS/ko_punc/256t_64b_1s_1rs/ckpt

# ## LoRA ##
# python analysis/cross_attn.py mode=nlu_ft task=KorSTS \
#     data.remain_lang=ko_punc data.do_hangeulize=false data.data_remove=false \
#     model.hf_model=True model.name=skt/kogpt2-base-v2 \
#     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
#     model.set_kombo=false model.kombo.do_combination=false model.kombo.combination.combination_type=gru model.kombo.tok_type=jamo_var model.kombo.add_lora=false \
#     logging.log_steps=10000 seed=1 \
#     model.ckpt_dir=/data3/user21/KOMBO_Generation/logs/skt_kogpt2-base-v2/lora/nlu_tasks_lrs/KorSTS/ko_punc/256t_64b_1s_1rs/ckpt

