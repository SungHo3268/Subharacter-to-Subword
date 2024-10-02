task=$1
remain_lang=$2

# do hangeulize & data_remove #
## LoRA ##
python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
    data.remain_lang=${remain_lang} data.do_hangeulize=true data.data_remove=true \
    optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
    optim.base_lr=1e-02 data.max_length=256 \
    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
    model.set_kombo=false model.kombo.do_combination=false model.kombo.add_lora=false
## KOMBO (+LoRA) ##
python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
    data.remain_lang=${remain_lang} data.do_hangeulize=true data.data_remove=true \
    optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
    optim.base_lr=1e-02 data.max_length=256 \
    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
    model.set_kombo=true model.kombo.do_combination=true model.kombo.add_lora=true model.kombo.lora.r=32 model.kombo.lora.alpha=128

# # do hangeulize #
# ## LoRA ##
# python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
#     data.remain_lang=${remain_lang} data.do_hangeulize=true data.data_remove=false \
#     optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
#     optim.base_lr=1e-02 data.max_length=256 \
#     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
#     model.set_kombo=false model.kombo.do_combination=false model.kombo.add_lora=false
# ## KOMBO (+LoRA) ##
# python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
#     data.remain_lang=${remain_lang} data.do_hangeulize=true data.data_remove=false \
#     optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
#     optim.base_lr=1e-02 data.max_length=256 \
#     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
#     model.set_kombo=true model.kombo.do_combination=true model.kombo.add_lora=true model.kombo.lora.r=32 model.kombo.lora.alpha=128

# data_remove #
## LoRA ##
python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
    data.remain_lang=${remain_lang} data.do_hangeulize=false data.data_remove=true \
    optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
    optim.base_lr=1e-02 data.max_length=256 \
    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
    model.set_kombo=false model.kombo.do_combination=false model.kombo.add_lora=false
## KOMBO (+LoRA) ##
python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
    data.remain_lang=${remain_lang} data.do_hangeulize=false data.data_remove=true \
    optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
    optim.base_lr=1e-02 data.max_length=256 \
    model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
    model.set_kombo=true model.kombo.do_combination=true model.kombo.add_lora=true model.kombo.lora.r=32 model.kombo.lora.alpha=128

# # Raw #
# ## LoRA ##
# python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
#     data.remain_lang=${remain_lang} data.do_hangeulize=false data.data_remove=false \
#     optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
#     optim.base_lr=1e-02 data.max_length=256 \
#     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
#     model.set_kombo=false model.kombo.do_combination=false model.kombo.add_lora=false
# ## KOMBO (+LoRA) ##
# python nlu_tasks/scripts/run_finetuning.py mode=nlu_ft task=$task \
#     data.remain_lang=${remain_lang} data.do_hangeulize=false data.data_remove=false \
#     optim.grad_acc=1 model.hf_model=True model.name=skt/kogpt2-base-v2 \
#     optim.base_lr=1e-02 data.max_length=256 \
#     model.set_lora=true model.lora.r=32 model.lora.alpha=128 \
#     model.set_kombo=true model.kombo.do_combination=true model.kombo.add_lora=true model.kombo.lora.r=32 model.kombo.lora.alpha=128
