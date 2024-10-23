"""
Thanks to the author of the following blog post for the implementation of LoRA:
https://velog.io/@blackeyes0u0/LoRA%EB%A5%BC-%ED%86%B5%ED%95%9C-PEFT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA_Config:
    def __init__(self, r, lora_alpha, lora_dropout, target_modules):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        assert lora_alpha % r == 0


class LoRA_Layer(nn.Module):
    """
    Be careful with the name of weight parameters. Only the weight of LoRA layer should have the name 'lora_'.
    The name which has 'lora_' will be trained. The other weights will be frozen.
    """
    def __init__(self, original_layer, config: LoRA_Config):
        super(LoRA_Layer, self).__init__()
        self.original_layer = original_layer
        input_dim = original_layer.weight.shape[0]          # input dim size of lora A
        output_dim = original_layer.weight.shape[1]         # output dim size of lora B

        # Initialization
        lora_A_tensor = torch.empty(input_dim, config.r)
        torch.nn.init.kaiming_uniform_(lora_A_tensor)
        self.lora_A = nn.Parameter(lora_A_tensor)

        lora_B_tensor = torch.zeros(config.r, output_dim)
        self.lora_B = nn.Parameter(lora_B_tensor)

        self.scaling = config.lora_alpha // config.r

        if config.lora_dropout > 0:
            self.dropout = nn.Dropout(p=config.lora_dropout)
        else:
            self.dropout = lambda x: x          # pass

    def forward(self, x):
        # Apply dropout before the matrix multiplication
        A_dropout = self.dropout(self.lora_A)
        B_dropout = self.dropout(self.lora_B)
        W_prime = self.original_layer.weight + self.scaling * A_dropout @ B_dropout     # W' = W_0 + BA
        # print(f"x.shape: {x.shape}")
        # print(f"A_dropout.shape: {A_dropout.shape}")
        # print(f"B_dropout.shape: {B_dropout.shape}")
        # print(f"W_prime.shape: {W_prime.shape}")
        # print(f"self.original_layer.bias.shape: {self.original_layer.bias.shape}")
        return F.linear(x, W_prime.T, self.original_layer.bias)                           # y = W'x = (W_0 + BA)x = W_0x + BAx

    def __repr__(self):
        return f'{self.__class__.__name__}(\n  (original_layer): {self.original_layer},\n  (lora_A): Parameter of size {self.lora_A.size()},\n  (lora_B): Parameter of size {self.lora_B.size()}\n)'


def print_trainable_parameters(model, logger=None):
    print("\n")
    trainable_params = 0
    all_param = 0
    # for param in model.parameters():
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:     # True 이면 learnable parameter에 추가
            trainable_params += param.numel()
    if logger:
        logger.info(f"Trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f} %")
    else:
        print(f"Trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f} %")
    print("\n")
    return trainable_params, all_param


def apply_lora_to_model(model, config, logger=None):
    print("\n")
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        if len(hierarchy) > 1:      # Ensure the module is not the top-level module
            parent_module = model

            parent_names = hierarchy[:-1]
            if parent_names[-1] in ['attn', 'self_attn', 'attention', 'self']: pass
            else: continue
            for submodule_name in parent_names:  # Navigate to the parent module
                parent_module = getattr(parent_module, submodule_name)

            layer_name = hierarchy[-1]

            for target_module in config.target_modules:
                if target_module == layer_name:
                    original_layer = getattr(parent_module, layer_name)
                    if isinstance(original_layer, nn.Linear) or repr(original_layer) == 'Conv1D()':
                        setattr(parent_module, layer_name, LoRA_Layer(original_layer, config))
                        if logger:
                            logger.info(f"Replaced {name} with LoRA_Layer")
                        else:
                            print(f"Replaced {name} with LoRA_Layer")
    return model


def make_only_lora_as_trainable(model, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    if bias == 'none':
        pass
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRA_Layer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from transformers import CLIPModel, AutoTokenizer, GPT2LMHeadModel

    # model_choice = CLIPModel
    # model_name = "openai/clip-vit-base-patch32"
    # target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    model_choice = GPT2LMHeadModel
    model_name = "skt/kogpt2-base-v2"
    target_modules = ["c_attn", "c_proj"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_choice.from_pretrained(model_name)

    # Check the original number of parameters
    origin_num = sum(p.numel() for p in model.parameters())
    print("Original number of parameters:", origin_num)

    # Configuration for LoRA
    lora_config = LoRA_Config(
        r=16,
        lora_alpha=64,
        lora_dropout=0.03,
        target_modules=target_modules,
    )

    # Apply LoRA to the model
    model = apply_lora_to_model(model, lora_config)
    make_only_lora_as_trainable(model, bias='lora_only')
    trainable_model_params_num, basic_model_params_num = print_trainable_parameters(model)
