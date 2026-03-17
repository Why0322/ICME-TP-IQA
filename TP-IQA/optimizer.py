from torch import optim as optim
import torch

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer

# def make_optimizer_1stage(config, model, logger):
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # # 打印一个代表参数，确认范围
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         logger.info(f"[TRAIN] {name}")
    #         # break   # 只打印一条，防止日志爆炸
    
#     opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    
#     if opt_lower == 'sgd':
#         optimizer = torch.optim.SGD(
#             trainable_params,
#             lr=config.STAGE1.BASE_LR,
#             momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
#             nesterov=True,
#             weight_decay=config.STAGE1.WEIGHT_DECAY
#         )
#     elif opt_lower == 'adamw':
#         optimizer = torch.optim.AdamW(
#             trainable_params,
#             lr=config.STAGE1.BASE_LR,
#             eps=config.TRAIN.OPTIMIZER.EPS,
#             betas=config.TRAIN.OPTIMIZER.BETAS,
#             weight_decay=config.STAGE1.WEIGHT_DECAY
#         )
#     else:
#         raise ValueError(f"Unsupported optimizer: {opt_lower}")

#     logger.info(f"number of params passed to optimizer: "
#                 f"{sum(p.numel() for p in trainable_params)}")
#     exit()
#     return optimizer
        # value.requires_grad_(False)
        

def make_optimizer_1stage(config, model, logger):
    params = []
    for key, value in model.named_parameters():
        # params.append(value)
        if "prompt_learner" in key:
            logger.info(key)
            params.append(value)
            continue

        if "encoder_proj" in key:
            logger.info(key)
            params.append(value)
            continue

        if "prompt_proj" in key:
            logger.info(key)
            params.append(value)
            continue

        if "prompt_embeddings" in key:
            logger.info(key)
            params.append(value)
            continue

        if "decoder" in key:
            logger.info(key)
            params.append(value)
            continue

        if "logit_scale" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "adaptive_max_pool" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "query_linear" in key:
            logger.info(key)
            params.append(value)
            continue
        
        # if "image_adption" in key:
        #     logger.info(key)
        #     params.append(value)
        #     continue
        
        if "query_tokens" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "deep_features_embeddings" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "features_learner.ctx" in key:
            logger.info(key)
            params.append(value)
            continue
        
        if "attn_pool" in key:
            logger.info(key)
            params.append(value)
            continue
        

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(params, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.STAGE1.BASE_LR, weight_decay=config.STAGE1.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(params, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.STAGE1.BASE_LR, weight_decay=config.STAGE1.WEIGHT_DECAY)
    
    n_params_optimizer = sum(p.numel() for p in params if p.requires_grad)
    logger.info(f"number of params passed to optimizer: {n_params_optimizer}")
    return optimizer
    
# def make_optimizer_1stage(config, model, logger):
#     """
#     构造第一阶段优化器，只训练需要更新的参数：
#     1. 各类 prompt / ctx / query 向量
#     2. 轻量级适配器、MLP、decoder
#     3. MANIQA 双分支头
#     4. 冻结 CLIP 原始权重
#     """
#     # 需要训练的关键字清单
#     keywords = [
#         # "deep_features_embeddings",   # 深层 prompt
#         "prompt_embeddings",          # 浅层 prompt（已在清单，兼容旧日志）
#         # "query_tokens",               # 可学习 query 向量（含 local/quality）//应该不需要
#         "query_linear",               # query 映射层
#         # "features_learner.ctx",       # 场景/失真/质量文本 prompt
#         "image_adption",              # 768→512 适配器
#         "prompt_linear",              # IQE prompt 生成
#         "encoder_proj",               # 图像编码后投影
#         "prompt_proj",                # prompt 投影
#         "decoder",                    # TransformerDecoder
#         "logit_scale",                # 温度系数
#         "adaptive_max_pool",          # 空间池化
#         "prompt_learner",
#     ]

#     params = []
#     for key, value in model.named_parameters():
#         if any(k in key for k in keywords):
#             logger.info(f"[OPT] {key}")
#             params.append(value)
#         # else: 其余参数默认 requires_grad=False，已冻结

#     if not params:
#         raise RuntimeError("优化器未收到任何参数，请检查关键字是否匹配！")

#     # 构建优化器
#     opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
#     if opt_lower == 'sgd':
#         optimizer = optim.SGD(
#             params,
#             lr=config.STAGE1.BASE_LR,
#             momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
#             nesterov=True,
#             weight_decay=config.STAGE1.WEIGHT_DECAY
#         )
#     elif opt_lower == 'adamw':
#         optimizer = optim.AdamW(
#             params,
#             lr=config.STAGE1.BASE_LR,
#             eps=config.TRAIN.OPTIMIZER.EPS,
#             betas=config.TRAIN.OPTIMIZER.BETAS,
#             weight_decay=config.STAGE1.WEIGHT_DECAY
#         )
#     else:
#         raise ValueError(f"Unsupported optimizer: {opt_lower}")

#     n_params = sum(p.numel() for p in params if p.requires_grad)
#     logger.info(f"Number of params passed to optimizer: {n_params}")
#     # exit()
#     return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
