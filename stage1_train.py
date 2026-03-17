import datetime
import time
import torch
from datetime import timedelta
import os
from torch.cuda import amp
import numpy as np
import torch.distributed as dist
from torch.nn import functional as F
from scipy import stats
from timm.utils import AverageMeter  # accuracy
from loss import SupConLoss, ImSupConLoss, Fidelity_Loss, Fidelity_Loss_distortion, Multi_Fidelity_Loss, InfoNCE_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loss import loss_quality, ranking_loss_multi, ranking_loss
from labelsmooth_loss import CrossEntropyLabelSmooth
from utils import save_checkpoint
from eval import cross_eval
import torch.nn as nn



def stage1_train(config, model, data_loader, epochs, optimizer, lr_scheduler, logger):
    max_plcc, max_srcc, max_plcc_c, max_srcc_c = 0.0, 0.0, 0.0, 0.0
    loss_scaler = torch.amp.GradScaler()
    loss_meter = AverageMeter()
    start_time = time.monotonic()
    logger.info("start training")
    logger.info(f"config.ALPHA: {config.ALPHA}, config.BETA: {config.BETA}")
    pred_scores_list, gt_scores_list = [], []
    train_srcc = 0.0
    # val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger)
    # huber = HuberLoss(delta=1.0).cuda()   # delta 可调，1.0 等价于 SmoothL1 的 β=1
    smooth_loss_global = CrossEntropyLabelSmooth(num_classes=config.DATA.SCENE_NUM_CLASSES)
    smooth_loss_local = CrossEntropyLabelSmooth(num_classes=config.DATA.DIST_NUM_CLASSES)
    smooth_loss_quality = CrossEntropyLabelSmooth(num_classes=config.DATA.QUALITY_NUM_CLASSES)
    smooth_loss_rank = CrossEntropyLabelSmooth(num_classes=config.DATA.QUALITY_NUM_CLASSES)
    for epoch in range(1, epochs + 1):
        pred_scores_list.clear()
        gt_scores_list.clear()
        logger.info(f"Epoch{epoch} training")
        loss_meter.reset()
        # lr_scheduler.step(epoch)
        model.train()
        for n_iter, (img, gt_score, scene_num, distortion_num ,quality_num) in enumerate(data_loader):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            gt_score = gt_score.cuda(non_blocking=True)
            scene_num = scene_num.cuda(non_blocking=True)
            distortion_num = distortion_num.cuda(non_blocking=True)
            quality_num = quality_num.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                pred_score, logits_global ,logits_local,logits_quality,logits_rank= model(img)
                # pred_score,logits_rank= model(img)
                
            global_loss, local_loss,quality_loss,ranking_loss = 0.0, 0.0, 0.0 ,0.0

            # multi-label
            if logits_global is not None:
                # smooth_loss_global = CrossEntropyLabelSmooth(num_classes=config.DATA.SCENE_NUM_CLASSES)
                # print("scene_num",scene_num)
                # exit()
                # print("",scene_num.type())
                global_loss = smooth_loss_global(logits_global, scene_num)
                # global_loss = ranking_loss_multi(logits_global, scene_num, scene2, scene3, scale_=4.0)

            if logits_local is not None:
                # smooth_loss_local = CrossEntropyLabelSmooth(num_classes=config.DATA.DIST_NUM_CLASSES)     
                # print("scene_num",distortion_num,distortion_num.type())
                # exit()      
                local_loss = smooth_loss_local(logits_local, distortion_num)
                # local_loss = ranking_loss(logits_local, distortion_num, scale_=4.0)

            if logits_quality is not None:
                # smooth_loss_quality = CrossEntropyLabelSmooth(num_classes=config.DATA.QUALITY_NUM_CLASSES)           
                quality_loss = smooth_loss_quality(logits_quality, quality_num)
                ranking_loss = smooth_loss_rank(logits_rank, quality_num)
                # local_loss = ranking_loss(logits_local, distortion_num, scale_=4.0)
            
            fidelity_loss = loss_quality(pred_score, gt_score)
            with torch.amp.autocast('cuda',enabled=config.AMP_ENABLE):
                smoothl1_loss = torch.nn.SmoothL1Loss()(pred_score.float(), gt_score.float())


                # smoothl1_loss = huber(pred_score.float(), gt_score.float())
            # smoothl1_loss = torch.nn.SmoothL1Loss()(pred_score, gt_score)
            # plcc_l = plcc_loss(pred_score, gt_score)          # 1-PLCC
            # delta = torch.tensor(1.0, device=pred_score.device, dtype=pred_score.dtype)
            # HuberLoss = F.huber_loss(pred_score, gt_score, delta=delta)
            loss = global_loss + config.ALPHA * local_loss +   config.BETA * smoothl1_loss + quality_loss + ranking_loss 

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            pred_scores_list = pred_scores_list + pred_score.squeeze().cpu().tolist()
            gt_scores_list = gt_scores_list + gt_score.squeeze().cpu().tolist()
            train_srcc, _ = stats.spearmanr(pred_scores_list, gt_scores_list)
            torch.cuda.synchronize()
            if n_iter % 10 == 0:
                logger.info(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(data_loader)}] Loss: {loss_meter.avg:.3f}, Global_Loss: {global_loss:.3f}, Local_Loss: {local_loss:.3f} Quality_Loss:{quality_loss:.3f} RANK_loss:{ranking_loss:.3f} Fidelity_Loss: {fidelity_loss:.3f} Smoothl1_loss:{smoothl1_loss:.3f} Base Lr: {lr_scheduler._get_lr(epoch)[0]:.2e}, train_srcc: {train_srcc:.3f}")
        
        lr_scheduler.step(epoch)
        # if epoch % 5 == 0:
        if epoch >= 5:
            val_plcc, val_srcc, val_plccc, val_srccc = cross_eval(config, model, logger)
            logger.info(f"stage 1 validate:{val_plcc}, {val_srcc}, {val_plccc}, {val_srccc}")
            # use the best of previous srccs for comparison
            maxsrcc = max(max_srcc, max_srcc_c)
            maxplcc = max(max_plcc, max_plcc_c)
            # if validation improves, update best and save checkpoint
            if val_plcc >= maxplcc or val_plccc >= maxplcc:
                max_plcc = val_plcc
                max_srcc = val_srcc
                max_plcc_c = val_plccc
                max_srcc_c = val_srccc
                logger.info(f"New best found at epoch {epoch}: plcc={max_plcc:.6f}, srcc={max_srcc:.6f}")
                # save best checkpoint (only on rank 0 if distributed)
                # try:
                #     if not dist.is_initialized() or dist.get_rank() == 0:
                #         save_checkpoint(config, epoch, model, max_plcc, optimizer, lr_scheduler, loss_scaler, logger)
                # except Exception:
                #     # best-effort save; do not crash training if saving fails
                #     logger.exception("Failed to save checkpoint for best model")
            logger.info(f"stage 1 max:{max_plcc}, {max_srcc}, {max_plcc_c}, {max_srcc_c}")

    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Stage1 running time: {}".format(total_time))
    return max_plcc, max_srcc, max_plcc_c, max_srcc_c
