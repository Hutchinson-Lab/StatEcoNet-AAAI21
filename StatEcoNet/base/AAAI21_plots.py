import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from heapq import nlargest


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def main_fig4(final2_model, x_dim, w_dim):
    """
    TODO: add description
    :param final2_model:
    :param x_dim:
    :param w_dim:
    :return:
    """
    true_x_fea_list = [0,1,2,3,4]
    true_w_fea_list = [0,1,2,3,4]

    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3), sharex=False)

    x = np.arange(x_dim)

    # StatEcoNet
    # Occupancy
    rid = 0
    data = torch.norm(list(final2_model.parameters())[0], dim=0).detach().numpy()
    x = np.arange(x_dim)
    bar = axes[0].bar(x, data)
    for i in true_x_fea_list:
        axes[0].bar(i, data[i], color="darkred")
    axes[0].set_xlabel("Site features")
    axes[0].set_ylabel("L2 norm")

    # Survey
    assert list(final2_model.parameters())[2].shape[1] == w_dim
    data = torch.norm(list(final2_model.parameters())[2], dim=0).detach().numpy()
    bar = axes[1].bar(x, data)
    for i in true_x_fea_list:
        axes[1].bar(i, data[i], color="darkred")
    axes[1].set_xlabel("Survey features")

    plt.tight_layout()
    plt.savefig("AAAI21/main/main_fig4.pdf", bbox_inches = "tight")


def main_fig5(y_test, psi1_hat, p1_hat, psi2_hat, p2_hat, \
              psi3_hat, p3_hat, psi4_hat, p4_hat, species_name, file_name=""):
    """
    TODO: add description
    :param y_test:
    :param psi1_hat:
    :param p1_hat:
    :param psi2_hat:
    :param p2_hat:
    :param psi3_hat:
    :param p3_hat:
    :param psi4_hat:
    :param p4_hat:
    :param species_name:
    :param file_name:
    :return:
    """
    labels = np.array(y_test).flatten()
    idx1 = np.where(labels==1)
    idx0 = np.where(labels==0)

    nVisits = 3
    decimal = 1
    text_size = 15
    neg_pnt = 50
    pos_pnt = 50
    alpha_size = 0.4
    lim_min = 0
    lim_max = 1

    # 1. OD-LR
    idx = 0
    NN_occ = np.repeat(np.array(psi1_hat), nVisits, axis=1).flatten()
    NN_det = np.array(p1_hat).flatten()

    NN_occ_round = np.round(NN_occ,decimal)
    NN_det_round = np.round(NN_det,decimal)

    matplotlib.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 7.5), sharex=False)
    fig.text(0.13, 1.01, 'OD-LR', va='center', ha='center', fontsize=18)
    fig.text(0.38, 1.01, 'OD-1NN', va='center', ha='center', fontsize=18)
    fig.text(0.62, 1.01, 'OD-BRT', va='center', ha='center', fontsize=18)
    fig.text(0.87, 1.01, 'StatEcoNet', va='center', ha='center', fontsize=18)

    h = axes[0,idx].hist2d(NN_occ_round[idx1], NN_det_round[idx1])
    axes[0,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[0,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[0,idx].set_title("detections (y=1)")
    axes[0,idx].set_xlim(lim_min,lim_max)
    axes[0,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[0,idx])

    h = axes[1,idx].hist2d(NN_occ_round[idx0], NN_det_round[idx0])
    axes[1,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[1,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[1,idx].set_title("non-detections (y=0)")
    axes[1,idx].set_xlim(lim_min,lim_max)
    axes[1,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[1,idx])

    # 2. OD-1NN
    idx = 1
    NN_occ = np.repeat(np.array(psi2_hat), nVisits, axis=1).flatten()
    NN_det = np.array(p2_hat).flatten()

    NN_occ_round = np.round(NN_occ,decimal)
    NN_det_round = np.round(NN_det,decimal)

    h = axes[0,idx].hist2d(NN_occ_round[idx1], NN_det_round[idx1])
    axes[0,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[0,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[0,idx].set_title("detections (y=1)")
    axes[0,idx].set_xlim(lim_min,lim_max)
    axes[0,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[0,idx])

    h = axes[1,idx].hist2d(NN_occ_round[idx0], NN_det_round[idx0])
    axes[1,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[1,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[1,idx].set_title("non-detections (y=0)")
    axes[1,idx].set_xlim(lim_min,lim_max)
    axes[1,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[1,idx])

    # 3. OD-BRT
    idx = 2
    NN_occ = np.repeat(np.array(psi3_hat), nVisits, axis=1).flatten()
    NN_det = np.array(p3_hat).flatten()

    NN_occ_round = np.round(NN_occ,decimal)
    NN_det_round = np.round(NN_det,decimal)

    h = axes[0,idx].hist2d(NN_occ_round[idx1], NN_det_round[idx1])
    axes[0,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[0,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[0,idx].set_title("detections (y=1)")
    axes[0,idx].set_xlim(lim_min,lim_max)
    axes[0,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[0,idx])

    h = axes[1,idx].hist2d(NN_occ_round[idx0], NN_det_round[idx0])
    axes[1,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[1,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[1,idx].set_title("non-detections (y=0)")
    axes[1,idx].set_xlim(lim_min,lim_max)
    axes[1,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[1,idx])

    # 3. StatEcoNet
    idx = 3
    NN_occ = np.repeat(np.array(psi4_hat), nVisits, axis=1).flatten()
    NN_det = np.array(p4_hat).flatten()

    NN_occ_round = np.round(NN_occ,decimal)
    NN_det_round = np.round(NN_det,decimal)

    h = axes[0,idx].hist2d(NN_occ_round[idx1], NN_det_round[idx1])
    axes[0,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[0,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[0,idx].set_title("detections (y=1)")
    axes[0,idx].set_xlim(lim_min,lim_max)
    axes[0,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[0,idx])

    h = axes[1,idx].hist2d(NN_occ_round[idx0], NN_det_round[idx0])
    axes[1,idx].set_xlabel("Est. Occupancy Prob.", fontsize=text_size)
    axes[1,idx].set_ylabel("Est. Detection Prob.", fontsize=text_size)
    axes[1,idx].set_title("non-detections (y=0)")
    axes[1,idx].set_xlim(lim_min,lim_max)
    axes[1,idx].set_ylim(lim_min,lim_max)
    fig.colorbar(h[3], ax=axes[1,idx])

    plt.tight_layout()
    if file_name == "":
        plt.savefig("AAAI21/main/main_fig5.pdf", bbox_inches = "tight")
    else:
        plt.savefig("AAAI21/supplement/"+file_name+".pdf", bbox_inches = "tight")


def main_fig6(f1_occ3, f1_occ4, f1_det3, f1_det4, x_dim, w_dim, fea_idx):
    """
    TODO: add description
    :param f1_occ3:
    :param f1_occ4:
    :param f1_det3:
    :param f1_det4:
    :param x_dim:
    :param w_dim:
    :param fea_idx:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), sharex=False)

    fig.text(0.29, 1.01, "Site Features", va='center', ha='center', fontsize=20)
    fig.text(0.76, 1.01, "Survey Features", va='center', ha='center', fontsize=20)

    fig.text(-0.02, 0.8, 'OD-BRT', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.32, 'StatEcoNet', va='center', rotation='vertical', fontsize=20)

    occ_x = np.arange(x_dim)
    det_x = np.arange(x_dim + w_dim)

    # 3. OD-BRT
    # Occupancy -- fold 1 =====================================
    topK = np.array(f1_occ3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ3[topK])

    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]

    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Selection#")

    # 4. StatEcoNet
    # Occupancy  -- fold 1 ==================================
    topK = np.array(f1_occ4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ4[topK])

    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]

    y_pos = np.arange(len(str_list))
    row_id = 1
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2 norm")

    # 3. OD-BRT
    # Detection -- fold 1 =====================================
    topK = np.array(f1_det3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_det3[topK])

    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]

    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Selection#")

    # 4. StatEcoNet
    # Detection  -- fold 1 ==================================
    topK = np.array(f1_det4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_det4[topK])

    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]

    y_pos = np.arange(len(str_list))
    row_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2 norm")

    plt.tight_layout()
    plt.savefig("AAAI21/main/main_fig6.pdf", bbox_inches = "tight")


def supp_syn_features(brt_site, brt_survey, final_model, final1_model, final2_model, \
                      x_dim, w_dim, file_name):
    """
    TODO: add description
    :param brt_site:
    :param brt_survey:
    :param final_model:
    :param final1_model:
    :param final2_model:
    :param x_dim:
    :param w_dim:
    :param file_name:
    :return:
    """
    true_x_fea_list = [0,1,2,3,4]
    true_w_fea_list = [0,1,2,3,4]

    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(7, 10), sharex=False)
    fig.text(-0.04, 0.89, 'OD-LR', va='center', rotation='vertical')
    fig.text(-0.04, 0.65, 'OD-1NN', va='center', rotation='vertical')
    fig.text(-0.04, 0.4, 'OD-BRT', va='center', rotation='vertical')
    fig.text(-0.04, 0.16, 'StatEcoNet', va='center', rotation='vertical')

    x = np.arange(x_dim)

    # OD-LR
    # Occupancy
    rid = 0
    data = torch.norm(list(final_model.parameters())[0], dim=0).detach().numpy()
    bar = axes[rid,0].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,0].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,0].set_xlabel("Site features")
    axes[rid,0].set_ylabel("Coeff. size")

    # Survey
    data = torch.norm(list(final_model.parameters())[2], dim=0).detach().numpy()
    bar = axes[rid,1].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,1].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,1].set_xlabel("Survey features")

    # OD-1MM
    # Occupancy
    rid = 1
    data = torch.norm(list(final1_model.parameters())[0], dim=0).detach().numpy()
    bar = axes[rid,0].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,0].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,0].set_xlabel("Site features")
    axes[rid,0].set_ylabel("L2 norm")

    # Survey
    #2. Not applicable
    axes[rid,1].axis('off')
    axes[rid,1].text(0.2, 0.4, 'Not Applicable')

    # OD-BRT
    # Occupancy
    rid = 2
    data = brt_site
    x = np.arange(x_dim)
    bar = axes[rid,0].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,0].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,0].set_xlabel("Site features")
    axes[rid,0].set_ylabel("Selection#")

    # Survey
    data = brt_survey
    bar = axes[rid,1].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,1].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,1].set_xlabel("Survey features")

    # StatEcoNet
    # Occupancy
    rid = 3
    data = torch.norm(list(final2_model.parameters())[0], dim=0).detach().numpy()
    x = np.arange(x_dim)
    bar = axes[rid,0].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,0].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,0].set_xlabel("Site features")
    axes[rid,0].set_ylabel("L2 norm")

    # Survey
    assert list(final2_model.parameters())[2].shape[1] == w_dim
    data = torch.norm(list(final2_model.parameters())[2], dim=0).detach().numpy()
    bar = axes[rid,1].bar(x, data)
    for i in true_x_fea_list:
        bar[i].set_color("darkred")
    axes[rid,1].axhline(y=np.min(nlargest(len(true_x_fea_list)+1, data)), color="k")
    axes[rid,1].set_xlabel("Survey features")

    plt.tight_layout()
    plt.savefig("AAAI21/supplement/"+file_name+".pdf", bbox_inches = "tight")


def supp_bird_site_features(f1_occ1, f2_occ1, f3_occ1, f1_occ2, f2_occ2, f3_occ2, \
                            f1_occ3, f2_occ3, f3_occ3, f1_occ4, f2_occ4, f3_occ4, \
                            x_dim, w_dim, fea_idx, key_feature, file_name):
    """
    TODO: add description
    :param f1_occ1:
    :param f2_occ1:
    :param f3_occ1:
    :param f1_occ2:
    :param f2_occ2:
    :param f3_occ2:
    :param f1_occ3:
    :param f2_occ3:
    :param f3_occ3:
    :param f1_occ4:
    :param f2_occ4:
    :param f3_occ4:
    :param x_dim:
    :param w_dim:
    :param fea_idx:
    :param key_feature:
    :param file_name:
    :return:
    """

    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 11), sharex=False)
    fig.text(0.5, 1.06, "Occupancy Important Features", va='center', ha='center', \
             fontsize=23)
    fig.text(0.18, 1.01, "Fold 1", va='center', ha='center', fontsize=20)
    fig.text(0.51, 1.01, "Fold 2", va='center', ha='center', fontsize=20)
    fig.text(0.84, 1.01, "Fold 3", va='center', ha='center', fontsize=20)
    fig.text(-0.02, 0.89, 'OD-LR', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.65, 'OD-1NN', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.4, 'OD-BRT', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.16, 'StatEcoNet', va='center', rotation='vertical', fontsize=20)
    occ_x = np.arange(x_dim)
    det_x = np.arange(x_dim + w_dim)

    # 1. OD-lR
    # Occupancy - fold1  ==================================
    topK = np.array(f1_occ1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Coeff. size")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy - fold2  ==================================
    topK = np.array(f2_occ1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_occ1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Coeff. size")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy - fold3 ==================================
    topK = np.array(f3_occ1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_occ1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 2
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Coeff. size")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # 2. OD-1NN
    # Occupancy - fold1 ==================================
    topK = np.array(f1_occ2).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ2[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 1
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2-norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy - fold2 ==================================
    topK = np.array(f2_occ2).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_occ2[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 1
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2-norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy - fold3 ==================================
    topK = np.array(f3_occ2).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_occ2[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 1
    col_id = 2
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2-norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # 2. OD-BRT
    # Occupancy -- fold 1 =====================================
    topK = np.array(f1_occ3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ3[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 2
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Selection#")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy -- fold 2 =====================================
    topK = np.array(f2_occ3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_occ3[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 2
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Selection#")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy -- fold 3 =====================================
    topK = np.array(f3_occ3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_occ3[topK])
    if len(set(sec_list)) != 1:
        l2 = [sec_list.index(x) for x in sorted(sec_list)]
        str_list = [str_list[i] for i in l2]
        sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 2
    col_id = 2
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("Selection#")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # 2. StatEcoNet
    # Occupancy  -- fold 1 ==================================
    topK = np.array(f1_occ4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_occ4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 3
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2 norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy  -- fold 2 ==================================
    topK = np.array(f2_occ4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_occ4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 3
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2 norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Occupancy  -- fold 3 ==================================
    topK = np.array(f3_occ4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_occ4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 3
    col_id = 2
    # Create horizontal barsfor i in range(len(str_list))
    bar = axes[row_id,col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id,col_id].set_yticks(y_pos)
    axes[row_id,col_id].set_yticklabels(str_list)
    axes[row_id,col_id].set_xlabel("L2 norm")
    axes[row_id,col_id].set_ylabel("Site Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    plt.tight_layout()
    plt.savefig("AAAI21/supplement/"+file_name+".pdf", bbox_inches = "tight")


def supp_bird_survey_features(f1_det1, f2_det1, f3_det1, f1_det3, f2_det3, f3_det3, \
                              f1_det4, f2_det4, f3_det4, x_dim, w_dim, \
                              fea_idx, key_feature, file_name):
    """
    TODO: add description
    :param f1_det1:
    :param f2_det1:
    :param f3_det1:
    :param f1_det3:
    :param f2_det3:
    :param f3_det3:
    :param f1_det4:
    :param f2_det4:
    :param f3_det4:
    :param x_dim:
    :param w_dim:
    :param fea_idx:
    :param key_feature:
    :param file_name:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 8), sharex=False)
    fig.text(0.5, 1.06, "Detection Important Features", va='center', ha='center', \
             fontsize=23)
    fig.text(0.18, 1.01, "Fold 1", va='center', ha='center', fontsize=20)
    fig.text(0.51, 1.01, "Fold 2", va='center', ha='center', fontsize=20)
    fig.text(0.84, 1.01, "Fold 3", va='center', ha='center', fontsize=20)
    fig.text(-0.02, 0.86, 'OD-LR', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.53, 'OD-BRT', va='center', rotation='vertical', fontsize=20)
    fig.text(-0.02, 0.22, 'StatEcoNet', va='center', rotation='vertical', fontsize=20)

    occ_x = np.arange(x_dim)
    det_x = np.arange(x_dim + w_dim)

    # 1. OD-lR
    # Detection - fold1  ==================================
    topK = np.array(f1_det1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_det1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Coeff. size")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Detection - fold2  ==================================
    topK = np.array(f2_det1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_det1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Coeff. size")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

            # Detection - fold3 ==================================
    topK = np.array(f3_det1).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_det1[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 0
    col_id = 2
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Coeff. size")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # 2. OD-BRT
    # Detection -- fold 1 =====================================
    topK = np.array(f1_det3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_det3[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 1
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Selection#")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Detection -- fold 2 =====================================
    topK = np.array(f2_det3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_det3[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Selection#")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Detection -- fold 3 =====================================
    topK = np.array(f3_det3).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_det3[topK])
    if len(set(sec_list)) != 1:
        l2 = [sec_list.index(x) for x in sorted(sec_list)]
        str_list = [str_list[i] for i in l2]
        sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    col_id = 2
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("Selection#")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # 2. StatEcoNet
    # Detection  -- fold 1 ==================================
    topK = np.array(f1_det4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f1_det4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    row_id = 2
    col_id = 0
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("L2 norm")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Detection  -- fold 2 ==================================
    topK = np.array(f2_det4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f2_det4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    col_id = 1
    # Create horizontal bars
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("L2 norm")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    # Detection  -- fold 3 ==================================
    topK = np.array(f3_det4).argsort()[-5:][::-1]
    str_list = fea_idx.iloc[topK].to_numpy().flatten().tolist()
    sec_list = list(f3_det4[topK])
    l2 = [sec_list.index(x) for x in sorted(sec_list)]
    str_list = [str_list[i] for i in l2]
    sec_list = [sec_list[i] for i in l2]
    y_pos = np.arange(len(str_list))
    col_id = 2
    # Create horizontal barsfor i in range(len(str_list))
    bar = axes[row_id, col_id].barh(y_pos, sec_list)
    # Create names on the y-axis
    axes[row_id, col_id].set_yticks(y_pos)
    axes[row_id, col_id].set_yticklabels(str_list)
    axes[row_id, col_id].set_xlabel("L2 norm")
    axes[row_id, col_id].set_ylabel("Survey Features")

    for i in range(len(str_list)):
        if str_list[i] == key_feature:
            bar[i].set_color("darkred")

    plt.tight_layout()
    plt.savefig("AAAI21/supplement/" + file_name + ".pdf", bbox_inches="tight")