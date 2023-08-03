import import_ipynb
from base.models import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import time
import copy


def train_model(trial, model_id, nL, nN, lr, task, \
                x_dim, w_dim, k, n_epoch, mixed_weight, \
                x_train, w_train, y_train, x_te, w_te, y_te, \
                occ_train, det_train, occ_te, det_te, batch_size, P=1):
    """
    TODO: add description
    :param trial:
    :param model_id:
    :param nL:
    :param nN:
    :param lr:
    :param task:
    :param x_dim:
    :param w_dim:
    :param k:
    :param n_epoch:
    :param mixed_weight:
    :param x_train:
    :param w_train:
    :param y_train:
    :param x_te:
    :param w_te:
    :param y_te:
    :param occ_train:
    :param det_train:
    :param occ_te:
    :param det_te:
    :param batch_size:
    :param P:
    :return:
    """
    # For mini batches
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}
    train_dataset = TensorDataset(x_train, w_train, y_train, \
                                  torch.tensor(np.array(occ_train), dtype=torch.float32), \
                                  torch.tensor(np.array(det_train).reshape(x_train.shape[0], k), \
                                               dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, **params)

    # Model selection
    torch.manual_seed(trial)
    if model_id == 0:
        model = OD_LR(x_dim, w_dim)
    elif model_id == 1:
        model = OD_1NN(x_dim, w_dim, k, nN)
    elif model_id == 2:
        occ_idx = 0
        det_idx = 2
        if nL == 1:
            model = StatEcoNet_H1(x_dim, w_dim, nN)
        elif nL == 3:
            model = StatEcoNet_H3(x_dim, w_dim, nN)
        else:
            print("not available model")
            assert False
    else:
        print("not available model")
        assert False

    # Set an optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Record training/testing results per iteration
    test_auprc = []
    df_train = pd.DataFrame(columns=['lr', 'batchSize', 'nLayers', 'nNeurons', \
                                     'nIter', 'loss', 'auroc', 'auprc', \
                                     'occCorr', 'detCorr'])
    df_test = pd.DataFrame(columns=['lr', 'batchSize', 'nLayers', 'nNeurons', \
                                    'nIter', 'loss', 'auroc', 'auprc', \
                                    'occCorr', 'detCorr'])

    # Start training a model
    best_iter = 0
    best_model = 0
    best_elapse = 0
    perfect = False  # is this needed?
    start = time.time()
    for i in tqdm(range(n_epoch)):
        # for mini batches
        train_loss = []
        train_y_true = []
        train_occ_true = []
        train_det_true = []
        train_y_pred = []
        train_occ_pred = []
        train_det_pred = []
        for i_batch, xy in enumerate(train_dataloader):
            # load a minibatch
            # tr is true?
            x_tr, w_tr, y_tr, occ_tr, det_tr = xy
            train_y_true.extend(list(torch.flatten(y_tr).detach().numpy()))
            train_occ_true.extend(list(torch.flatten(occ_tr).detach().numpy()))
            train_det_true.extend(list(torch.flatten(det_tr).detach().numpy()))

            # Train a model **************************************************
            model.train()
            optimizer.zero_grad()
            psi_hat_train, p_hat_train = model(x_tr, w_tr)
            train_occ_pred.extend( \
                list(torch.flatten(psi_hat_train).detach().numpy()))
            train_det_pred.extend( \
                list(torch.flatten(p_hat_train).detach().numpy()))

            # Compute training loss
            loss = my_loss_function(y_tr, psi_hat_train, p_hat_train, \
                                    x_tr.shape[0], k)
            if mixed_weight:
                params = list(model.parameters())
                loss += mixed_weight * \
                        (torch.sum(torch.norm(params[occ_idx], dim=0))) ** (1 / P)
                loss += mixed_weight * \
                        (torch.sum(torch.norm(params[det_idx], dim=0))) ** (1 / P)
            train_loss.append(loss.item())

            # Compute Y from psi_hat and p_hat
            NN_pred = p_hat_train.reshape(p_hat_train.shape[:2]) * \
                      torch.cat([psi_hat_train] * k, 1)
            NN_pred = torch.flatten(NN_pred).detach().numpy()
            train_y_pred.extend(list(NN_pred))

            loss.backward()
            optimizer.step()

        # After learning all samples
        assert np.sum(train_y_true) == torch.sum(y_train)

        # Compute accuracy on train Y ****************************************
        fpr, tpr, thresholds = metrics.roc_curve(train_y_true, train_y_pred)
        auroc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(train_y_true, \
                                                               train_y_pred)
        auprc = metrics.auc(recall, precision)

        # Compute correlation on prob.
        occCorr = pearsonr(train_occ_true, train_occ_pred)[0]
        detCorr = pearsonr(train_det_true, train_det_pred)[0]

        # Record training results per iteration
        df_train.loc[i] = [lr, batch_size, nL, nN, i, np.mean(train_loss), \
                           auroc, auprc, occCorr, detCorr]

        # Evalaute the trained model *****************************************
        model.eval()
        with torch.no_grad():
            psi_hat_test, p_hat_test = model(x_te, w_te)

            # Compute test loss
            loss_t = my_loss_function(y_te, psi_hat_test, p_hat_test, \
                                      x_te.shape[0], k)
            if mixed_weight:
                params = list(model.parameters())
                loss_t += mixed_weight * \
                          (torch.sum(torch.norm(params[occ_idx], dim=0))) ** (1 / P)
                loss_t += mixed_weight * \
                          (torch.sum(torch.norm(params[det_idx], dim=0))) ** (1 / P)

                # Compute Y from psi_hat and p_hat
            NN_pred = p_hat_test.reshape(p_hat_test.shape[:2]) * \
                      torch.cat([psi_hat_test] * k, 1)
            NN_pred = torch.flatten(NN_pred).detach().numpy()

            # Compute accuracy on Y
            fpr, tpr, thresholds = metrics.roc_curve(torch.flatten(y_te), \
                                                     NN_pred)
            auroc = metrics.auc(fpr, tpr)
            precision, recall, thresholds = \
                precision_recall_curve(torch.flatten(y_te), NN_pred)
            auprc = metrics.auc(recall, precision)
            test_auprc.append(auprc)

            # CHECKING THE BEST ITERATION =================
            if test_auprc[-1] == np.max(test_auprc):
                best_iter = i
                best_model = copy.deepcopy(model)
                best_elapse = time.time() - start
            else:
                if task == "train" and i > best_iter + 200:
                    print("No more improvement. This is the early stop point.")
                    break
            # =============================================

            # Compute correlation on prob.
            occCorr = pearsonr(occ_te.to_numpy().flatten(), \
                               psi_hat_test.detach().numpy().flatten())[0]
            detCorr = pearsonr(det_te.to_numpy().flatten(), \
                               p_hat_test.detach().numpy().flatten())[0]

            df_test.loc[i] = [lr, batch_size, nL, nN, i, loss_t.item(), \
                              auroc, auprc, occCorr, detCorr]

    assert best_iter < n_epoch
    if task == "train":
        return df_train, df_test, best_iter, best_model, best_elapse
    else:
        return df_train, df_test, model, psi_hat_test, p_hat_test, NN_pred


def test_model(model, x_te, w_te, y_te, occ_te, det_te, k):
    """
    TODO: add description
    :param model:
    :param x_te:
    :param w_te:
    :param y_te:
    :param occ_te:
    :param det_te:
    :param k:
    :return:
    """
    model.eval()
    with torch.no_grad():
        psi_hat_test, p_hat_test = model(x_te, w_te)

        # compute Y from psi_hat and p_hat
        NN_pred = p_hat_test.reshape(p_hat_test.shape[:2]) * \
                  torch.cat([psi_hat_test]*k, 1)
        NN_pred = torch.flatten(NN_pred).detach().numpy()

        # compute accuracy on Y
        fpr, tpr, thresholds = metrics.roc_curve(torch.flatten(y_te), NN_pred)
        auroc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = \
                        precision_recall_curve(torch.flatten(y_te), NN_pred)
        auprc = metrics.auc(recall, precision)

        # compute correlation on prob.
        occCorr = pearsonr(occ_te.to_numpy().flatten(), \
                           psi_hat_test.detach().numpy().flatten())[0]
        detCorr = pearsonr(det_te.to_numpy().flatten(), \
                           p_hat_test.detach().numpy().flatten())[0]

        return(psi_hat_test, p_hat_test, NN_pred, \
               auroc, auprc, occCorr, detCorr)

    