import torch.nn as nn
import torch
import torch.nn.functional as F


def my_loss_function(y, psi_hat, p_hat, nSite, k):
    """
    TODO: add description
    :param y:
    :param psi_hat:
    :param p_hat:
    :param nSite:
    :param k:
    :return:
    """
    maybe_absent = \
        (torch.sum(y, 1) == 0).to("cpu", dtype=torch.float32).reshape(-1,1)
    y_dist_if_present = \
        torch.distributions.bernoulli.Bernoulli(probs=p_hat.reshape(nSite,k))
    A = torch.mul(psi_hat, \
            torch.prod(y_dist_if_present.log_prob(y).exp(), 1).reshape(-1,1))
    B = 1 - psi_hat
    log_A_plus_B = torch.log(A + B * maybe_absent)

    if torch.sum(torch.isinf(log_A_plus_B)):
        log_A_plus_B[(log_A_plus_B<=0)] = torch.log(torch.tensor(1e-45))
    loss = -torch.mean(log_A_plus_B)
    return loss


class OD_LR(nn.Module):
    def __init__(self, x_dim, w_dim):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        """
        super(OD_LR, self).__init__()
        self.xlinear = nn.Linear(x_dim, 1)
        self.wlinear = nn.Linear(w_dim, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        psi = torch.sigmoid(self.xlinear(x))
        p = torch.sigmoid(self.wlinear(w))
        return psi, p


class OD_1NN(nn.Module):
    def __init__(self, x_dim, w_dim, k, nN):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param k:
        :param nN:
        """
        super(OD_1NN, self).__init__()
        self.k = k
        self.x_in = nn.Linear(x_dim, nN)
        self.x_out = nn.Linear(nN, 1)
        self.xw_inout = nn.Linear(nN + w_dim, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        x2 = F.elu(self.x_in(x))
        psi = torch.sigmoid(self.x_out(x2))

        x2 = x2[:, None, :]
        x2 = x2.repeat(1, self.k, 1)
        xw = torch.cat([x2, w], 2)

        p = torch.sigmoid(self.xw_inout(xw))
        return psi, p


class StatEcoNet_H1(nn.Module):
    def __init__(self, x_dim, w_dim, nN):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param nN:
        """
        super(StatEcoNet_H1, self).__init__()
        self.x_in = nn.Linear(x_dim, nN)
        self.w_in = nn.Linear(w_dim, nN)

        self.x_out = nn.Linear(nN, 1)
        self.w_out = nn.Linear(nN, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        x = F.elu(self.x_in(x))
        psi = torch.sigmoid(self.x_out(x))

        w = F.elu(self.w_in(w))
        p = torch.sigmoid(self.w_out(w))
        return psi, p


class StatEcoNet_H3(nn.Module):
    def __init__(self, x_dim, w_dim, nN):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param nN:
        """
        super(StatEcoNet_H3, self).__init__()
        self.x_in = nn.Linear(x_dim, nN)
        self.w_in = nn.Linear(w_dim, nN)

        self.x_h1 = nn.Linear(nN, nN * 2)
        self.x_h2 = nn.Linear(nN * 2, nN)

        self.w_h1 = nn.Linear(nN, nN * 2)
        self.w_h2 = nn.Linear(nN * 2, nN)

        self.x_out = nn.Linear(nN, 1)
        self.w_out = nn.Linear(nN, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        x = F.elu(self.x_in(x))
        x = F.elu(self.x_h1(x))
        x = F.elu(self.x_h2(x))
        psi = torch.sigmoid(self.x_out(x))

        w = F.elu(self.w_in(w))
        w = F.elu(self.w_h1(w))
        w = F.elu(self.w_h2(w))
        p = torch.sigmoid(self.w_out(w))
        return psi, p


# For bird data, we combine site and detection features
class OD_LR_Combined(nn.Module):
    def __init__(self, x_dim, w_dim, k):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param k:
        """
        super(OD_LR_Combined, self).__init__()
        self.k = k
        self.xlinear = nn.Linear(x_dim, 1)
        self.wlinear = nn.Linear(x_dim + w_dim, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        psi = torch.sigmoid(self.xlinear(x))

        x = x[:, None, :]
        x = x.repeat(1, self.k, 1)
        xw = torch.cat([x, w], 2)

        p = torch.sigmoid(self.wlinear(xw))
        return psi, p


class StatEcoNet_H1_Combined(nn.Module):
    def __init__(self, x_dim, w_dim, nN, k):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param nN:
        :param k:
        """
        super(StatEcoNet_H1_Combined, self).__init__()
        self.k = k
        self.x_in = nn.Linear(x_dim, nN)
        self.w_in = nn.Linear(x_dim + w_dim, nN)

        self.x_out = nn.Linear(nN, 1)
        self.w_out = nn.Linear(nN, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        x2 = F.elu(self.x_in(x))
        psi = torch.sigmoid(self.x_out(x2))

        x = x[:, None, :]
        x = x.repeat(1, self.k, 1)
        xw = torch.cat([x, w], 2)

        w = F.elu(self.w_in(xw))
        p = torch.sigmoid(self.w_out(w))
        return psi, p


class StatEcoNet_H3_Combined(nn.Module):
    def __init__(self, x_dim, w_dim, nN, k):
        """
        TODO: add description
        :param x_dim:
        :param w_dim:
        :param nN:
        :param k:
        """
        super(StatEcoNet_H3_Combined, self).__init__()
        self.k = k
        self.x_in = nn.Linear(x_dim, nN)
        self.w_in = nn.Linear(x_dim + w_dim, nN)

        self.x_h1 = nn.Linear(nN, nN * 2)
        self.x_h2 = nn.Linear(nN * 2, nN)

        self.w_h1 = nn.Linear(nN, nN * 2)
        self.w_h2 = nn.Linear(nN * 2, nN)

        self.x_out = nn.Linear(nN, 1)
        self.w_out = nn.Linear(nN, 1)

    def forward(self, x, w):
        """
        TODO: add description
        :param x:
        :param w:
        :return:
        """
        x2 = F.elu(self.x_in(x))
        x2 = F.elu(self.x_h1(x2))
        x2 = F.elu(self.x_h2(x2))
        psi = torch.sigmoid(self.x_out(x2))

        x = x[:, None, :]
        x = x.repeat(1, self.k, 1)
        xw = torch.cat([x, w], 2)

        w = F.elu(self.w_in(xw))
        w = F.elu(self.w_h1(w))
        w = F.elu(self.w_h2(w))
        p = torch.sigmoid(self.w_out(w))
        return psi, p

