import torch
import time
import numpy as np
import random
from torch.nn import functional as F
import utils


class RoSAS:
    def __init__(self, device='cuda', nbatch_per_epoch=16, epochs=100, batch_size=128,
                 network='e1s1', n_emb=128, lr=0.005, margin=1., alpha=0.5, beta=1., T=2, k=2,
                 score_loss='smooth', milestones=None,
                 prt_step=1, use_es=True, seed=42):
        self.device = device

        self.epochs = epochs
        self.nbatch_per_epoch = nbatch_per_epoch
        self.batch_size = batch_size
        self.lr = lr

        self.network = network
        self.n_emb = n_emb

        self.k = k
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.score_loss = score_loss
        self.milestones = milestones if milestones is not None else (self.epochs,)

        self.prt_step = prt_step
        self.use_es = use_es

        self.basenet = None
        self.criterion = None
        self.data = None
        self.dim = None

        self.param_lst = locals()
        del self.param_lst['self']
        del self.param_lst['device']
        del self.param_lst['prt_step']
        print(self.param_lst)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        return

    def fit(self, train_x, train_semi_y, val_x, val_y):
        device = self.device
        dim = train_x.shape[1]
        self.dim = dim

        # self.basenet = get_arch(self.network, dim=self.dim, n_emb=self.n_emb, device=self.device)

        n_hidden = dim + int((self.n_emb - dim) * 0.5)
        n_hidden2 = int(0.5 * self.n_emb)
        self.basenet = EDOSNet(n_feature=dim, n_hidden=n_hidden,
                               n_hidden2=n_hidden2, n_emb=self.n_emb)
        self.basenet = self.basenet.to(self.device)

        self.data = DataGenerator(train_x, train_semi_y, batch_size=self.batch_size)

        self.criterion = Loss(
            l2_reg_weight=1e-2, score_loss=self.score_loss,
            margin=self.margin, alpha=self.alpha, beta=self.beta, T=self.T, k=self.k,
            device=self.device
        )

        optimizer = torch.optim.Adam(self.basenet.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.4)
        early_stp = utils.EarlyStopping(patience=15, model_name='edos', verbose=False)

        pre_loss_emb, pre_loss_score = 1, 1
        print("start training epochs...")
        for step in range(self.epochs):
            start = time.time()

            batch_triplets = self.data.load_batches(n_batches=self.nbatch_per_epoch)
            batch_triplets = torch.from_numpy(batch_triplets).float().to(device)

            losses, losses1, losses2 = [], [], []
            losses_out, losses_intra = [], []
            self.basenet.train()
            for batch_triplet in batch_triplets:
                anchor, pos, neg = batch_triplet[:, 0], batch_triplet[:, 1], batch_triplet[:, 2]

                loss, loss1, loss2, loss_out, loss_intra = self.criterion(
                    self.basenet, anchor, pos, neg, pre_loss_emb, pre_loss_score
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.data.cpu().item())
                losses1.append(loss1.data.cpu().item())
                losses2.append(loss2.data.cpu().item())
                losses_out.append(loss_out.data.cpu().item())
                losses_intra.append(loss_intra.data.cpu().item())

            end = time.time()

            val_start = time.time()
            self.basenet.eval()

            # --- validate on validation set --- #
            val_score = self.predict(val_x)
            try:
                val_auroc, val_aupr = utils.evaluate(val_y, val_score)
                if self.use_es:
                    early_metric = (1 - val_aupr) + (1 - val_auroc)
                    early_stp(early_metric, model=self.basenet)
            except ValueError:
                val_auroc, val_aupr = -1, -1
                if self.use_es:
                    early_metric = (1 - val_aupr) + (1 - val_auroc)
                    early_stp(early_metric, model=self.basenet)
                    early_stp.early_stop = True
            val_end = time.time()

            t = end - start
            val_t = val_end - val_start
            losses, losses1, losses2 = np.array(losses), np.array(losses1), np.array(losses2)
            losses_out, losses_intra = np.array(losses_out), np.array(losses_intra)
            if (step + 1) % self.prt_step == 0 or step == 0:
                print(f'epoch {step+1}, '
                      f'loss (combine/emb/score): {losses.mean():.4f} / {losses1.mean():.4f} / {losses2.mean():.4f}, '
                      f'loss (out/intra): {losses_out.mean():.4f} / {losses_intra.mean():.4f}, '
                      f'val-auroc/pr: {val_auroc:.4f}/{val_aupr:.4f}, time: {t:.2f}s')
                # print("epoch: %3d, loss(combine/emb/score): %.4f/%.4f/%.4f, val-auroc/pr: %.4f/%.4f, time: %.2fs, %.2fs"
                #       % (step + 1, losses.mean(), losses1.mean(), losses2.mean(), val_auroc, val_aupr, t, val_t))

            if self.use_es and early_stp.early_stop:
                self.basenet.load_state_dict(torch.load(early_stp.path))
                print("early stop", self.use_es)
                break

            scheduler.step()

            pre_loss_emb = losses1.mean()
            pre_loss_score = losses2.mean()
        return

    def predict(self, x_test):
        device = self.device
        with torch.no_grad():
            self.basenet.eval()
            xx = torch.from_numpy(x_test).float().to(device)
            _, xx_s = self.basenet(xx)
            xx_s = xx_s.flatten()
            xx_s = xx_s.data.cpu().numpy()
        return xx_s




class EDOSNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_emb, n_hidden2):
        super(EDOSNet, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden, bias=False)
        self.emb_layer = torch.nn.Linear(n_hidden, n_emb, bias=False)

        self.hidden_layer2 = torch.nn.Linear(n_emb, n_hidden2, bias=False)
        self.hidden_layer2_dup = torch.nn.Linear(n_emb, n_hidden2, bias=False)
        self.out_layer = torch.nn.Linear(n_hidden2, 1)

    def forward(self, x, dup=False):
        x = F.leaky_relu(self.hidden_layer(x))
        emb_x = self.emb_layer(x)

        s = F.leaky_relu(self.hidden_layer2(emb_x))
        s = torch.tanh(self.out_layer(s))

        s2 = F.leaky_relu(self.hidden_layer2_dup(emb_x))
        s2 = torch.tanh(self.out_layer(s2))

        if not dup:
            return emb_x, s
        else:
            return emb_x, s, s2

class Loss(torch.nn.Module):
    def __init__(self, l2_reg_weight=0., margin=1., alpha=1., beta=2., T=2, k=2, score_loss='smooth', device='cuda'):
        super(Loss, self).__init__()
        self.l2_reg_weight = l2_reg_weight
        self.loss_tri = torch.nn.TripletMarginLoss(margin=margin)
        self.T = T
        self.alpha = alpha
        self.k = k

        if score_loss == 'mse':
            self.loss_reg = torch.nn.MSELoss(reduction='none')
        elif score_loss == 'mae':
            self.loss_reg = torch.nn.MSELoss(reduction='none')
        elif score_loss == 'smooth':
            self.loss_reg = torch.nn.SmoothL1Loss(
                reduction='none',
                # beta=beta
            )
        else:
            raise ValueError('unsupported loss')

        self.device = device

        return

    def forward(self, basenet, anchor, pos, neg, pre_emb_loss, pre_score_loss):
        anchor_emb, anchor_s = basenet(anchor)
        pos_emb, pos_s = basenet(pos)
        neg_emb, neg_s = basenet(neg)

        # embedding loss
        loss_emb = self.loss_tri(anchor_emb, pos_emb, neg_emb)
        l2_reg = torch.norm(anchor_emb + pos_emb + neg_emb, p=2)

        # # # regression loss on anomalies
        # loss_reg1 = self.loss_reg(neg_s, torch.ones_like(neg_s)).mean()
        #
        # # # regression loss on normal
        # loss_reg0 = self.loss_reg(pos_s, -1 * torch.ones_like(neg_s)).mean()


        # # different lambdas in different pairs
        # # normal-normal
        # Beta00 = torch.distributions.dirichlet.Dirichlet(torch.tensor([1., 1.]))
        # lambdas00 = Beta00.sample(target_i.flatten().shape).to(self.device)[:, 1]
        #
        # # anomaly-anomaly
        # Beta11 = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.5, 0.5]))
        # lambdas11 = Beta11.sample(target_i.flatten().shape).to(self.device)[:, 1]
        #
        # # anomaly-normal
        # Beta01 = torch.distributions.dirichlet.Dirichlet(torch.tensor([2., 2.]))
        # lambdas01 = Beta01.sample(target_i.flatten().shape).to(self.device)[:, 1]
        #
        # lambdas = torch.zeros_like(lambdas00)
        # idx = (target_i.flatten() == -1.) & (target_j.flatten() == -1.)
        # lambdas[idx] = lambdas00[idx]
        # idx = (target_i.flatten() == 1.) & (target_j.flatten() == 1.)
        # lambdas[idx] = lambdas11[idx]
        # idx = (target_i.flatten() == -1.) & (target_j.flatten() == 1.)
        # lambdas[idx] = lambdas01[idx]
        # idx = (target_i.flatten() == 1.) & (target_j.flatten() == -1.)
        # lambdas[idx] = lambdas01[idx]

        pp=self.k
        if pp == 2:
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat(
                (torch.ones_like(anchor_s) * -1, torch.ones_like(anchor_s) * -1, torch.ones_like(neg_s)), 0)

            indices_j = torch.randperm(x_i.size(0)).to(self.device)
            x_j = x_i[indices_j]
            target_j = target_i[indices_j]

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas = Beta.sample(target_i.flatten().shape).to(self.device)[:, 1]

            x_tilde = x_i * lambdas.view(lambdas.size(0), 1) + x_j * (1 - lambdas.view(lambdas.size(0), 1))
            _, score_tilde = basenet(x_tilde)

            _, score_xi = basenet(x_i)
            _, score_xj = basenet(x_j)

            score_mix = score_xi * lambdas.view(lambdas.size(0), 1) + score_xj * (1 - lambdas.view(lambdas.size(0), 1))
            y_tilde   = target_i * lambdas.view(lambdas.size(0), 1) + target_j * (1 - lambdas.view(lambdas.size(0), 1))
            loss_out = self.loss_reg(score_tilde, y_tilde)
            loss_intra = self.loss_reg(score_tilde, score_mix)

            loss_score = loss_out + loss_intra
            loss_score = loss_score.mean()
            loss_out = loss_out.mean()
            loss_intra = loss_intra.mean()

        else:
            # # # ----------------------- n-samples mixup --------------------------- #
            x_i = torch.cat((anchor, pos, neg), 0)
            target_i = torch.cat((torch.ones_like(anchor_s)*-1, torch.ones_like(anchor_s)*-1, torch.ones_like(neg_s)), 0)
            _, score_xi = basenet(x_i)

            x_dup = [x_i]
            target_dup = [target_i]
            score_dup = [score_xi]
            for k in range(1, pp):
                indices_j = torch.randperm(x_i.size(0)).to(self.device)
                x_j = x_i[indices_j]
                target_j = target_i[indices_j]
                _, score_xj = basenet(x_j)

                x_dup.append(x_j)
                target_dup.append(target_j)
                score_dup.append(score_xj)

            Beta = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha, self.alpha]))
            lambdas_dup = Beta.sample((target_i.flatten().shape[0], pp)).to(self.device)[:, :, 1]

            s = torch.sum(lambdas_dup, 1).unsqueeze(0).T.repeat(1, pp)
            lambdas_dup = lambdas_dup / s

            x_tilde = lambdas_dup[:, 0].unsqueeze(0).T * x_i
            y_tilde = lambdas_dup[:, 0].unsqueeze(0).T * target_i
            score_mix = lambdas_dup[:, 0].unsqueeze(0).T * score_xi
            for k in range(1, pp):
                x_tilde += lambdas_dup[:, k].unsqueeze(0).T * x_dup[k]
                y_tilde += lambdas_dup[:, k].unsqueeze(0).T * target_dup[k]
                score_mix += lambdas_dup[:, k].unsqueeze(0).T * score_dup[k]

            _, score_tilde = basenet(x_tilde)

            loss_out = self.loss_reg(score_tilde, y_tilde)
            loss_intra = self.loss_reg(score_tilde, score_mix)

            loss_score = loss_out + loss_intra
            loss_score = loss_score.mean()
            loss_out = loss_out.mean()
            loss_intra = loss_intra.mean()

        # # from matplotlib import pyplot as plt
        # # yy = y_tilde.flatten().data.cpu().numpy()
        # # plt.hist(yy, bins=20)
        # # plt.show()

        k1 = torch.exp((loss_emb / pre_emb_loss) / self.T) if pre_emb_loss != 0 else 0
        k2 = torch.exp((loss_score / pre_score_loss) / self.T) if pre_score_loss != 0 else 0
        loss = (k1 / (k1 + k2)) * loss_emb + (k2 / (k1 + k2)) * loss_score + self.l2_reg_weight * l2_reg

        return loss, loss_emb, loss_score, loss_out, loss_intra



class DataGenerator:
    def __init__(self, x, y, batch_size=256):
        self.x = x
        self.y = y

        self.anom_idx = np.where(self.y == 1)[0]
        self.anom_x = self.x[self.anom_idx]
        self.norm_idx = np.where(self.y == 0)[0]
        self.norm_x = self.x[self.norm_idx]

        self.batch_size = batch_size
        return

    def load_batches(self, n_batches=10):
        batch_set = []

        for i in range(n_batches):
            anom_idx = np.random.choice(len(self.anom_x), self.batch_size)
            anchor_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)
            pos_idx = np.random.choice(len(self.norm_x), self.batch_size, replace=False)

            batch = [[self.norm_x[a], self.norm_x[p], self.anom_x[n]] for a, p, n in zip(anchor_idx, pos_idx, anom_idx)]
            batch_set.append(batch)
        return np.array(batch_set)

