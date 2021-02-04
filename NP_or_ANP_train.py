from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.NP import NeuralProcess as NP
from module.utils import compute_loss, comput_kl_loss, to_numpy, load_plot_data
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

def validation(data_test, model, test_batch = 64, mode='GP'):
    total_ll = 0
    model.eval()
    for i in range(test_batch):
        if mode == 'GP':
            data = data_test.generate_curves(include_context=False)
            (x_context, y_context), x_target = data.query
        else:
            for _, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
                (x_context, y_context), x_target = data.query
        (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)

def save_plot(epoch, data, model):
    ax, fig = plt.subplots()
    (x_context, y_context), x_target = data.query
    x_grid = torch.arange(-2, 2, 0.01)[None, :, None].repeat([x_context.shape[0], 1, 1]).to(device)
    (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_grid.to(device))
    # plot scatter:
    plt.scatter(to_numpy(x_context[0]), to_numpy(y_context[0]), label = 'context points', c = 'red', s = 15)
    # plot sampled function:
    plt.scatter(to_numpy(x_target[0]), to_numpy(data.y_target[0]), label = 'target points', marker='x', color = 'k')
    # plot predicted function:
    plt.plot(to_numpy(x_grid[0]), to_numpy(mean[0]), label = '%s predicted mean'%MODELNAME, c = 'blue')
    # mu +/- 1.97* sigma: 97.5% confidence
    plt.fill_between(to_numpy(x_grid[0,:,0]), to_numpy(mean[0,:,0] - 1.97*var[0,:,0]), to_numpy(mean[0, :, 0] + 1.97*var[0, :, 0]), color ='blue', alpha = 0.15)
    plt.legend(loc = 'upper right')
    plt.title("epoch:%d"%epoch)
    plt.ylim(-0.5, 1.7)
    model_path = "saved_fig/" + MODELNAME
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/"+MODELNAME+"/"+kernel+"/"+"%04d"%(epoch//100)+".png")
    plt.close()
    return fig

def main_GP():
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINT = 50
    VAL_AFTER = 1e3
    BEST_LOSS = -numpy.inf
    MODELNAME = 'ANP'  # 'NP' or 'ANP'
    kernel = 'period'  # EQ or period
    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    # writer = SummaryWriter('runs/' + kernel + '_' + MODELNAME + '_' + time_stamp)
    # load data set
    dataset = GPCurvesReader(kernel=kernel, batch_size=64, max_num_context=MAX_CONTEXT_POINT, device=device)
    # data for recording training progress
    # plot_data = load_plot_data(kernel)

    np = NP(input_dim=1, latent_dim=128, output_dim=1, use_attention=MODELNAME == 'ANP').to(device)
    optim = torch.optim.Adam(np.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        data = dataset.generate_curves()
        (x_context, y_context), x_target = data.query
        (mean, var), prior, poster = np(x_context.to(device), y_context.to(device), x_target.to(device),
                                        data.y_target.to(device))
        nll_loss = compute_loss(mean, var, data.y_target.to(device))
        kl_loss = comput_kl_loss(prior, poster)
        loss = nll_loss + kl_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        # writer.add_scalars("Log-likelihood", {"train/overall": -loss.item(),
        #                                       "train/ll": -nll_loss.item(),
        #                                       "train/kl": kl_loss.item()}, epoch)
        if (epoch % 100 == 0 and epoch < VAL_AFTER) or epoch % VAL_AFTER == 0:
            val_loss = validation(dataset, np)
            # save_plot(epoch, plot_data, np)
            # writer.add_scalars("Log-likelihood", {"val": val_loss}, epoch)
            if val_loss > BEST_LOSS:
                BEST_LOSS = val_loss
                print("save module at epoch: %d, val log-likelihood: %.4f" % (epoch, val_loss))
                torch.save(np.state_dict(), 'saved_model/' + kernel + '_' + MODELNAME + '.pt')
    # writer.close()
    print("finished training: " + MODELNAME)

def main_realworld():
    # define hyper parameters
    dataname = 'NIFTY50'  # EQ or period
    MODELNAME = 'ANP'  # 'NP' or 'ANP'
    TRAINING_ITERATIONS = int(2e4)
    MAX_CONTEXT_POINT = 50
    VAL_AFTER = 1e2
    BEST_LOSS = -numpy.inf

    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    # writer = SummaryWriter('runs/' + dataname + '_ConvCNP_' + time_stamp)

    # load data set
    dataset = NIFTYReader(batch_size=50, max_num_context=MAX_CONTEXT_POINT, device=device)
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    np = NP(input_dim=1, latent_dim=128, output_dim=1, use_attention=MODELNAME == 'ANP').to(device)
    optim = torch.optim.Adam(np.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        for i, data in enumerate(train_loader):  # 50 stocks per epoch, 1 batch is enough
            (x_context, y_context), x_target = data.query
        # y_context_norm, y_mean, y_std = normalize(y_context)
        # y_target_norm, _, _ = normalize(data.y_target, y_mean, y_std)
        (mean, var), prior, poster = np(x_context.to(device), y_context.to(device),
                                        x_target.to(device), data.y_target.to(device))
        nll_loss = compute_loss(mean, var, data.y_target.to(device))
        kl_loss = comput_kl_loss(prior, poster)
        loss = nll_loss + kl_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        # writer.add_scalars("Log-likelihood", {"train": -loss.item()}, epoch)
        # print("epoch: %d,  training log-liklihood: %.4f" % (epoch, -loss.item()))
        if (epoch % 50 == 0 and epoch < VAL_AFTER) or epoch % VAL_AFTER == 0:
            val_loss = validation(val_loader, np, test_batch=1, mode="NIFTY")
            # save_plot(epoch, plot_data, cnp)  # save training process, optional
            # writer.add_scalars("Log-likelihood", {"val": val_loss}, epoch)
            if val_loss > BEST_LOSS:
                BEST_LOSS = val_loss
                print("save module at epoch: %d, val log-likelihood: %.4f, training loss:%.4f" %
                      (epoch, val_loss, -loss))
                torch.save(np.state_dict(), 'saved_model/' + dataname + '_' + MODELNAME + '.pt')
    # writer.close()
    print("finished training " + MODELNAME + " !"+ dataname)


if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    main_GP()
    # main_realworld()



