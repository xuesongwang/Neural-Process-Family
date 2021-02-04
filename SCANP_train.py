from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.SCANP import SCANP, UNet
from module.utils import compute_loss, to_numpy, load_plot_data, normalize
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

def validation(data_test, model, test_batch = 64, mode='gp'):
    total_ll = 0
    total_ll_unnorm = 0
    model.eval()
    for i in range(test_batch):
        if mode == "gp":
            data = data_test.generate_curves(include_context=False)
            (x_context, y_context), x_target = data.query
            y_context_norm, y_mean, y_std = normalize(y_context)
        else:
            for i, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
                (x_context, y_context), x_target = data.query
            y_context_norm, y_mean, y_std = normalize(y_context)
            y_context_norm = y_context
            y_target_norm = data.y_target
            y_target_norm, _, _ = normalize(data.y_target, y_mean, y_std)
        mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var,  data.y_target.to(device))
        unnormalized_loss = compute_loss(mean * y_std.to(device) + y_mean.to(device), var * y_std.to(device), data.y_target.to(device))
        total_ll += -loss.item()
        total_ll_unnorm += -unnormalized_loss.item()
    return total_ll / (i+1), total_ll_unnorm/(i+1)

def save_plot(epoch, data, model):
    ax, fig = plt.subplots()
    (x_context, y_context), x_target = data.query
    x_grid = torch.arange(-2, 2, 0.01)[None, :, None].repeat([x_context.shape[0], 1, 1]).to(device)
    mean, var = model(x_context.to(device), y_context.to(device), x_grid.to(device))
    # plot scatter:
    plt.scatter(to_numpy(x_context[0]), to_numpy(y_context[0]), label = 'context points', c = 'red', s = 15)
    # plot sampled function:
    plt.scatter(to_numpy(x_target[0]), to_numpy(data.y_target[0]), label = 'target points', marker='x', color = 'k')
    # plot predicted function:
    plt.plot(to_numpy(x_grid[0]), to_numpy(mean[0]), label = 'ConvCNP predicted mean', c = 'blue')
    # mu +/- 1.97* sigma: 97.5% confidence
    plt.fill_between(to_numpy(x_grid[0,:,0]), to_numpy(mean[0,:,0] - 1.97*var[0,:,0]), to_numpy(mean[0, :, 0] + 1.97*var[0, :, 0]), color ='blue', alpha = 0.15)
    plt.legend(loc = 'upper right')
    plt.title("epoch:%d"%epoch)
    plt.ylim(-0.5, 1.7)
    model_path = "saved_fig/ConvCNP"
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/ConvCNP/"+kernel+"/"+"%04d"%(epoch//100)+".png")
    plt.close()
    return fig

def main_GP():
    # define hyper parameters
    # device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINT = 50
    VAL_AFTER = 1e3
    BEST_LOSS = -np.inf
    MODELNAME = 'SCANP'
    kernel = 'EQ'  # EQ or period

    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    # writer = SummaryWriter('runs/'+kernel+'_' + MODELNAME +'_'+ time_stamp)
    # load data set
    dataset = GPCurvesReader(kernel=kernel, batch_size=64, max_num_context= MAX_CONTEXT_POINT, device=device)
    # data for recording training progress
    # plot_data = load_plot_data(kernel)

    convcnp = SCANP(rho=UNet(), points_per_unit=64, device = device).to(device)
    optim = torch.optim.Adam(convcnp.parameters(), lr=3e-4, weight_decay=1e-5)
    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        data = dataset.generate_curves(include_context=False)
        (x_context, y_context), x_target = data.query
        mean, var = convcnp(x_context, y_context, x_target)
        loss = compute_loss(mean, var,  data.y_target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # print("normalized loss:%.4f, unnormalized_loss:%.4f, scaled_mean loss: %.4f"%(-loss.item(), -unnormalized_loss.item()))
        # writer.add_scalars("Log-likelihood", {"train": -loss.item()}, epoch)
        if (epoch % 100 == 0 and epoch<VAL_AFTER) or  epoch % VAL_AFTER == 0:
            val_loss, unnormed_val_loss = validation(dataset, convcnp)
            # save_plot(epoch, plot_data, convcnp)  # save training process, optional
            # writer.add_scalars("Log-likelihood", {"val": val_loss}, epoch)
            if val_loss > BEST_LOSS:
                BEST_LOSS = val_loss
                print("save module at epoch: %d, val log-likelihood: %.4f, raw NLL:%.4f" %(epoch, val_loss, unnormed_val_loss))
                torch.save(convcnp.state_dict(), 'saved_model/'+kernel+'_' + MODELNAME+'_znorm.pt')
    # writer.close()
    print("finished training: " + MODELNAME)


def main_realword():
    # define hyper parameters
    dataname = 'NIFTY50'  # EQ or period
    TRAINING_ITERATIONS = int(2e4)
    MAX_CONTEXT_POINT = 50
    VAL_AFTER = 1e2
    BEST_LOSS = -np.inf

    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    # writer = SummaryWriter('runs/' + dataname + '_ConvCNP_' + time_stamp)

    # load data set
    dataset = NIFTYReader(batch_size=50, max_num_context=MAX_CONTEXT_POINT, device=device)
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    convcnp = SCANP(rho=UNet(), points_per_unit=32, device=device).to(device)
    optim = torch.optim.Adam(convcnp.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        for i, data in enumerate(train_loader): # 50 stocks per epoch, 1 batch is enough
            (x_context, y_context), x_target = data.query
        # y_context_norm, y_mean, y_std = normalize(y_context)
        # y_target_norm, _, _ = normalize(data.y_target, y_mean, y_std)
        y_context_norm = y_context
        y_target_norm = data.y_target
        mean, var = convcnp(x_context.to(device), y_context_norm.to(device), x_target.to(device))
        loss = compute_loss(mean, var, y_target_norm.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        # writer.add_scalars("Log-likelihood", {"train": -loss.item()}, epoch)
        # print("epoch: %d,  training log-liklihood: %.4f" % (epoch, -loss.item()))
        if (epoch % 50 == 0 and epoch < VAL_AFTER) or epoch % VAL_AFTER == 0:
            val_loss, unnormed_val_loss = validation(val_loader, convcnp, test_batch=1, mode="NIFTY")
            # save_plot(epoch, plot_data, cnp)  # save training process, optional
            # writer.add_scalars("Log-likelihood", {"val": val_loss}, epoch)
            if val_loss > BEST_LOSS:
                BEST_LOSS = val_loss
                print("save module at epoch: %d, val log-likelihood: %.4f, unnormed_loss:%.4f, training loss:%.4f" %
                      (epoch, val_loss, unnormed_val_loss,-loss))
                torch.save(convcnp.state_dict(), 'saved_model/'+dataname+'_ConvCNP.pt')
    # writer.close()
    print("finished training ConvCNP!" + dataname)

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    main_GP()
    # main_realword()

