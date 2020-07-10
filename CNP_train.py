import os
from data.GP_data_sampler import GPCurvesReader
from module.CNP import ConditionalNeuralProcess as CNP
from module.utils import compute_loss, to_numpy, load_plot_data, save_plot_data
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def validation(data_test, model, test_batch = 64):
    total_ll = 0
    model.eval()
    for i in range(test_batch):
        data = data_test.generate_curves(include_context=False)
        (x_context, y_context), x_target = data.query
        mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)


def save_plot(epoch, data, model):
    ax, fig = plt.subplots()
    (x_context, y_context) , x_target = data.query
    x_grid = torch.arange(-2, 2, 0.01)[None, :, None].repeat([x_context.shape[0], 1, 1]).to(device)
    mean, var = model(x_context.to(device), y_context.to(device), x_grid.to(device))
    # plot scatter:
    plt.scatter(to_numpy(x_context[0]), to_numpy(y_context[0]), label = 'context points', c = 'red', s = 15)
    # plot sampled function:
    plt.scatter(to_numpy(x_target[0]), to_numpy(data.y_target[0]), label = 'target points', marker='x', color = 'k')
    # plot predicted function:
    plt.plot(to_numpy(x_grid[0]), to_numpy(mean[0]), label = 'CNP predicted mean', c = 'blue')
    # mu +/- 1.97* sigma: 97.5% confidence
    plt.fill_between(to_numpy(x_grid[0,:,0]), to_numpy(mean[0,:,0] - 1.97*var[0,:,0]), to_numpy(mean[0, :, 0] + 1.97*var[0, :, 0]), color ='blue', alpha = 0.15)
    plt.legend(loc = 'upper right')
    plt.title("epoch:%d"%epoch)
    plt.ylim(-0.5, 1.7)
    # save plot
    model_path = "saved_fig/CNP"
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/CNP/"+kernel+"/"+"%04d"%(epoch//100)+".png")
    plt.close()
    return fig

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kernel = 'period' # EQ or period
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINT = 50
    VAL_AFTER = 1e3
    BEST_LOSS = -np.inf

    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    writer = SummaryWriter('runs/'+kernel+'_CNP_'+ time_stamp)

    # load data set
    dataset = GPCurvesReader(kernel = kernel, batch_size=64, max_num_context= MAX_CONTEXT_POINT, device=device)
    # load plot dataset for recording training progress
    # generate and save data for the first run
    # plot_data = save_plot_data(dataset, kernel)
    plot_data = load_plot_data(kernel)

    cnp = CNP(input_dim=1, latent_dim = 128, output_dim=1).to(device)
    optim = torch.optim.Adam(cnp.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        data = dataset.generate_curves()
        (x_context, y_context), x_target = data.query
        mean, var = cnp(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        writer.add_scalars("Log-likelihood", {"train":-loss.item()}, epoch)
        if (epoch % 100 == 0 and epoch<VAL_AFTER) or  epoch % VAL_AFTER == 0:
            val_loss = validation(dataset, cnp)
            save_plot(epoch, plot_data, cnp) # save training process, optional
            writer.add_scalars("Log-likelihood", {"val": val_loss}, epoch)
            if val_loss > BEST_LOSS:
                BEST_LOSS = val_loss
                print("save module at epoch: %d, val log-likelihood: %.4f" %(epoch, val_loss))
                torch.save(cnp.state_dict(), 'saved_model/'+kernel+'_CNP.pt')
    writer.close()
    print("finished training CNP!"+kernel)


