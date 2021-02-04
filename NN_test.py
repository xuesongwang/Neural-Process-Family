from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.NeuralNetwork import NeuralNetwork as NN
from module.utils import compute_loss, to_numpy, compute_MSE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def testing(data_test, model, test_batch = 64, mode ='GP'):
    total_ll = 0
    total_mse = 0
    model.eval()
    for i in tqdm(range(test_batch)):
        if mode == 'GP':
            data = data_test.generate_curves(include_context=False)
            (x_context, y_context), x_target = data.query
        else:
            for _, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
                (x_context, y_context), x_target = data.query
        mean, var = model(x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        mse_loss = compute_MSE(mean, data.y_target.to(device))
        total_ll += -loss.item()
        total_mse += mse_loss.item()
    return total_ll / (i+1), total_mse / (i+1)

def plot_sample(dataset, model):
    ax, fig = plt.subplots()
    # load test data set
    data = dataset.generate_curves(include_context=False)
    (x_context, y_context), x_target = data.query
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
    plt.legend()
    plt.savefig("saved_fig/CNP_"+kernel+".png")
    plt.show()
    return fig




def main_GP():
    # define hyper parameters
    TESTING_ITERATIONS = int(1024)
    MAX_CONTEXT_POINT = 50
    kernel = 'matern'  # EQ or period

    # load data set
    dataset = GPCurvesReader(kernel=kernel, batch_size=64, max_num_context=MAX_CONTEXT_POINT, device=device)
    # load module parameters
    nn = NN(input_dim=1, latent_dim=128, output_dim=1).to(device)
    nn.load_state_dict(torch.load('saved_model/' + kernel + '_NN.pt', map_location=device))
    print("successfully load NN module!")
    total_loss = []
    total_mse = []
    for _ in range(6):
        test_loss, test_mse = testing(dataset, nn, TESTING_ITERATIONS)
        total_loss.append(test_loss)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" %(np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    # test_ll, test_mse = testing(dataset, cnp, TESTING_ITERATIONS)
    # print("CNP loglikelihood on 1024 samples: %.4f, mse: %.4f" % (test_ll, test_mse))

    # fig = plot_sample(dataset, cnp)
    # print("save plots!")


def main_realworld():
    # define hyper parameters
    dataname = 'NIFTY50'  # EQ or period
    MAX_CONTEXT_POINT = 50
    TESTING_ITERATIONS = int(1024)
    # load data set
    dataset = NIFTYReader(batch_size=50, max_num_context=MAX_CONTEXT_POINT, device=device)
    test_loader = dataset.test_dataloader()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    nn = NN(input_dim=1, latent_dim=128, output_dim=1).to(device)
    nn.load_state_dict(torch.load('saved_model/'+dataname+'_NN.pt'))
    print("successfully load %s module!" % dataname)

    total_loss = []
    total_mse = []
    for _ in range(6):
        test_loss, test_mse = testing(test_loader, nn, TESTING_ITERATIONS, mode='NIFTY')
        total_loss.append(test_loss)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))

    # test_ll, test_mse = testing(test_loader, nn, TESTING_ITERATIONS, mode='NIFTY')

    # writer.close()
    # print("NN loglikelihood on 1024 samples: %.4f, mse: %.4f" % (test_ll, test_mse))




if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    main_realworld()
    # main_GP()