from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.utils import compute_loss, to_numpy, compute_MSE, to_tensor
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import stheno.torch as stheno
import stheno as sth

def testing(data_test, gp, test_batch = 64, mode ='GP'):
    import matplotlib.pyplot as plt
    total_ll = 0
    total_mse = 0
    for i in tqdm(range(test_batch)):
        if mode == 'GP':
            # data = data_test.generate_curves(include_context=False)
            # (x_context, y_context), x_target = data.query
            num_context = np.random.randint(3, 50)
            num_target = np.random.randint(3, 50)
            num_points = 200
            x_all = np.linspace(-2., 2., num_points)
            rand_indices = np.random.permutation(num_points)
            y_all = gp(x_all).sample()
            x_context = x_all[rand_indices][:num_context]
            y_context = y_all[rand_indices][:num_context]
            x_target = x_all[rand_indices][-num_target:]
            y_target = y_all[rand_indices][-num_target:]
        else:
            for _, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
                (x_context, y_context), x_target = data.query
            x_context = to_numpy(x_context[0, :, 0])
            y_context = to_numpy(y_context[0, :, 0])
            x_target = to_numpy(x_target[0, :, 0])
            y_target = to_numpy(data.y_target[0, :, 0])
        post = gp | (x_context, y_context)
        mean, lower, upper = post(x_target).marginals()
        var = (upper - lower)/4
        # plt.scatter(x_context, y_context[:,0], s=20, c='r', marker='x')
        # plt.scatter(x_target, y_target[:,0], s=20, c='b', marker='o')
        # plt.scatter(x_target, mean, c = 'green', marker='^')
        # plt.scatter(x_target, mean + 2*var, c='k', marker='8')
        # plt.scatter(x_target, mean - 2*var, c='k', marker='8')
        # plt.show()
        # break
        loss = compute_loss(to_tensor(mean).unsqueeze(1), to_tensor(var).unsqueeze(1), to_tensor(y_target))
        mse_loss = compute_MSE(to_tensor(mean), to_tensor(y_target))
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
    if kernel == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif kernel == 'EQ':
        kernel = stheno.EQ().stretch(0.4)
    elif kernel == 'period':
        kernel = stheno.EQ().periodic(period=0.25)

    gp = stheno.GP(kernel, graph = sth.Graph())


    total_loss = []
    total_mse = []
    for _ in range(6):
        test_ll, test_mse = testing(dataset, gp, TESTING_ITERATIONS)
        total_loss.append(test_ll)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" %(np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    # test_ll, test_mse = testing(dataset, gp, TESTING_ITERATIONS)
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

    kernel = stheno.EQ().periodic(period=0.25) * stheno.EQ().stretch(0.4)
    gp = stheno.GP(kernel, graph=sth.Graph())
    total_loss = []
    total_mse = []
    for _ in range(6):
        test_ll, test_mse = testing(test_loader, gp, TESTING_ITERATIONS, mode='NIFTY')
        total_loss.append(test_ll)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    # test_ll, test_mse = testing(test_loader, gp, TESTING_ITERATIONS, mode='NIFTY')
    # writer.close()
    # print("GP loglikelihood on 1024 samples: %.4f, mse: %.4f" % (test_ll, test_mse))




if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    main_realworld()
    # main_GP()