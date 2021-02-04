from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.SCANP import SCANP, UNet, to_multiple
from module.utils import compute_loss, to_numpy, compute_mse_loss, normalize,compute_MSE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def testing(data_test, model, test_batch = 64, mode = 'GP'):
    model.eval()
    total_nll = 0
    total_mse = 0
    total_nll_unnorm = 0
    for i in tqdm(range(test_batch)):
        if mode == 'GP':
            data = data_test.generate_curves(include_context=False)
            (x_context, y_context), x_target = data.query
        else:
            for _, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
                (x_context, y_context), x_target = data.query

        y_context_norm, y_mean, y_std = normalize(y_context)
        mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        loss_unnorm = compute_loss(mean * y_std.to(device) + y_mean.to(device), var * y_std.to(device), data.y_target.to(device))
        mse_loss = compute_MSE(mean, data.y_target.to(device))
        total_nll += -loss.item()
        total_mse += mse_loss.item()
        total_nll_unnorm += -loss_unnorm.item()
    return total_nll/(i+1), total_mse / (i+1)

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
    plt.plot(to_numpy(x_grid[0]), to_numpy(mean[0]), label =  MODELNAME + ' predicted mean', c = 'blue')
    # mu +/- 1.97* sigma: 97.5% confidence
    plt.fill_between(to_numpy(x_grid[0,:,0]), to_numpy(mean[0,:,0] - 1.97*var[0,:,0]), to_numpy(mean[0, :, 0] + 1.97*var[0, :, 0]), color ='blue', alpha = 0.15)
    plt.legend()
    plt.savefig(MODELNAME+".png")
    plt.show()
    return fig

def test(data_test, model, test_batch = 64):
    total_ll = 0
    model.eval()
    for i in tqdm(range(test_batch)):
        for _, data in enumerate(data_test):  # 50 stocks per epoch, 1 batch is enough
            (x_context, y_context), x_target = data.query
        y_context_norm, y_mean, y_std = normalize(y_context)
        # y_target_norm, _, _ = normalize(data.y_target, y_mean, y_std)
        # y_context_norm = y_context
        # y_target_norm = data.y_target
        # y_target_norm, _, _ = normalize(data.y_target, y_mean, y_std)
        mean, var = model(x_context.to(device), y_context_norm.to(device), x_target.to(device))
        loss = compute_loss(mean * y_std.to(device) + y_mean.to(device), var * y_std.to(device),
                                         data.y_target.to(device))
        # loss = compute_loss(mean, var, data.y_target.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)

def main_GP():
    TESTING_ITERATIONS = int(1024)
    MAX_CONTEXT_POINT = 50
    MODELNAME = 'SCANP'
    kernel = 'EQ'  # EQ or period
    criterion = torch.nn.MSELoss()

    # load data set
    dataset = GPCurvesReader(kernel=kernel, batch_size=64, max_num_context=MAX_CONTEXT_POINT, device=device)
    scanp = SCANP(rho=UNet(), points_per_unit=64, device=device).to(device)
    scanp.load_state_dict(torch.load('saved_model/' + kernel + '_' + MODELNAME + '_kernel_nobatchnorm.pt', map_location=device))
    print("successfully load %s module!" % MODELNAME)

    # total_ll = []
    # total_mse = []
    # for _ in range(6):
    #     test_ll, test_mse = testing(dataset, convcnp, TESTING_ITERATIONS)
    #     total_ll.append(test_ll)
    #     total_mse.append(test_mse)
    # print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_ll), np.std(total_ll)))
    # print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    #
    test_ll, test_mse = testing(dataset, scanp, TESTING_ITERATIONS)
    print("NLL on 1024 samples:%.4f, NLL on raw samples:%.4f" % (test_ll, test_mse))

    # fig = plot_sample(dataset, convcnp)
    # print("save plots!")


def main_realworld():
    # define hyper parameters
    dataname = 'NIFTY50'  # EQ or period
    MODELNAME = 'ConvCNP'
    TESTING_ITERATIONS = int(1024)
    MAX_CONTEXT_POINT = 50

    # load data set
    dataset = NIFTYReader(batch_size=50, max_num_context=MAX_CONTEXT_POINT, device=device)
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    convcnp = ConvCNP(rho=UNet(), points_per_unit=32, device=device).to(device)
    convcnp.load_state_dict(torch.load('saved_model/' + dataname + '_' + MODELNAME + '.pt'))
    print("successfully load %s module!" % MODELNAME)

    total_ll = []
    total_mse = []
    for _ in range(6):
        test_ll, test_mse = testing(test_loader, convcnp, TESTING_ITERATIONS, mode='NIFTY')
        total_ll.append(test_ll)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_ll), np.std(total_ll)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))

    # test_ll, test_mse = testing(test_loader, convcnp, TESTING_ITERATIONS, mode='NIFTY')

    # writer.close()
    # print("ConvCNP loglikelihood on 1024 samples: %.4f, %.4f" % (test_ll, test_mse))



if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    # main_realworld()
    main_GP()

