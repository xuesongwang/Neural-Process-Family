from data.GP_data_sampler import GPCurvesReader
from module.NP import NeuralProcess as NP
from module.utils import compute_loss, to_numpy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def testing(data_test, model, test_batch = 64):
    total_ll = 0
    model.eval()
    for i in tqdm(range(test_batch)):
        data = data_test.generate_curves(include_context=False)
        (x_context, y_context), x_target = data.query
        (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, data.y_target.to(device))
        total_ll += -loss.item()
    return total_ll/(i+1)


def plot_sample(dataset, model):
    ax, fig = plt.subplots()
    # load test data set
    data = dataset.generate_curves(include_context=False)
    (x_context, y_context), x_target = data.query
    x_grid = torch.arange(-2, 2, 0.01)[None, :, None].repeat([x_context.shape[0], 1, 1]).to(device)
    (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_grid.to(device))
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

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TESTING_ITERATIONS = int(1024)
    MAX_CONTEXT_POINT = 50
    MODELNAME = 'ANP' # 'NP' or 'ANP'
    kernel = 'EQ'  # EQ or period
    # set up tensorboard
    # load data set
    dataset = GPCurvesReader(kernel=kernel, batch_size=64, max_num_context= MAX_CONTEXT_POINT, device=device)

    # load module parameters
    np = NP(input_dim=1, latent_dim = 128, output_dim=1, use_attention= MODELNAME=='ANP').to(device)
    np.load_state_dict(torch.load('saved_model/'+kernel+'_' + MODELNAME+'.pt'))
    print("successfully load %s module!" %MODELNAME)

    # total_loss = []
    # for _ in range(10):
    #     test_loss = testing(dataset, np, TESTING_ITERATIONS)
    #     total_loss.append(test_loss)
    # print("for 10 runs, mean: %.4f, std:%.4f" % (numpy.mean(total_loss), numpy.std(total_loss)))

    test_loss = testing(dataset, np, TESTING_ITERATIONS)
    print ("loglikelihood on 1024 samples: %.4f"%(test_loss))

    # fig = plot_sample(dataset, np)
    # print("save plots!")






