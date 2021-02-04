import torch
from data.GP_data_sampler import GPCurvesReader
from data.NIFTY_data_sampler import NIFTYReader
from module.utils import to_numpy, normalize
from module.CNP import ConditionalNeuralProcess as CNP
from module.NP import NeuralProcess as NP
from module.convCNP import ConvCNP, UNet
from module.NP_PROV import ConvCNP as NP_PROV, UNet as PROV_UNet
import matplotlib.pyplot as plt


class Comparison:
    def __init__(self, kernel, device):
        self.kernel = kernel
        self.device = device
        self.load_model()

    def load_model(self):
        kernel = self.kernel
        device = self.device

        cnp = CNP(input_dim=1, latent_dim=128, output_dim=1).to(device)
        # cnp.load_state_dict(torch.load('saved_model/' + kernel + '_CNP.pt', map_location=device))
        self.cnp = cnp

        # load module parameters
        MODELNAME = 'NP'
        np = NP(input_dim=1, latent_dim=128, output_dim=1, use_attention=MODELNAME == 'ANP').to(device)
        # np.load_state_dict(torch.load('saved_model/' + kernel + '_' + MODELNAME + '.pt' , map_location=device))
        self.np = np

        # load module parameters
        MODELNAME = 'ANP'
        anp = NP(input_dim=1, latent_dim=128, output_dim=1, use_attention=MODELNAME == 'ANP').to(device)
        # anp.load_state_dict(torch.load('saved_model/' + kernel + '_' + MODELNAME + '.pt', map_location=device))
        self.anp = anp

        convcnp = ConvCNP(rho=UNet(), points_per_unit=64 if kernel != 'NIFTY50' else 32, device=device).to(device)
        convcnp.load_state_dict(torch.load('saved_model/' + kernel + '_ConvCNP.pt', map_location=device))
        self.convcnp = convcnp

        npprov = NP_PROV(rho=PROV_UNet(), points_per_unit=64 if kernel != 'NIFTY50' else 32, device=device).to(device)
        npprov.load_state_dict(torch.load('saved_model/' + kernel + '_NP_PROV.pt', map_location=device))
        self.npprov = npprov
        self.modellist = [self.cnp, self.np, self.anp, self.convcnp, self.npprov]
        self.colorlist = ['limegreen','royalblue','orange', 'purple', 'green' ]
        self.namelist = ['CNP', 'NP', 'ANP', 'ConvCNP', 'SCANP']
        # self.cololist = ['purple', 'limegreen', 'chocolate', 'cornflowerblue', 'deeppink']

    def plot_sample(self, x_context, y_context, x_target, y_target, xlim=(-2, 2), len_legend=2, scale=1):
        fig = plt.figure(figsize=(30, 20))
        ax = plt.subplot()
        # Plot context set and model predictions.
        plt.scatter(to_numpy(x_context), to_numpy(y_context), label='Context', color='red', s= 500, zorder = 2)
        plt.scatter(to_numpy(x_target), to_numpy(y_target), label='Target', marker='P', c='k',  s = 500, zorder = 2)
        y_context_norm, mu, std = normalize(y_context)
        y_target_norm, _, _ = normalize(y_target, mu, std)
        # plt.scatter(to_numpy(x_context), to_numpy(y_context_norm), label='Normalized context',  marker='P', color='olive', s=500, alpha=0.7)
        # plt.scatter(to_numpy(x_target), to_numpy(y_target_norm), label='Normalized target', c='m',alpha=0.7, s=500)
        # plt.vlines([-2, 2], -3, 3, linestyles='dashed')
        # plt.ylim(-3., 13)
        # plt.xlim(xlim[0], xlim[1])
        x_all = torch.linspace(xlim[0], xlim[-1], 200)[None, :, None]

        for i, model in enumerate(self.modellist):
            name = self.namelist[i]
            if name not in ['ConvCNP', 'SCANP']:
                continue
            color = self.colorlist[i]
            with torch.no_grad():
                # Make predictions with model.
                if name not in ['NP', 'ANP']:
                    y_mean, y_std = model(x_context.to(device), y_context_norm.to(device), x_all.to(device))
                else:
                    (y_mean, y_std), _, _ = model(x_context.to(device), y_context_norm.to(device), x_all.to(device))

            # Plot model predictions.
            # plt.plot(to_numpy(x_all.squeeze()), to_numpy(y_mean.squeeze()), linestyle = '--', label='Model: normed %s' % name,
            #          color= 'm' if name == 'ConvCNP' else 'olive',  linewidth =7, zorder =1)
            # plt.fill_between(to_numpy(x_all.squeeze()),
            #                  to_numpy(y_mean.squeeze() + 2 * y_std.squeeze()),
            #                  to_numpy(y_mean.squeeze() - 2 * y_std.squeeze()),
            #                  color= 'm' if name == 'ConvCNP' else 'olive', alpha=0.1)
            #
            y_all_mean = y_mean*std + mu
            y_all_std = y_std *std
            plt.plot(to_numpy(x_all.squeeze()), to_numpy(y_all_mean.squeeze()), label='Model: %s' % name, color=color, linewidth =7,
                     zorder = 1)
            plt.fill_between(to_numpy(x_all.squeeze()),
                             to_numpy(y_all_mean.squeeze() + 2 * y_all_std.squeeze()),
                             to_numpy(y_all_mean.squeeze() - 2 * y_all_std.squeeze()),
                             color=color, alpha=0.2)
        #     plt.axis('off')
        # plt.legend()
        plt.xlabel("Location", size=60)
        plt.ylabel("Function value", size=60)
        plt.legend(ncol=len_legend, prop={'size': 50})
        plt.grid("on", linewidth=3)
        ax.tick_params(axis="x", labelsize=50)
        ax.tick_params(axis="y", labelsize=50)
        plt.savefig("saved_fig/"+self.kernel+"_"+str(scale)+".png")
        plt.show()


def save_data(data):
    import pandas as pd
    import numpy as np
    (x_context, y_context), x_target = data.query
    x_context = to_numpy(x_context[0])
    y_context = to_numpy(y_context[0])
    x_target = to_numpy(x_target[0])
    y_target = to_numpy(data.y_target[0])
    context = pd.DataFrame(np.concatenate([x_context, y_context], axis=1))
    target = pd.DataFrame(np.concatenate([x_target, y_target], axis=1))
    context.to_csv("context.csv", index=False, header=False)
    target.to_csv("target.csv", index=False, header=False)
    print("saving succeed!")



def load_data():
    import pandas as pd
    import numpy as np
    import collections

    NPRegressionDescription = collections.namedtuple(
        "NPRegressionDescription",
        ("query", "y_target", "num_total_points", "num_context_points"))

    scale = 7
    context = pd.read_csv("context.csv", header=None).values
    target = pd.read_csv("target.csv", header=None).values
    x_context = torch.from_numpy(context[:,0]).float()[None, :, None].to(device)
    y_context = torch.from_numpy(context[:,1]*scale).float()[None, :, None].to(device)
    x_target = torch.from_numpy(target[:,0]).float()[None, :, None].to(device)
    y_target = torch.from_numpy(target[:,1]*scale).float()[None, :, None].to(device)
    query = ((x_context, y_context), x_target)
    data = NPRegressionDescription(query=query,
            y_target=y_target,
            num_total_points= 100,
            num_context_points=50)
    return data, scale


def main_GP():
    kernel = 'EQ'  # EQ or period or NIFTY50
    models = Comparison(kernel, device)
    # dataset = GPCurvesReader(kernel, batch_size=1, max_num_context=50, device=device)
    # data = dataset.generate_curves(include_context=False)
    # save_data(data)
    data, scale = load_data()
    (x_context, y_context), x_target = data.query

    models.plot_sample(x_context, y_context, x_target, data.y_target, scale=scale)

def main_realworld():
    kernel = 'NIFTY50'  # EQ or period or NIFTY50
    models = Comparison(kernel, device)
    dataset = NIFTYReader(batch_size=1, max_num_context=50, device=device)
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    for i, data in enumerate(test_loader):  # 50 stocks per epoch, 1 batch is enough
        (x_context, y_context), x_target = data.query
    x_min = min(torch.min(x_context).cpu().numpy(),
                torch.min(x_target).cpu().numpy(), 0.) - 0.1
    x_max = max(torch.max(x_context).cpu().numpy(),
                torch.max(x_target).cpu().numpy(), 2.) + 0.1
    models.plot_sample(x_context.to(device), y_context.to(device), x_target.to(device), data.y_target.to(device), xlim=(x_min, x_max),
                       scale=4)



if __name__ == '__main__':
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    main_GP()
    # main_realworld()