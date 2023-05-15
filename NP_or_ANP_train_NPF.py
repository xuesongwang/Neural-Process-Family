# from data.GP_data_sampler import GPCurvesReader
from data.GP_sampler_NPF import gp_transport_task,collate_fns
from module.NP import NeuralProcess as NP
from module.utils import compute_loss, comput_kl_loss, to_numpy, load_plot_data
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import argparse


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

def save_plot(dataset, kernel, model, max_num_context = 50, max_num_extra_target=50, name='tau', MODELNAME='NP'):
    import numpy as np
    # randomly pick five samples
    # get the first five samples from the dataset. [5, n_points, x_dim/y_dim]

    # get context and target split
    batch_size = min(5, len(dataset))
    x, y, kernel_idx = dataset[: batch_size]

    # Sample a subset of random size
    num_context = np.random.randint(10, max_num_context)
    num_extra_target = np.random.randint(10, max_num_extra_target)

    inds = np.arange(x.shape[1])
    np.random.shuffle(inds)
    x_context = x[:, inds][:, :num_context]
    y_context = y[:, inds][:, :num_context]

    x_target = x[:, inds][:, num_context:]
    y_target = y[:, inds][:, num_context:]
    (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_target.to(device))

    # plot scatter:
    fig, axes = plt.subplots(nrows=batch_size, figsize=(20, 13))
    for i in range(batch_size):
        ax = axes[i]
        ax.scatter(to_numpy(x_context[i]), to_numpy(y_context[i]), label = 'context points', c = 'red', s = 15)
        # plot sampled function:
        ax.plot(to_numpy(x[i]), to_numpy(y[i]), label = 'ground truth', c='green', linestyle='--')

        target_idx = np.argsort(to_numpy(x_target[i,:,0]))
        x_target_order = x_target[i,target_idx,0]

        # plot prediction
        linecolor = '#F5A081'
        fill_color = '#6D90F1'
        # print("mean shape", mean.shape)
        # print("var shape",var.shape)
        mean_order = mean[:, i,target_idx, 0]
        var_order= var[:, i, target_idx, 0]

        # plot predicted function:
        # ax.plot(to_numpy(x_target_order), to_numpy(mean_order), label = '%s predicted mean'%kernel, c = 'blue')

        num_samples = mean.shape[0]
        # print("num of samples", num_samples)
        ax.plot(to_numpy(x_target_order), to_numpy(mean_order.T), linewidth=2, zorder=1, alpha=8 / num_samples,
                color=linecolor)

        # mu +/- 1.97* sigma: 97.5% confidence
        # ax.fill_between(to_numpy(x_target_order), to_numpy(mean_order - 1.97*var_order), to_numpy(mean_order + 1.97*var_order), color ='blue', alpha = 0.15)
        for sample_i in range(num_samples):
            # print("interval sample", mean_order[sample_i,:5], var_order[sample_i,:5])
            # print("upper bound", mean_order[sample_i,:5]+1.97* var_order[sample_i,:5])
            ax.fill_between(to_numpy(x_target_order),
                            to_numpy(mean_order[sample_i] - 1.97 * var_order[sample_i]),
                            to_numpy(mean_order[sample_i] + 1.97 * var_order[sample_i]),
                            alpha=2 / (10 * num_samples),
                            color=fill_color)
        ax.legend(loc = 'upper right')
    model_path = "saved_fig/pngs/"
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/pngs/"+kernel+"/"+MODELNAME+"_"+ name +".png")
    plt.close()
    # print("+++++++++++++")
    return fig

def val_model_loader(model, dataloader):
    model.eval()
    total_loss = 0
    for i, data in enumerate(dataloader):
        x_context, y_context, x_target, y_target, kernel_idx = data
        (mean, var), prior, poster = model(x_context.to(device), y_context.to(device), x_target.to(device))
        nll_loss = compute_loss(mean, var, y_target.to(device))
        loss = nll_loss
        total_loss += loss.item() * x_context.shape[0]
    data_size = len(dataloader.dataset)
    return total_loss/data_size


def train_model_loader(model, dataloader, optim, mode_name, writer, val_loader = None, start_epoch=0, training_epoch =1):
    model.train()
    val_every_epoch = 10 if torch.cuda.is_available() else 1
    max_epoch = 0
    for epoch in tqdm(range(training_epoch)):
        print("====training epoch no. %s ====="%epoch)
        for i, data in enumerate(dataloader):
            x_context, y_context, x_target, y_target, kernel_idx = data
            (mean, var), prior, poster = model(x_context.to(device), y_context.to(device), x_target.to(device),
                                            y_target.to(device), num_samples=1)
            nll_loss = compute_loss(mean, var, y_target.to(device))
            kl_loss = comput_kl_loss(prior, poster)
            loss = nll_loss + kl_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            if val_loader is None:
                if i == 0 or (i + 1) % val_every_epoch == 0:
                    writer.add_scalars("Log-likelihood", {"%s"%mode_name: -loss.item()}, int(max_epoch+i/val_every_epoch))
            else:
                model.eval()
                if i == 0 or (i + 1) % val_every_epoch == 0:
                    if isinstance(val_loader, list):
                        # validate on rbf_val and period_val
                        eval_loss_k1 = val_model_loader(model, val_loader[0])
                        eval_loss_k2 = val_model_loader(model, val_loader[1])
                        writer.add_scalars("Log-likelihood", {"mixed/val_%s"%mode_name[0]: -eval_loss_k1}, max_epoch+int(i/val_every_epoch) + start_epoch)
                        writer.add_scalars("Log-likelihood", {"mixed/val_%s"%mode_name[1]: -eval_loss_k2}, max_epoch+int(i/val_every_epoch) + start_epoch)
                    else:
                        # validate on rbf_val
                        eval_loss = val_model_loader(model, val_loader)
                        writer.add_scalars("Log-likelihood", {"%s"%mode_name: -eval_loss},  max_epoch+int(i/val_every_epoch) + start_epoch)
        max_epoch += i/val_every_epoch
    return max_epoch



def main_GP(kernel_idx1=0, kernel_idx2=1, MODELNAME = 'NP'):
    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    batch_size = 64 if torch.cuda.is_available() else 2
    # load data set
    kernel_dict = {0: 'RBF', 1: 'Periodic_Kernel',2: 'Matern'}
    kernel_1, kernel_2 = kernel_dict[kernel_idx1], kernel_dict[kernel_idx2]
    tau_dataset, dataset_tr_k1, mixed_tr, dataset_val_k1, dataset_val_k2 = gp_transport_task(kernel1=kernel_1, kernel2=kernel_2)
    writer = SummaryWriter('runs/' + MODELNAME + '_' + time_stamp + 'k1_'+kernel_1 + 'k2_'+kernel_2)
    tau_loader = torch.utils.data.DataLoader(tau_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fns(50, 50)) # tau dataset to learn a rbf prior
    train_k1_loader = torch.utils.data.DataLoader(dataset_tr_k1, batch_size=batch_size, shuffle=True, collate_fn = collate_fns(50, 50))
    train_mixed_loader = torch.utils.data.DataLoader(mixed_tr, batch_size=batch_size, shuffle=True, collate_fn = collate_fns(50, 50))
    val_k1_loader = torch.utils.data.DataLoader(dataset_val_k1, batch_size=batch_size, shuffle=True, collate_fn = collate_fns(50, 50))
    val_k2_loader = torch.utils.data.DataLoader(dataset_val_k2, batch_size=batch_size, shuffle=True, collate_fn = collate_fns(50, 50))

    # data for recording training progress
    # plot_data = load_plot_data(kernel)

    np = NP(input_dim=1, latent_dim=128, output_dim=1, use_attention=MODELNAME == 'ANP').to(device)
    optim = torch.optim.Adam(np.parameters(), lr=3e-4, weight_decay=1e-5)

    # Warm up the model with tau dataset and save the model
    training_epoch = 200 if torch.cuda.is_available() else 1
    # star_epoch = train_model_loader(np, tau_loader, optim, kernel_1, writer, training_epoch=training_epoch)
    # save model output
    # torch.save(np.state_dict(), 'saved_model/' + '%s_tau'%kernel_1 + '_' + MODELNAME + '.pt')
    print("finished training/loading on tau dataset!")
    # test the performance of the prior
    np.load_state_dict(torch.load('saved_model/' + '%s_tau'%kernel_1 + '_' + MODELNAME + '.pt', map_location=device))
    save_plot(tau_dataset,kernel_1, np, name='tau_%s'%kernel_1, MODELNAME=MODELNAME)

    # Step 1: load the model from the kernel-1 and continue training using kernel-1
    # star_epoch = 15620
    np.load_state_dict(torch.load('saved_model/' + '%s_tau'%kernel_1 + '_' + MODELNAME + '.pt', map_location=device))
    new_epoch = 50
    # train_model_loader(np, train_k1_loader, optim, kernel_1, writer, val_loader=val_k1_loader,
    #                    start_epoch=star_epoch, training_epoch=new_epoch)
    # torch.save(np.state_dict(), 'saved_model/' + '%s_only'%kernel_1 + '_' + MODELNAME + '.pt')
    np.load_state_dict(torch.load('saved_model/' + '%s_only'%kernel_1 + '_' + MODELNAME + '.pt', map_location=device))
    save_plot(dataset_val_k1, kernel_1, np, name='only_%s'%kernel_1, MODELNAME=MODELNAME)

    # step 2: load the model from the original kernel-1 again and training on a mixture of kernel 1 and 2
    np.load_state_dict(torch.load('saved_model/' + '%s_tau'%kernel_1+ '_' + MODELNAME + '.pt', map_location=device))
    # train_model_loader(np, train_mixed_loader, optim,  [kernel_1, kernel_2], writer, val_loader=[val_k1_loader, val_k2_loader],
    #                    start_epoch=star_epoch, training_epoch=new_epoch)
    # torch.save(np.state_dict(), 'saved_model/' + '%s_mix_%s' % (kernel_1, kernel_2) + '_' + MODELNAME + '.pt')
    np.load_state_dict(torch.load('saved_model/' + '%s_mix_%s' % (kernel_1, kernel_2)  + '_' + MODELNAME + '.pt', map_location=device))
    save_plot(dataset_val_k1, kernel_1, np, name='%s_mix_%s_k1' % (kernel_1, kernel_2), MODELNAME=MODELNAME)
    save_plot(dataset_val_k2, kernel_1, np, name='%s_mix_%s_k2' % (kernel_1, kernel_2), MODELNAME=MODELNAME)

    writer.close()
    print("finished training!")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='NP', help='NP or ANP')
    parser.add_argument('--kernel1', type=int, default=0, help='tau kernel')
    parser.add_argument('--kernel2', type=int, default=1, help='new kernel')
    opt = parser.parse_args()
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_GP(kernel_idx1=opt.kernel1, kernel_idx2=opt.kernel2, MODELNAME=opt.modelname)
    # main_realworld()



