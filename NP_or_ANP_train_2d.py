from data.Image_data_sampler import ImageReader
from module.NP import NeuralProcess as NP
from module.utils import compute_loss, comput_kl_loss, to_numpy, load_plot_data, img_mask_to_np_input, generate_mask, np_input_to_img
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

def validation(data_test, model):
    total_ll = 0
    model.eval()
    for i, (img, _) in tqdm(enumerate(data_test)):
        context_mask, target_mask = generate_mask(img)
        x_context, y_context, x_target, y_target = img_mask_to_np_input(img, context_mask, target_mask, \
                                                                        include_context=False)
        (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, y_target.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)

def save_plot(epoch, data, model, imgsize):
    imgsize = list(imgsize)
    imgsize[0] = 1 # change batch_size = 1
    ax, fig = plt.subplots()
    (x_context, y_context) , x_target = data.query
    (mean, var), _, _ = model(x_context.to(device), y_context.to(device), x_target.to(device))
    # recover prediction
    mean_recover = np_input_to_img(x_target, mean.cpu(), imgsize)
    var_recover = np_input_to_img(x_target, var.cpu(), imgsize)
    # recover x_target
    target_recover = np_input_to_img(x_target, data.y_target, imgsize)
    # recover x_context
    context_recover = np_input_to_img(x_context, y_context, imgsize)

    mean_recover += context_recover  # fill in context data with prediction
    raw_image = target_recover + context_recover # recover raw image
    img_recover = torch.cat([raw_image, mean_recover, var_recover], dim=3)
    plt.title("epoch:%d" % epoch)
    # last squeeze for removing color channel for gray scale image
    plt.imshow(to_numpy(img_recover.squeeze(0).permute(1,2,0).squeeze()))
    model_path = "saved_fig/" + MODELNAME
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/"+MODELNAME+"/"+kernel+"/"+"%03d"%(epoch)+".png")
    plt.close()
    return fig

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAINING_ITERATIONS = int(200)
    BEST_LOSS = -np.inf
    MODELNAME = 'NP' # 'NP' or 'ANP'
    kernel = 'MNIST'  # EQ/ period / MNIST/ SVHN / celebA
    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    writer = SummaryWriter('runs/'+kernel+'_' + MODELNAME +'_'+ time_stamp)
    # load data set
    dataset = ImageReader(dataset=kernel, batch_size=64, datapath='/share/scratch/xuesongwang/metadata/')
    # load plot dataset for recording training progress
    # plot_data = save_plot_data(dataset.valloader, kernel)  # generate and save data for the first run
    plot_data = load_plot_data(kernel)

    np = NP(input_dim=2, latent_dim = 128, output_dim=3 if kernel != 'MNIST' else 1, use_attention=MODELNAME=='ANP').to(device)
    optim = torch.optim.Adam(np.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        for i, (img, _) in tqdm(enumerate(dataset.trainloader)):
            context_mask, target_mask = generate_mask(img)
            x_context, y_context, x_target, y_target = img_mask_to_np_input(img, context_mask, target_mask, \
                                                                            include_context = True)
            (mean, var), prior, poster = np(x_context.to(device), y_context.to(device), x_target.to(device), y_target.to(device))
            nll_loss = compute_loss(mean, var, y_target.to(device))
            kl_loss = comput_kl_loss(prior, poster)
            loss = nll_loss + kl_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        writer.add_scalars("Image Log-likelihood", {"train/overall": -loss.item(),
                                              "train/ll": -nll_loss.item(),
                                              "train/kl": kl_loss.item()}, epoch)

        val_loss = validation(dataset.valloader, np)
        save_plot(epoch, plot_data, np, img.shape)
        writer.add_scalars("Image Log-likelihood", {"val": val_loss}, epoch)
        if val_loss > BEST_LOSS:
            BEST_LOSS = val_loss
            print("save module at epoch: %d, val log-likelihood: %.4f" %(epoch, val_loss))
            torch.save(np.state_dict(), 'saved_model/'+kernel+'_' + MODELNAME+'.pt')
        writer.close()
    print("finished training: " + MODELNAME)



