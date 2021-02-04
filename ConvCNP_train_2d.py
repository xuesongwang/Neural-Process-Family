from data.Image_data_sampler import ImageReader
from module.convCNP_2d import ConvCNP2d
from module.utils import compute_loss, to_numpy, load_plot_data, img_mask_to_np_input, generate_mask, np_input_to_img
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from torchmeta.datasets import MiniImagenet
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Compose, Resize, ToTensor

def validation(data_val, model):
    total_ll = 0
    model.eval()
    for i, (img, _) in tqdm(enumerate(data_val)):
        mean, var = model(img.to(device))
        loss = compute_loss(mean, var, img.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)

def validation_meta(data_test, model, VAL_ITER=10):
    total_ll = 0
    model.eval()
    for i, batch in tqdm(enumerate(data_test)):
        if i == VAL_ITER :
            break
        img, _ = batch["test"]
        bs, n_shot, c, w, h = img.shape
        img = torch.reshape(img, (bs*n_shot, c, w, h))
        mean, var = model(img.to(device))
        loss = compute_loss(mean, var, img.to(device))
        total_ll += -loss.item()
    return total_ll / (i + 1)


def save_plot(epoch, data, model, imgsize):
    imgsize = list(imgsize)
    imgsize[0] = 1 # change batch_size = 1
    ax, fig = plt.subplots()
    (x_context, y_context) , x_target = data.query
    mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
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
    # save plot
    model_path = "saved_fig/CNP"
    kernel_path = os.path.join(model_path, kernel)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(kernel_path):
        os.mkdir(kernel_path)
    plt.savefig("saved_fig/CNP/"+kernel+"/"+"%03d"%(epoch)+".png")
    plt.close()
    return fig

def main_nonmeta():
    TRAINING_ITERATIONS = int(200)
    BEST_LOSS = -np.inf
    kernel = 'celebA'  # use kernel to be consistent with 1d dataset, kernel can be "MNIST", "SVHN", "celebA"
    # set up tensorboard
    time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
    # writer = SummaryWriter('runs/' + kernel + '_CNP_' + time_stamp)

    # load data set
    dataset = ImageReader(dataset=kernel, batch_size=64, datapath='/share/scratch/xuesongwang/metadata/')
    # load plot dataset for recording training progress
    # plot_data = save_plot_data(dataset.valloader, kernel) # generate and save data for the first run
    # plot_data = load_plot_data(kernel)

    convcnp = ConvCNP2d(channel=3 if kernel != 'MNIST' else 1).to(device)
    optim = torch.optim.Adam(convcnp.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in tqdm(range(TRAINING_ITERATIONS)):
        for i, (img, _) in tqdm(enumerate(dataset.trainloader)):
            mean, var = convcnp(img.to(device))
            loss = compute_loss(mean, var, img.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            # writer.add_scalars("Image Log-likelihood", {"train": -loss.item()}, epoch)
        val_loss = validation(dataset.valloader, convcnp)
        # save_plot(epoch, plot_data, cnp, img.shape)  # save training process, optional
        # writer.add_scalars("Image Log-likelihood", {"val": val_loss}, epoch)
        if val_loss > BEST_LOSS:
            BEST_LOSS = val_loss
            print("save module at epoch: %d, val log-likelihood: %.4f" % (epoch, val_loss))
            torch.save(convcnp.state_dict(), 'saved_model/' + kernel + '_ConvCNP.pt')
        # writer.close()
    print("finished training ConvCNP!" + kernel)

def main_meta():
    data_name = 'miniImagenet'
    dataset = MiniImagenet("/share/scratch/xuesongwang/metadata/",
                           num_classes_per_task=2,
                           transform=Compose([Resize(32), ToTensor()]),
                           meta_train =True,
                           download=False)
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=15, num_test_per_class=15)
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=8)
    convcnp = ConvCNP2d(channel=3).to(device)
    optim = torch.optim.Adam(convcnp.parameters(), lr=3e-4, weight_decay=1e-5)

    TRAINING_ITERATIONS = int(500)
    BEST_LOSS = -np.inf
    for epoch in range(TRAINING_ITERATIONS):
        for i, batch in tqdm(enumerate(dataloader)):
            img, _ = batch["train"]
            bs, n_shot, c, w, h = img.shape
            img = torch.reshape(img, (bs * n_shot, c, w, h))
            mean, var = convcnp(img.to(device))
            loss = compute_loss(mean, var, img.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
        val_loss = validation_meta(dataloader, convcnp)
        if val_loss > BEST_LOSS:
            BEST_LOSS = val_loss
            print("save module at epoch: %d, val log-likelihood: %.4f" % (epoch, val_loss))
            torch.save(convcnp.state_dict(), 'saved_model/' + data_name + '_ConvCNP.pt')
    print("finished training ConvCNP!" + data_name)





if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # main_nonmeta()
    main_meta()
    # main_test()
