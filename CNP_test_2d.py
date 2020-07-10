from data.Image_data_sampler import ImageReader
from module.CNP import ConditionalNeuralProcess as CNP
from module.utils import compute_loss, to_numpy, img_mask_to_np_input, generate_mask, np_input_to_img
import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def testing(data_test, model):
    total_ll = 0
    model.eval()
    for i, (img, _) in tqdm(enumerate(data_test)):
        context_mask, target_mask = generate_mask(img)
        x_context, y_context, x_target, y_target = img_mask_to_np_input(img, context_mask, target_mask, \
                                                                        include_context=False)
        mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
        loss = compute_loss(mean, var, y_target.to(device))
        total_ll += -loss.item()
    return total_ll / (i+1)

def plot_sample(data, model):
    img, _ = next(iter(data))
    context_mask, target_mask = generate_mask(img)
    x_context, y_context, x_target, y_target = img_mask_to_np_input(img, context_mask, target_mask, \
                                                                    include_context=False)
    imgsize = list(img.shape)
    ax, fig = plt.subplots()
    mean, var = model(x_context.to(device), y_context.to(device), x_target.to(device))
    # recover prediction
    mean_recover = np_input_to_img(x_target, mean.cpu(), imgsize)
    var_recover = np_input_to_img(x_target, var.cpu(), imgsize)
    # recover x_target
    target_recover = np_input_to_img(x_target, y_target, imgsize)
    # recover x_context
    context_recover = np_input_to_img(x_context, y_context, imgsize)

    mean_recover += context_recover  # fill in context data with prediction
    raw_image = target_recover + context_recover # recover raw image
    img_recover = torch.cat([raw_image, mean_recover, var_recover], dim=3)

    # last squeeze for removing color channel for gray scale image
    img_recover = make_grid(img_recover, nrow=4, pad_value=1.).permute(1, 2, 0)
    plt.imshow(to_numpy(img_recover))
    plt.savefig("saved_fig/CNP_"+kernel+".png")
    plt.close()
    return fig

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    kernel = 'celebA' # use kernel to be consistent with 1d dataset, kernel can be "MNIST", "SVHN", "celebA"

    # load data set
    dataset = ImageReader(dataset = kernel, batch_size=64, datapath='/share/scratch/xuesongwang/metadata/')
    cnp = CNP(input_dim=2, latent_dim = 128, output_dim=3 if kernel != 'MNIST' else 1).to(device)
    cnp.load_state_dict(torch.load('saved_model/' + kernel + '_CNP.pt'))
    print("successfully load CNP module!")
    total_loss = []
    # for _ in range(10):
    #     test_loss = testing(dataset.testloader, cnp)
    #     total_loss.append(test_loss)
    # print("for 10 runs, mean: %.4f, std:%.4f" %(np.mean(total_loss), np.std(total_loss)))
    test_loss = testing(dataset.testloader, cnp)
    print ("CNP loglikelihood on %d samples: %.4f"%(len(dataset.testloader), test_loss))

    # fig = plot_sample(dataset.testloader, cnp)
    # print("save plots!")



