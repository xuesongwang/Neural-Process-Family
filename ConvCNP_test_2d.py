from data.Image_data_sampler import ImageReader
from module.convCNP_2d import ConvCNP2d
# from module.convCNP_2d_old import ConvCNP2d
from module.utils import compute_loss, to_numpy, img_mask_to_np_input, generate_mask, np_input_to_img, compute_MSE
import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmeta.datasets import MiniImagenet
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Compose, Resize, ToTensor

def testing(data_test, model):
    total_ll = 0
    total_mse = 0
    model.eval()
    for i, (img, _) in tqdm(enumerate(data_test)):
        mean, var = model(img.to(device))
        loss = compute_loss(mean, var, img.to(device))
        mse_loss = compute_MSE(mean, img.to(device))
        total_ll += -loss.item()
        total_mse += mse_loss.item()
    return total_ll / (i+1), total_mse/ (i+1)

def testing_meta(data_test, model):
    total_ll = 0
    total_mse = 0
    model.eval()
    for i, batch in tqdm(enumerate(data_test)):
        img, _ = batch["test"]
        bs, n_shot, c, w, h = img.shape
        img = torch.reshape(img, (bs*n_shot, c, w, h))
        mean, var = model(img.to(device))
        loss = compute_loss(mean, var, img.to(device))
        mse_loss = compute_MSE(mean, img.to(device))
        total_ll += -loss.item()
        total_mse += mse_loss.item()
    return total_ll / (i + 1), total_mse / (i + 1)

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

def main_nonmeta():
    kernel = 'celebA'  # use kernel to be consistent with 1d dataset, kernel can be "MNIST", "SVHN", "celebA"

    # load data set
    dataset = ImageReader(dataset=kernel, batch_size=32, datapath='/share/scratch/xuesongwang/metadata/')
    convcnp = ConvCNP2d(channel=3 if kernel != 'MNIST' else 1).to(device)
    # convcnp.load_state_dict(torch.load('saved_model/' + kernel + '_ConvCNP_olderversion.pt'))
    convcnp.load_state_dict(torch.load('saved_model/' + kernel + '_ConvCNP.pt'))
    print("successfully load ConvCNP module!")
    total_loss = []
    total_mse = []
    for _ in range(6):
        test_ll, test_mse = testing(dataset.testloader, convcnp)
        total_loss.append(test_ll)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))

    # test_ll, test_mse = testing(dataset.testloader, cnp)
    # print ("CNP loglikelihood on %d samples: %.4f, mse: %.4f"%(len(dataset.testloader), test_ll, test_mse))

    # fig = plot_sample(dataset.testloader, cnp)
    # print("save plots!")

def main_meta():
    data_name = 'miniImagenet'
    dataset = MiniImagenet("/share/scratch/xuesongwang/metadata/",
                           num_classes_per_task=2,
                           transform=Compose([Resize(32), ToTensor()]),
                           meta_test =True,
                           download=False)
    dataset = ClassSplitter(dataset, shuffle=False, num_train_per_class=15, num_test_per_class=15)
    testloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=8)
    convcnp = ConvCNP2d(channel=3).to(device)
    convcnp.load_state_dict(torch.load('saved_model/' + data_name + '_ConvCNP.pt'))
    print("successfully load ConvCNP module!")
    total_loss = []
    total_mse = []
    for _ in range(6):
        test_ll, test_mse = testing_meta(testloader, convcnp)
        total_loss.append(test_ll)
        total_mse.append(test_mse)
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_loss), np.std(total_loss)))
    print("for 6 runs, mean: %.4f, std:%.4f" % (np.mean(total_mse), np.std(total_mse)))
    # test_ll, test_mse = testing_meta(testloader, cnp)
    # print ("CNP loglikelihood on miniImageNet: %.4f, mse: %.4f"%(test_ll, test_mse))

if __name__ == '__main__':
    # define hyper parameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # main_nonmeta()
    main_meta()



