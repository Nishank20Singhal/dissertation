import os
import sys
import pandas as pd
import scipy.io as sio
import numpy as np
import cv2
from math import cos, sin, acos, atan2
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
def getDataFrame(dataset):
    """This function creates a pandas dataframe with containing the metadata of the dataset.

    Parameters:
        dataset -> (str) The name of the dataset, example "wallstreet5k"

    Returns: 
        frame -> A pandas dataframe with dataset metadata
    """

    csvFile = os.path.join(os.path.dirname(os.path.realpath(__file__)),'metadata', dataset + '.csv')

    frame = pd.read_csv(csvFile)
    print(frame.head(5))
    return frame

def getFeatures(dataset, indices, features_type='ES', model='v1', zoom=18, domain='X'):
    """This function return a numpy array with the features predicted from the given model.

        Parameters:
            dataset ->  (str) The name of the dataset (Example wallstreet5k)
            indices ->  (array) Indices of the features to return.
            features_type -> (str) 'ES' for embedding space or 'BSD' for binary semantic descriptors
            model -> (str) The name of the model that generates the features (must match the name of the folder)
            zoom -> (str) Map zoom level (z18 or z19)
            domain -> Domain of features to return 'X' for maps and 'Y' for images
        
        Returns:
            features -> A numpy array with the feature vector descriptors.
    """
    
    if features_type == 'ES':
        zoom = 'z'+str(zoom)
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'features', features_type, model, zoom, dataset + '.mat')
        features = sio.loadmat(path)[domain][indices]
    
    return features

def rotate_panorama(image, roll=0.0, pitch=0.0, yaw=0.0):
    """ Rotates a equirectangular image given angles

        Parameters:
            image (Numpy array) -> equirectangular image
            roll (float) -> Roll angle in degrees
            pitch (float) -> Pitch angle in degrees 
            yaw (float) -> Yaw angle in degrees 
    """
    
    h,w = image.shape[0:2]
    euler_angles = np.radians(np.array([roll,pitch,yaw]))
    [R, _] = cv2.Rodrigues(euler_angles) 
    
    # Project equirectangular points to original sphere
    lat = np.pi / h * np.arange(0,h)
    lat = np.expand_dims(lat,1)
    lon = (2*np.pi / w * np.arange(0,w)).T
    lon = np.expand_dims(lon,0)

    # Convert to cartesian coordinates
    x = np.sin(lat)*np.cos(lon)
    y = np.sin(lat)*np.sin(lon)
    z = np.tile(np.cos(lat), [1,w])

    # Rotate points
    xyz = np.stack([x,y,z],axis=2).reshape(h*w,3).T
    rotated_points = np.dot(R,xyz).T    
    
    # Go back to spherical coordinates
    new_lat = np.arccos(rotated_points[:,2]).reshape(h,w)
    new_lon = np.arctan2(rotated_points[:,1],rotated_points[:,0]).reshape(h,w)
    neg = np.where(new_lon<0)
    new_lon[neg] += 2*np.pi
    
    # Remap image
    y_map = new_lat * h / np.pi
    x_map = new_lon * w / (2 * np.pi)
    new_image = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_CUBIC)

    return new_image

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result(input, target, gen_image, epoch, training=True, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    imgs = [input, gen_image, target]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + '.png'
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_pix2pix_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)
