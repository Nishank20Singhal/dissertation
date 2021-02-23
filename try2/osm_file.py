import os, sys
import subprocess
import cv2
import numpy as np
import pandas as pd
from math import sin, cos
from utils import getDataFrame
from Location import Location # Location class from SLutils 

def look_at_generator(loc, elevation):
    """ 
    This function generates a string that contains the 'look_at' argument for perspective generation in OSM2World.

    Arguments:
        loc (Location object) -> A Location object with the information where render has to be created (latitude, longitude and yaw)
        elevation (float) -> Elevation for the perspective view 

    Returns:
        look_at -> A string with 'look_at' parameters to generate the perspective view

    """
    lat_shift = 0.001*cos(loc.yaw*np.pi/180) # Look at this latitude
    lon_shift = 0.001*sin(loc.yaw*np.pi/180) # Look at this longitude
    print(loc.yaw, lat_shift, lon_shift)
    look_at = "+{},{},+{}".format(loc.lat+lat_shift, loc.lon+lon_shift, elevation) # Look at this direction
    return look_at

def generate_and_save(loc, config):
    """ 
    This functions generate a perspective view using OSM2World and then save it to the disk'

    Arguments:
        loc -> A location object
        config -> A dictionary with configuration information, should contain the fallowing keys
            'name' -> Name of the dataset, options {'wallstreet5k', 'hudsonriver5k', 'wallstreet5k'}
            'jarPath' -> Path to OSM2World.jar
            'elevation' -> Elevation
            'index' -> location index. It will be used as the name for generated renders.

    """

    # Create directories to save results, you can adapt this to the place where you want to save results
    # in my case I have an environmental variable called 'dataset' pointing to a directory where I store all datasets

    savedirA = os.path.join( 'streetlearn_osm2world', config['name'] + 'A') 
    savedirB = os.path.join('streetlearn_osm2world', config['name'] + 'B')
    savedirAB = os.path.join('streetlearn_osm2world', config['name'])

    if not os.path.isdir(savedirAB):
        os.makedirs(savedirAB)
    
    if not os.path.isdir(savedirA):
        os.makedirs(savedirA)
    
    if not os.path.isdir(savedirB):
        os.makedirs(savedirB)


    # Perspective parameters
    pview_pos = "+{},{},+{}".format(loc.lat, loc.lon, config['elevation']) # Define camera position in the coordinates given by location
    look_at = look_at_generator(loc, config['elevation'])                  # Define view direction "look_at"

    mapFilePath = os.path.join( 'map_data', config['name'] + '.osm')       # Path to the osm file
    
    # Call OSM2World to do the job. It will render the 3D given parameters and save result to a temporal file called osm2world.png
    # This is indeed the peace of code that does the interesting part.

    subprocess.call( ['java', '-jar', config['jarPath'],    # Call osm2world creating a subprocess
                      '-i', mapFilePath,                    # Input mapfile
                      '-o', 'osm2world.png',                # name of the output temporal filename 
                      '--resolution','256,256',             # size of output image
                      '--pview.fovy','90',                  # field of view (vertical)
                      '--pview.pos', pview_pos,             # camera position 
                      '--pview.lookAt', look_at])           # look_at direction
    
    #read rendered image 
    img_rendered = cv2.imread('osm2world.png') # Read the image rendered in the previous step
    #cv2.imshow('vjgu', img_rendered)
    #cv2.waitKey(0)
    snap = loc.cropPano(90,0,0,256,256)        # Use Location class to crop views from the corresponding panorama 

    #img_combined = cv2.addWeighted(snap,0.5,img_rendered,0.5,0)
    #img = np.concatenate([snap, img_rendered, img_combined], axis=1)
    image = np.concatenate([snap, img_rendered], axis=1)    # Concatenate the snap from the panorama and the rendered image generated by osm2world
    
    # save AB image (two domains side by side)
    image_path = os.path.join(savedirAB, str(config['index'])+'.jpg')
    cv2.imwrite(image_path, image)

    # save A image (Real image cropped from panorama)
    image_path = os.path.join(savedirA, str(config['index'])+'.jpg')
    cv2.imwrite(image_path, snap)
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(config['index'])
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    
    # save B image (3D render image from osm2world)
    image_path = os.path.join(savedirB, str(config['index'])+'.jpg')
    cv2.imwrite(image_path, img_rendered)


# Define configuration
config = {
    'name' : 'hudsonriver5k',                                                            # Dataset area name
    'jarPath' : os.path.join('OSM2world','OSM2World.jar'), # path to OSM2World
    'elevation' : 2.5,                                                                  # Elevation
    'samples': 5000                                                                    # Number of samples to generate
}

#np.random.seed(442)                                                                     # Set the seed for reproducibility
#indices = np.random.choice(np.arange(0,5000), config['samples'], replace=False)         # Randomly select 1000 points

# Create a pandas dataframe for the dataset area and take a sample

frame = getDataFrame(config['name'])
print(config['name'])

subframe = frame.sample(n=config['samples'], random_state=442).reset_index(drop=True)
print(subframe.head())

print(config)

for i in range(config['samples']):
    config['index'] = subframe.loc[i,'local_index']
    loc =  Location(config['name'], int(config['index']), 'manhattan', base_index='local')
    loc.lat = subframe.loc[i,'lat']
    loc.lon = subframe.loc[i,'lon']
    loc.yaw = subframe.loc[i,'yaw']
    
    generate_and_save(loc, config)
    print(i, loc)
