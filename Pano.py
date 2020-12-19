import os, sys
import numpy as np
import cv2
import pandas as pd
import E2P
import glob
class Pano():
    """ Create object panorama"""
    
    def __init__(self, panoDir, ID):
        self.panoDir = panoDir
        if type(ID) is str:
            self.panoName = ID
        
        elif type(ID) is int:
            filename = os.path.join( self.panoDir, 'nodes.txt' )
            names = ["pano_id", "yaw", "lat", "lon"]
            frame = pd.read_csv(filename, names=names)
            pano_id = frame.loc[ID, 'pano_id']
            self.panoName = pano_id + '.jpg'

        else:
            sys.exit("Pano ID not found")

        self.path = self.getPath()
        self.pano = self.getPano()
        self.shape = self.pano.shape
    def getPath(self):
        path = os.path.join( self.panoDir, self.panoName)
        return path

    def getPano(self, size=(1024,512), flip=False):
        pano = cv2.imread(self.path)
        
        pano = cv2.resize(pano, size)
        #cv2.imshow(self.panoName, pano)
        #cv2.imwrite("Hundsonriver5K_input/" + self.panoName, pano)
        if flip:
            pano = cv2.flip(pano, 1)
        return pano
    
    def showPano(self):
        cv2.imshow(self.panoName, self.pano)
        cv2.waitKey(0)

    def getZoom(self):
        """Returns pano's zoom level"""
        return int(np.ceil(self.pano.shape[0] / 512))

    def getSnaps(self, size=224, mode='list', rotation=0.0, flip=False, noise=False):
        
        snaps = []
        equ = E2P.Equirectangular(self.path)
        
        views = [0,-90,90,180] 
        
        if noise:
            fov_shift = np.random.normal(loc=0, scale=10)
            pitch_shift = np.random.normal(loc=0,scale=10)
            tetha_shift = np.random.normal(loc=0,scale=10)

        else:
            fov_shift = 0
            pitch_shift = 0
            tetha_shift = 0        

        tetha_shift = tetha_shift + rotation    
        snaps = [equ.GetPerspective(100+fov_shift, t+tetha_shift, pitch_shift, size, size) for t in views]
        
        if mode == 'list' and not flip:
            return snaps
        
        elif mode == 'list' and flip:
            new_list = [cv2.flip(snaps[i], 1) for i in [0,2,1,3]] # flip snaps 
            return new_list 

        elif mode == 'grid' and not flip:    
            row1 = np.concatenate([snaps[0], snaps[3]], axis=1) # Concatenate F and B
            row2 = np.concatenate([snaps[1], snaps[2]], axis=1) # Concatenate L and R
            img = np.concatenate([row1, row2], axis=0) # [F,R;L,R]
            return img 
        
        elif mode == 'grid' and flip:
            snaps = [cv2.flip(snap, 1) for snap in snaps]
            row1 = np.concatenate([snaps[0], snaps[3]], axis=1) # Concatenate F and B
            row2 = np.concatenate([snaps[2], snaps[1]], axis=1) # Concatenate L and R
            img = np.concatenate([row1, row2], axis=0) # [F,R;L,R]
            return img
            

    def getSnapswithInfo(self, size=224, colour = (255,255,255), text=None):
        """ Returns a list with snaps in directions 0, 90, -90, 180"""
        thick = int(0.05 * size) # Thickness is 5 % 
        snaps = self.getSnaps(size)
        snaps = [cv2.copyMakeBorder(snap, thick,thick,thick,thick, cv2.BORDER_CONSTANT, None, colour) for snap in snaps] 
        directions = ['F', 'L', 'R', 'B']
        if text is not None:
            for i, direction in enumerate(directions):
                txt = 'ID: ' + text + ' ' + direction
                cv2.putText(snaps[i], txt, (10,size), cv2. FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
        return snaps 

    def cropPano(self, fov, theta, pitch, h, w):
        """ Returns a pano the the specified parameters"""
        equ = E2P.Equirectangular(self.path)
        snap = equ.GetPerspective(fov, theta, pitch, h, w)
        return snap

    def saveSnaps(self, size=224, directory=None, option='group'):
        savedir = os.getcwd() if directory == None else directory
        basename = os.path.join(savedir, self.panoName.split('.')[0]) 

        if option == 'group':
            snaps = self.getSnapswithInfo(size=size, text=None)
            row1 = np.concatenate([snaps[0], snaps[3]], axis=1) # FB
            row2 = np.concatenate([snaps[1], snaps[2]], axis=1) # RL
            image = np.concatenate([row1, row2], axis=0)
            cv2.imwrite("Hundsonriver5K_pano_output_group/" + self.panoName, image)
            cv2.waitKey(0)    

        elif option == 'individual':
            snaps = self.getSnapswithInfo(size=size, text=None)
            directions = ['F', 'L', 'R', 'B']
            for i, snap in enumerate(snaps):
                direction = directions[i]
                cv2.imwrite("Hundsonriver5K_pano_output_ind/"+ direction  + '_' + self.panoName , snap)
                cv2.waitKey(0)
        else:
            print("Option not found, image not saved")

    def getCoordinates(self):
        filename = os.path.join( self.panoDir, 'nodes.txt' )
        names = ["pano_id", "yaw", "lat", "lon"]
        frame = pd.read_csv(filename, names=names)
        row = frame.loc[frame['pano_id'] == self.panoName.split('.')[0]]
        index = row.index[0]
        yaw, lat, lon = row['yaw'].values[0], row['lat'].values[0], row['lon'].values[0]
        return (index, lat, lon, yaw)

    def __str__(self):
        index, lat, lon, yaw = self.getCoordinates()
        return "Pano name: {}, index: {}, shape: {}, coordinates: ({},{},{})".format(self.panoName, index, self.pano.shape, lat, lon, yaw)

if __name__ == "__main__":
    list_of_images = glob.glob('hudsonriver5k/*.jpg')           # create the list of file
    for image_name in list_of_images:
        #print(os.path.basename(image_name))
        
        pano = Pano('hudsonriver5k', os.path.basename(image_name))
        snaps = pano.getSnapswithInfo(size=256)
        pano.saveSnaps(size=256, directory=None, option='individual')
        img = np.concatenate(snaps, axis=1)
        #img = pano.cropPano(0, 0, 0, 512, 1024)
        #cv2.imshow("pano", img)
        #cv2.waitKey(0)
        