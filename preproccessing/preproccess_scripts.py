import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes






def get_cancer_mole(img, k_size=5):

    """
    runs gray and blur for np images
    in a list and returns new list

    Inputs
    ----------------
    img: list of numpy RGB img files

    Outputs
    -----------------------
    proccessed: list of gray and blurred 3-D np files
    """
    
    #grey image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # blur image
    blur = cv2.GaussianBlur(img,(k_size,k_size),0)
    
    # apply Otsu's threshold
    ret3, mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Run blur to close contours and remove remaining noise
    se = np.ones((11,11), dtype='uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
        
    return mask



def get_cancer_mask(img, k_size=5):

    """
    runs gray and blur for np images
    in a list and returns new list

    Inputs
    ----------------
    img: list of numpy RGB img files

    Outputs
    -----------------------
    proccessed: list of gray and blurred 3-D np files
    """
    
    
    mask = get_cancer_mole(img, k_size)
    

    # invert 
    mask = (255 -mask)
    
    #reshape to fit 3-D RGB
    mask = mask.reshape([mask.shape[0], mask.shape[1], 1])
    
    # Convert the mask to bools then back to int to get to 0 and 1
    mask = mask.astype(bool).astype(np.int8)
    
        
    return mask


def juxtapose_mole_and_background(cancer_img, background_img):
    
    """
    Given an unprocessed cancer image and background image,
    gets the cancer mole and superimposeses it on the background
    image, and returns the juxtaposed image

    Inputs
    ----------------------------------
    cancer_image : RGB 3-D NP array
    background_image : RGB 3-D NP array

    Output
    -------------------
    Juxtaposed_image: Background with Mole superimpsed; RGB 3-D NP Array
     """


    mask = get_cancer_mask(cancer_img)

    mask=mask[:,:,0:1].astype(np.int8)
    
    
    mole = cv2.bitwise_and(cancer_img, cancer_img, mask=mask)
    
    
    cleared_background = cv2.bitwise_and(background_img, background_img, mask=(1-mask))
    
    
    mixed_img = cv2.bitwise_or(mole, cleared_background)
    
    return mixed_img


def load_data(img_path, descr_path, start=0, end=100, base_width=244, base_height=244):
    
    filenames = os.listdir(descr_path)
    
    df = pd.DataFrame(columns= ['diagnosis', 'benign_malignant', 'melanocytic', 'images'])
    
    images = []
    diagnosis = []
    benign_malignant = []
    melanocytic = []
    
    i = start
    
    for file in filenames:
        
        if i > end:
            break

        data = json.load(open(descr_path+file))

        clinical = data["meta"]["clinical"]
        
        if "diagnosis" and "benign_malignant" and "melanocytic" in clinical.keys():
        
            diagnosis.append(clinical["diagnosis"])
            benign_malignant.append(clinical["benign_malignant"])
            melanocytic.append(clinical["melanocytic"])  
            
            img = cv2.imread(str(img_path)+str(file)+".jpg")
            img = cv2.resize(img, (base_width, base_height))
            
            images.append(img)
            
            i += 1
        
    df['diagnosis'] = np.array(diagnosis)
    df['benign_malignant'] = np.array(benign_malignant)
    df['melanocytic'] = np.array(melanocytic)
    
    return df, np.array(images)
