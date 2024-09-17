
import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
    
def seedfill(im, seed_row, seed_col, fill_color,bckg):
    """
    im: The image on which to perform the seedfill algorithm
    seed_row and seed_col: position of the seed pixel
    fill_color: Color for the fill
    bckg: Color of the background, to be filled
    Returns: Number of pixels filled
    Behavior: Modifies image by performing seedfill
    """
    size=0  # keep track of patch size
    n_row, n_col = im.shape
    front={(seed_row,seed_col)}  # initial front
    while len(front)>0:
        r, c = front.pop()  # remove an element from front
        if im[r, c]==bckg: 
            im[r, c]=fill_color  # color the pixel
            size+=1
            # look at all neighbors
            for i in range(max(0,r-1), min(n_row,r+2)):
                for j in range(max(0,c-1),min(n_col,c+2)):
                    # if background, add to front
                    if im[i,j]==bckg and\
                       (i,j) not in front:
                        front.add((i,j))
    return size


def fill_cells(edge_image):
    """
    Args:
        edge_image: A black-and-white image, with black background and
                    white edges
    Returns: A new image where each close region is filled with a different
             grayscale value
    """
    filled_image=edge_image.copy()
    n_regions_found_so_far=0
    # start by filling the background to dark gray, from pixel (0,0)
    s=seedfill(filled_image, 0 ,0, 0.1,0)
    for i in range(filled_image.shape[0]): 
        for j in range(filled_image.shape[1]):
            # if pixel is black, seedfill from here
            if filled_image[i,j]==0:
                col = 0.5+0.001*n_regions_found_so_far
                seedfill(filled_image, i ,j, col,0)
                n_regions_found_so_far+=1
    return filled_image


def classify_cells(original_image, labeled_image, \
                   min_size=1000, max_size=5000, \
                   infected_grayscale=0.5, min_infected_percentage=0.02) -> int:
    """
    Args:
        original_image: A graytone image
        labeled_image: A graytone image, with each closed region colored
                       with a different grayscal value
        min_size, max_size: The min and max size of a region to be called a cell
        infected_grayscale: Maximum grayscale value for a pixel to be called infected
        min_infected_percentage: Smallest fraction of dark pixels needed to call a cell infected
    Returns: An integer representing the number of cells that are infected.
    """
    n_row, n_col = original_image.shape
    # Build a set of all grayscale values in the labeled image
    grayscales = {labeled_image[i,j] for i in range(n_row) for j in range(n_col) if labeled_image[i,j]>=0.5 and labeled_image[i,j]<1}
    # Counter of number of infected cells. 
    infected = 0
    
    #Create a for loop to traverse each grayscale value in the grayscales set
    for value in grayscales: 
        #Create dark and light count variables
        dark_count = 0
        light_count = 0        
        #Scan the image by going through all the pixels
        for i in range(n_row):
            for j in range(n_col):
                #Check if the pixels of grayscale value are same in labeled image
                if labeled_image[i,j] == value:
                    #If the pixels in original image is less than or equal to 0.5
                    if original_image [i,j] <= infected_grayscale:
                        #The cell is infected meaning the cell has dark patch
                        #Count the dark pixels
                        dark_count += 1                  
                    #If the pixels is greater than 0.5, than it is light
                    else:
                        #Count the light pixels
                        light_count += 1 
        
        #Add the dark and light count together to count the total infected cells
        total_pixels = (dark_count + light_count)
        #A region with grayscale value is a valid cell if it between 1000 and 5000 pixels
        #A cell is infected if at least 2% of the pixels it contains have pixel
        #grayscale value below 0.5 in the original grayscale image
        if total_pixels >= min_size and total_pixels <= max_size and (dark_count/total_pixels) >= min_infected_percentage:
            #Increment the counter of infected cells
            infected +=1
    
    return infected

def edLessons(image):
    # Converting the image to graytone
    image_gray = rgb2gray(image)
    #STEP1
    image_sobel = sobel(image_gray)
    #STEP2
    image_sobel_T005=np.where(image_sobel>=0.05,1.0, 0.0)
    #STEP3
    n_row, n_col = image_sobel_T005.shape
    sobel_clean = image_sobel_T005.copy()
    for i in range(n_row):
        for j in range(n_col):
            if np.min(image_gray[max(0,i-1):min(n_row,i+2),max(0,j-1):min(n_col,j+2)])<0.5:
                sobel_clean[i,j]=0
    #STEP4
    image_filled=fill_cells(sobel_clean)
    #STEP5
    infected = classify_cells(image_gray, image_filled)  
    return infected

if __name__ == "__main__":     
    
    # Reading the original image
    image = io.imread("malaria_3.jpeg")
    cells_infected = edLessons(image)
    print("The number of infected cells are: " + str(cells_infected)) 
