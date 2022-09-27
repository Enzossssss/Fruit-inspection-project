#Imported Libraries
import cv2
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def visualize_imgs_array(images):
    """
    To visualize a list of image.

    Parameters
    ----------
    analysis : list of the labels of the masks of each images
    images : list of the images
    """
    plt.figure()
    if len(images) > 1:
        f, a = plt.subplots(1,len(images), figsize=(15, 15))
        i = 0
        for img in images:
            a[i].imshow(img, cmap = 'gray')
            i += 1
    elif len(images) == 1:
        plt.imshow(images[0], cmap = 'gray')
    plt.show()

def create_masks(analysis, images):
    """
    Given the analysis of the connected components, it returns the mask of the connected component with a higher area.

    Parameters
    ----------
    analysis : list of the labels of the masks of each images
    images : list of the images
    """
    masks = [np.zeros((img.shape[0], img.shape[1]), dtype="uint8") for img in images]
    mask_index = 0

    for a in analysis:
        max_area = 0
        for i in range(1, a[0]):
            component = np.zeros_like(a[1], dtype=np.uint8)
            component[a[1] == i] = 255
            area = cv2.countNonZero(component)
            
            if area > max_area:
                mask = component
                max_area = area

        masks[mask_index] = mask
        mask_index += 1

    return masks

def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def create_defects_masks(analysis, images, lower, upper):
    """
    Given the analysis of the connected components, it returns the mask of the connected components that have an area bigger than the lower value and smaller than the upper value both in input.

    Parameters
    ----------
    analysis : list of the labels of the masks of each images
    images : list of the images
    lower : float value, lower threshold of the area
    upper : float value, upper threshold of the area
    """
    
    masks = [np.zeros((img.shape[0], img.shape[1]), dtype="uint8") for img in images]
    mask_index = 0
    for a in analysis:
        intramasks = [np.zeros((images[0].shape[0], images[0].shape[1]), dtype="uint8") for i in range(a[0]-2)]
        intramask_index = 0
        for i in range(0, a[0]+1):
            component = np.zeros_like(a[1], dtype=np.uint8)
            component[a[1] == i] = 255
            area = cv2.countNonZero(component)

            if area < upper and area > lower:
                intramasks[intramask_index] = component
                intramask_index += 1
        masks[mask_index] = intramasks
        mask_index += 1
    
    return masks

def get_bbox(masks, images):
    """
    Given the masks compute the bounding box verifying that there isn't overlapping.

    Parameters
    ----------
    masks : list of the masks of the defects
    images : list of the images
    """
    rect = [[] for i in range(len(masks))]
    temp = [()]
    for i in range(len(masks)):
        for j in range(len(masks[i])):
            if np.sum(masks[i][j]) != 0:
                contours,_ = cv2.findContours(masks[i][j].copy(), 1, 1)
                temp[0] = cv2.minAreaRect(contours[0])
                rect[i] = rect[i] + temp

    box = [[] for i in range(len(rect))]
    rect2 = [np.zeros((img.shape[0], img.shape[1]), dtype="uint8") for img in images]

    for i in range(len(rect)):
        for j in range(len(rect[i])):
            tmp = cv2.boxPoints(rect[i][j])
            tmp = np.int0(tmp)
            box[i] = box[i] + [tmp]
        for j in range(len(box[i])):
            for k in range(len(box[i])):
                if j != k and  box[i][j] != [] and box[i][k] != []:
                    not_valid_box = valid_bbox(box[i][j], box[i][k])
                    if not_valid_box != []:
                        box[i][k] = []
        for j in range(len(box[i])-1, -1, -1):
            if box[i][j] == []:
                del box[i][j]
        rect2[i] = cv2.drawContours(images[i],box[i],-1,(0,255,0),2)
    return rect2

def valid_bbox(boxA, boxB):
    """
    verify if the two bounding boxes are overlapped, and it returns the smaller one.

    Parameters
    ----------
    boxA : bounding box A
    boxB : bounding box B
    """
    polygon_1 = Polygon(boxA)
    polygon_2 = Polygon(boxB)

    intersect = polygon_1.intersection(polygon_2).area / polygon_1.union(polygon_2).area

    if round(intersect*100, 2) > 0:
        if polygon_1.area > polygon_2.area:
            return boxB
    
    return []


def mahalanobis_distance_with_samples(russets, isolate_fruits_gray, isolate_fruits):
    """
    Compute the Mahalanobis distance between each pixel of the images and the samples.

    Parameters
    ----------
    russets : list of samples of russet
    isolate_fruits_gray : list of the fruits with the mask (gray version)
    isolate_fruits : list of the fruits with the mask
    """
    cov = [np.zeros((2, 2)) for r in russets]
    mean = [[] for r in russets]
    inv_covariance = [np.zeros((2, 2)) for r in russets]
    copy = [[isolate_fruits_gray[total_image].copy() for r in russets] for total_image in range(len(isolate_fruits))]

    for total_image in range(len(isolate_fruits)):
        for i in range(len(russets)):
            cov[i], mean[i] = cv2.calcCovarMatrix(russets[i][:, :, 1:3].reshape(russets[i].shape[0] * russets[i].shape[1], 2), None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
            inv_covariance[i] = cv2.invert(cov[i], cv2.DECOMP_SVD)[1]
            for r in range(isolate_fruits[total_image].shape[0]):
                for c in range(isolate_fruits[total_image].shape[1]):

                    p = np.array(isolate_fruits[total_image][r][c])[1:3].reshape(1, 2)

                    dist = cdist(p, mean[i], 'mahalanobis', VI=inv_covariance[i])

                    if dist < 1.5:
                        copy[total_image][i][r][c] = 255
    return copy

def voting(isolate_fruits, smooth_edges):
    """
    vote for the pixels that are part of a russet.

    Parameters
    ----------
    isolate_fruits : list of the fruits with the mask
    smooth_edges : smoothing edges 
    """
    bins = [np.zeros(isolate_fruits[i].shape[:2]) for i in range(len(isolate_fruits))]
    for i in range(len(bins)):
        for j in range(len(smooth_edges[i])):
            for r in range(bins[i].shape[0]):
                for c in range(bins[i].shape[1]):
                        if smooth_edges[i][j][r][c] != 0:
                            bins[i][r][c] += 1

    for i in range(len(bins)):
        for r in range(bins[i].shape[0]):
            for c in range(bins[i].shape[1]):
                if bins[i][r][c] > 1 and bins[i][r][c] < len(smooth_edges[i])-5:
                    bins[i][r][c] = 255
                else:
                    bins[i][r][c] = 0
    return bins