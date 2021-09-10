import numpy as np
import cv2

import Imath
import OpenEXR

def main():
    img = cv2.imread('F:\MSc_Embedded\Research Project\Files_from_University_PC\Results\Final_results_after_filter\Divided_images/ground_truth.png')
    width, height, channels = img.shape
    cv2.imshow('Image',img)
    cv2.waitKey(0)

    fov = (60 * np.pi) / 180
    theta = fov / 2
    f_new = (width - 1) /(2* np.tan(theta) )
    baseline = 30

    red_masked_image = masking(img[:,: ,2])
    green_masked_image = masking(img[:, :, 1])
    blue_masked_image = masking(img[:, :, 0])

    masked_image = np.copy(red_masked_image)
    masked_image = np.bitwise_or(masked_image, green_masked_image)
    masked_image = np.bitwise_or(masked_image, blue_masked_image)

    depth_map = (f_new * baseline)/masked_image
    cv2.imshow('Depth_Map',depth_map)
    cv2.waitKey(0)

    disp_cpy = np.copy(depth_map).astype(np.float32)
    cv2.imshow('Depth_Map', disp_cpy)
    cv2.waitKey()
    #np.save('/home/haahm/Development/projects/Results/depth_maps/working_with_julian/actuak_depth_map' + '.npy', depth_map)
    cv2.imwrite('F:\GitHub_Live_Projects\Research_Project/ground_truth_0.3.png', disp_cpy)
    #export_exr( disp_cpy,'/home/haahm/Development/projects/Results/Final_results_after_filter/depth_map/ground_truth_0.3' + '.exr')

def export_exr(map: np.ndarray, file_path: str):
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
    # source: https://excamera.com/articles/26/doc/intro.html
    pixels = map.tostring()
    header = OpenEXR.Header(map.shape[1], map.shape[0])
    channel = Imath.Channel(PIXEL_TYPE)
    header['channels'] = dict([(c, channel) for c in "RGB"])
    exr = OpenEXR.OutputFile(file_path, header)
    exr.writePixels({'R': pixels, 'G': pixels, 'B': pixels})
    exr.close()


def masking(img):

    invalid_mask_a = (img == 0) # For instance, 0 is disturbing in disparitiy maps if we invert them.
    invalid_mask_b = (img == np.inf)
    invalid_mask_c = (img == -np.inf)
    invalid_mask_d = (img != img) #looks strange but "NaN != NaN" -> detects NaNs
    # this stuff works also with multidimensional arrays (like color images)


    # combine all masks with OR operator
    invalid_mask = np.copy(invalid_mask_a) # make a copy if you need invalid_mask_a afterwards
    invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_b)
    invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_c)
    invalid_mask = np.bitwise_or(invalid_mask, invalid_mask_d)

    if len(img.shape) != 3:
        print("grayscale invalid mask:")
        print(invalid_mask)
        print()
        print("grayscale image before removing trouble making values:")
        print(img)
        print()
        print("grayscale image after removing trouble making values:")
        grayscale2 = np.copy(img)
        grayscale2[invalid_mask] = -1
        print(grayscale2)
        return grayscale2

    else:

        # expand grayscale mask
        invalid_mask_b = np.expand_dims(invalid_mask, axis=2)
        invalid_mask_g = np.copy(invalid_mask_b)
        invalid_mask_r = np.copy(invalid_mask_b) # make copies to avoid toughing blue mask values if you alter red mask values...
        invalid_mask_bgr = np.concatenate((invalid_mask_b, invalid_mask_g, invalid_mask_r), axis=2)

        # RGB / BGR masking
        image2 = np.copy(img)
        image2[invalid_mask_bgr] = -1 # just an exemplary value

        print("image red channel:")
        print(img[:,:,2])
        print("then...")
        print(image2[:,:,2])
        print()
        print("image green channel:")
        print(img[:,:,1])
        print("then...")
        print(image2[:,:,1])
        print()
        print("image blue channel:")
        print(img[:,:,0])
        print("then...")
        print(image2[:,:,0])
        print()
        return image2

if __name__ == '__main__':
    main()
