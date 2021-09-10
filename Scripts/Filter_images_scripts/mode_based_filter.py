import os
#from PIL import Image
import collections
import shutil
import cv2 as cv


def main():
 path = '/mnt/data/haahm/Finalising_everything/testing/'
 # list of all content in a directory, filtered so only directories are returned
 dir_list = [path+directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
 print(dir_list)
 #for items in dir_list:
 listing = os.listdir(dir_list[1])
 print(len(listing))
 print(dir_list[1])

 counter_10 = 0
 for img in listing:
     file0=os.path.join(dir_list[1],img)

     im = cv.imread(file0)

     gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
     height, width = im.shape[0], im.shape[1]
     area_shape = height * width
     print(area_shape)

     ret, thresh = cv.threshold(gray, 100, 255, 0)
     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
     
     largest_area = 0;
     largest_contour_index = 0

     for cnt in contours:
         area = cv.contourArea(cnt)
         if (area > largest_area):
             largest_area = area
             largest_contour_index = cnt
     print('Largest contour', largest_area)
     discard_probability = (largest_area / area_shape) * 100
     print('Probability', discard_probability)

     new_folder = '/mnt/data/haahm/Finalising_everything_after_filter/testing/'

     f1 = (os.path.basename(dir_list[0]))
     f2 = (os.path.basename(dir_list[1]))
     f3 = (os.path.basename(dir_list[2]))
     folders = [f1,f2,f3]
     path_for_saving_image_3 = new_folder + f2 +'/'
     path_for_saving_image_2 = new_folder + f3 + '/'
     path_for_saving_disp_0 = new_folder + f1 + '/'

     if not os.path.exists(new_folder):
         for folder in folders:
          os.makedirs(os.path.join(new_folder,folder))


     if (discard_probability) <= 70:
         counter_10 += 1
         shutil.copy(file0,path_for_saving_image_3)

         file1=os.path.join(dir_list[2],img)
         shutil.copy(file1, path_for_saving_image_2)

         file2=os.path.join(dir_list[0],img)
         shutil.copy(file2, path_for_saving_disp_0)

def mode(inp_list):

    # calculate the frequency of each item
    data = collections.Counter(inp_list)
    #print(data)
    data_list = dict(data)
    # Print the items with frequency
   # print(data_list)

    # Find the highest frequency
    max_value = max(list(data.values()))
    print(max_value)
    mode_val = [num for num, freq in data_list.items() if freq == max_value]
    #print(mode_val)
    if len(mode_val) == len(inp_list):
        print("No mode in the list")
    else:
        return mode_val, max_value
        #print("The Mode of the list is : " + ', '.join(map(str, mode_val)))
        #print("The Most Occurance is : ", + max_value)


if __name__ == "__main__":
    main()