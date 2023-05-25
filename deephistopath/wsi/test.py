import os
os.add_dll_directory(r"D:\openslide-win64-20221217\bin")

import time

import numpy as np

import slide, filter, util, tiles
import matplotlib

# slide.slide_info()
# slide.slide_stats()
# slide.singleprocess_training_slides_to_images()

# slide_path = "./11T000413JP2.svs"
# slide.training_slide_to_image(1, slide_path)
# img_path = slide.get_training_image_path(1, slide_path)
# img = slide.open_image(img_path)
# util.ADDITIONAL_NP_STATS = False
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB") # 生成图像文件会自动加密，所以会出现提示弹窗而不能打开文件

'''
"""
阈值
"""

# # 生成灰度图
grayscale = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(grayscale, "Grayscale")

# # 生成灰度反转图，255变0，0变255
complement = filter.filter_complement(grayscale)
# util.display_img(complement, "Complement")

# # 对灰度反转图使用阈值处理生成二值图像，在阈值门槛以上生成True，在阈值门槛以下生成False
thresh = filter.filter_threshold(complement, threshold=100)
# util.display_img(thresh, "Threshold")

# # 对灰度反转图使用滞后阈值
hyst = filter.filter_hysteresis_threshold(complement)
# util.display_img(hyst, "Hystersis Threshold")

# # 对灰度反转图使用otsu
otsu = filter.filter_otsu_threshold(complement)
# util.display_img(otsu, "otsu Threshold")

# # 对灰度反转图使用对比拉伸
contrast_stretch = filter.filter_contrast_stretch(complement, low=100, high=200)
# util.display_img(contrast_stretch, "Contrast Stretch")

# # 对灰度图使用直方图均衡化
hist_equ = filter.filter_histogram_equalization(grayscale)
# util.display_img(hist_equ, "Histogram Equalization")

# # 对灰度图使用自适应均衡化
adaptive_equ = filter.filter_adaptive_equalization(grayscale)
# util.display_img(adaptive_equ, "Adaptive Equalization")


# # 将RGB图像转换为HED图像，然后获得苏木精(hematoxylin)和伊红(eosin)通道滤镜下展示的图像
hed = filter.filter_rgb_to_hed(rgb)
hema = filter.filter_hed_to_hematoxylin(hed)
eosin = filter.filter_hed_to_eosin(hed)
# util.display_img(hema, "Hematoxylin Channel")
# util.display_img(eosin, "Eosin Channel")

# # 将RGB图像在绿色通道滤镜下展示。 紫色和粉色在色环中是绿色的对位，白色的绿色通道值很高，所以在绿色滤镜下，白色背景为黑色，紫粉色组织为白色
not_green = filter.filter_green_channel(rgb)
# util.display_img(not_green, "Green Channel Filter")

# # 将RGB图像用灰度滤镜过滤, 灰色色素值为(128,128,128)，灰度滤镜能过滤掉白色和黑色
not_grays = filter.filter_grays(rgb)
# util.display_img(not_grays, "Grays Filter")


"""
颜色
"""

# # 红色滤镜，将RGB图像中的红色部分提取出来（红变黑）
not_red = filter.filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90,
                            display_np_info=True)
# util.display_img(not_red, "Red Filter (150, 80, 90)")
# # 将红色滤镜和逆红色滤镜作为蒙版应用于原始图像（红色变黑色）
# util.display_img(util.mask_rgb(rgb, not_red), "Not Red")
# util.display_img(util.mask_rgb(rgb, ~not_red), "Red")

# # 红笔滤镜，更激进的红色滤镜，过滤除去更多的红色
not_red_pen = filter.filter_red_pen(rgb)
# util.display_img(not_red_pen, "Red Pen Filter")
# # 将红笔滤镜和逆红笔滤镜作为蒙版应用于原始图像（红色变黑色）
# util.display_img(util.mask_rgb(rgb, not_red_pen), "Not Red Pen")
# util.display_img(util.mask_rgb(rgb, ~not_red_pen), "Red Pen")

# # 蓝色滤镜，将RGB图像中的蓝色部分提取出来（蓝变黑）
not_blue = filter.filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180,
                              display_np_info=True)
# util.display_img(not_blue, "Blue Filter (130, 155, 180)")
# # 将蓝色滤镜和逆蓝色滤镜作为蒙版（蓝色变黑色）
# util.display_img(util.mask_rgb(rgb, not_blue), "Not Blue")
# util.display_img(util.mask_rgb(rgb, ~not_blue), "Blue")

# # 蓝笔滤镜，更激进的蓝色滤镜，过滤除去更多的蓝色
not_blue_pen = filter.filter_blue_pen(rgb)
# util.display_img(not_blue_pen, "Blue Pen Filter")
# # 将蓝笔滤镜作为蒙版应用于原始图像（蓝色变黑色）
# util.display_img(util.mask_rgb(rgb, not_blue_pen), "Not Blue Pen")
# util.display_img(util.mask_rgb(rgb, ~not_blue_pen), "Blue Pen")
# # 量化filter_blue()和filter_blue_pen()过滤结果之间的差异
# print("filter_blue: " + filter.mask_percentage_text(filter.mask_percent(not_blue)))
# print("filter_blue_pen: " + filter.mask_percentage_text(filter.mask_percent(not_blue_pen)))

# # 绿色滤镜，将RGB图像中的绿色部分提取出来（绿变黑）
not_green = filter.filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140,
                                display_np_info=True)
# util.display_img(not_green, "RGB")
# # 将绿色滤镜和逆绿色滤镜作为蒙版（绿色变黑色）
# util.display_img(util.mask_rgb(rgb, not_green), "Not Green")
# util.display_img(util.mask_rgb(rgb, ~not_green), "Green")

# k-means聚类分割(otsu前后)
kmeans_seg = filter.filter_kmeans_segmentation(rgb, n_segments=3000)
otsu_mask = util.mask_rgb(rgb,
                          filter.filter_otsu_threshold(filter.filter_complement(filter.filter_rgb_to_grayscale(rgb)),
                                                       output_type="bool"))
kmeans_seg_otsu = filter.filter_kmeans_segmentation(otsu_mask, n_segments=3000)
# util.display_img(kmeans_seg, "K-Means Segmentation", bg=True)
# util.display_img(otsu_mask, "Image after Otsu Mask", bg=True)
# util.display_img(kmeans_seg_otsu, "K-Means Segmentation after Otsu Mask", bg=True)

# # RAG将颜色相近的区域组合起来
rag_thresh_9 = filter.filter_rag_threshold(rgb)
rag_thresh_1 = filter.filter_rag_threshold(rgb, threshold=1)
rag_thresh_20 = filter.filter_rag_threshold(rgb, threshold=20)
# util.display_img(rag_thresh_9, "RAG Threshold (9)", bg=True)
# util.display_img(rag_thresh_1, "RAG Threshold (1)", bg=True)
# util.display_img(rag_thresh_20, "RAG Threshold (20)", bg=True)

# # RGB->HSV， 将rgb图像转换为hsv展示
# matplotlib.use('Agg')
# tiles.display_image_with_rgb_and_hsv_histograms(rgb)

"""
形态学
主要的形态学操作包括腐蚀、膨胀、开和闭。 开运算是先侵蚀后膨胀，闭运算是先膨胀后侵蚀。 通过形态学运算，某种结构元素（正方形、圆形、十字等）会沿着对象
的边缘传递。 形态学运算通常在二值或灰度图像上执行。在我们的示例中，我们将形态学运算应用于二进制图像（True/False, 1.0/0.0, 255/0）
"""
# # 圆盘侵蚀（erosion）， 5px和20px半径
no_grays = filter.filter_grays(rgb, output_type="bool")
bin_erosion_5 = filter.filter_binary_erosion(no_grays, disk_size=5)
bin_erosion_20 = filter.filter_binary_erosion(no_grays, disk_size=20)
# util.display_img(no_grays, "No Grays", bg=True)
# util.display_img(bin_erosion_5, "Binary Erosion (5)", bg=True)
# util.display_img(bin_erosion_20, "Binary Erosion (20)", bg=True)

# # 圆盘膨胀（Dilation）， 5px和20px半径
bin_dilation_5 = filter.filter_binary_dilation(no_grays, disk_size=5)
bin_dilation_20 = filter.filter_binary_dilation(no_grays, disk_size=20)
# util.display_img(bin_dilation_5, "Binary Dilation (5)", bg=True)
# util.display_img(bin_dilation_20, "Binary Dilation (20)", bg=True)

# # 开操作
bin_opening_5 = filter.filter_binary_opening(no_grays, disk_size=5)
bin_opening_20 = filter.filter_binary_opening(no_grays, disk_size=20)
# util.display_img(bin_opening_5, "Binary Opening (5)", bg=True)
# util.display_img(bin_opening_20, "Binary Opening (20)", bg=True)

# # 闭操作
bin_closing_5 = filter.filter_binary_closing(no_grays, disk_size=5)
bin_closing_20 = filter.filter_binary_closing(no_grays, disk_size=20)
# util.display_img(bin_closing_5, "Binary Closing (5)", bg=True)
# util.display_img(bin_closing_20, "Binary Closing (20)", bg=True)

# # 移除噪声
remove_small_objs_100 = filter.filter_remove_small_objects(no_grays, min_size=100)
remove_small_objs_10000 = filter.filter_remove_small_objects(no_grays, min_size=10000)
# util.display_img(remove_small_objs_100, "Remove Small Objects (100)", bg=True)
# util.display_img(remove_small_objs_10000, "Remove Small Objects (10000)", bg=True)

# # 去除小孔
remove_small_holes_100 = filter.filter_remove_small_holes(no_grays, min_size=100)
remove_small_holes_10000 = filter.filter_remove_small_holes(no_grays, min_size=10000)
# util.display_img(remove_small_holes_100, "Remove Small Holes (100)", bg=True)
# util.display_img(remove_small_holes_10000, "Remove Small Holes (10000)", bg=True)

# # 补上小孔
fill_holes = filter.filter_binary_fill_holes(no_grays)
remove_holes_100 = filter.filter_remove_small_holes(no_grays, min_size=100, output_type="bool")
remove_holes_10000 = filter.filter_remove_small_holes(no_grays, min_size=10000, output_type="bool")
# util.display_img(fill_holes, "Fill Holes", bg=True)
# util.display_img(fill_holes ^ remove_holes_100, "Differences between Fill Holes and Remove Small Holes (100)", bg=True)
# util.display_img(fill_holes ^ remove_holes_10000, "Differences between Fill Holes and Remove Small Holes (10000)", bg=True)


# 熵
# 根据复杂度来过滤图像，切片的背景部分更不复杂

gray = filter.filter_rgb_to_grayscale(rgb)
entropy = filter.filter_entropy(gray, output_type="bool")
# util.display_img(gray, "Grayscale")
# util.display_img(entropy, "Entropy")
# util.display_img(util.mask_rgb(rgb, entropy), "Original with Entropy Mask")
# util.display_img(util.mask_rgb(rgb, ~entropy), "Original with Inverse of Entropy Mask")


# Canny边缘检测
# 图像的边缘是会出现图像亮度显著陡峭变化的部分。Canny边缘检测算法则是基于此


gray = filter.filter_rgb_to_grayscale(rgb)
canny = filter.filter_canny(gray, output_type="bool")
rgb_crop = rgb[300:900, 300:900]
canny_crop = canny[300:900, 300:900]
# util.display_img(canny, "Canny", bg=True)
# util.display_img(rgb_crop, "Original", size=24, bg=True)
# util.display_img(util.mask_rgb(rgb_crop, ~canny_crop), "Original with ~Canny Mask", size=24, bg=True)



"""
混合滤镜
把多种滤镜混合使用
"""

# 去蓝笔和绿笔
no_green_pen = filter.filter_green_pen(rgb)
no_blue_pen = filter.filter_blue_pen(rgb)
no_gp_bp = no_green_pen & no_blue_pen
# util.display_img(no_green_pen, "No Green Pen")
# util.display_img(no_blue_pen, "No Blue Pen")
# util.display_img(no_gp_bp, "No Green Pen, No Blue Pen")
# util.display_img(util.mask_rgb(rgb, no_gp_bp), "Original with No Green Pen, No Blue Pen")

# 去蓝笔、绿笔、无灰色，绿通道、无小物体
mask = filter.filter_grays(rgb) & filter.filter_green_channel(rgb) & filter.filter_green_pen(rgb) & filter.filter_blue_pen(rgb)
mask = filter.filter_remove_small_objects(mask, min_size=100, output_type="bool")
# util.display_img(mask, "No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
# util.display_img(util.mask_rgb(rgb, mask), "Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
# util.display_img(util.mask_rgb(rgb, ~mask), "Original with Inverse Mask")

# 去除覆盖在组织上的灰绿色笔迹
rgb, _ = filter.apply_filters_to_image(1, slide_path, display=False, save=False)
not_greenish = filter.filter_green(rgb, red_upper_thresh=125, green_lower_thresh=30, blue_lower_thresh=30, display_np_info=True)
not_grayish = filter.filter_grays(rgb, tolerance=30)
rgb_new = util.mask_rgb(rgb, not_greenish & not_grayish)

row1 = np.concatenate((rgb[1200:1800, 150:750], rgb[1150:1750, 2050:2650]), axis=1)
row2 = np.concatenate((rgb_new[1200:1800, 150:750], rgb_new[1150:1750, 2050:2650]), axis=1)
result = np.concatenate((row1, row2), axis=0)
# util.display_img(result)


# Filter Example 滤镜示例
mask_not_green = filter.filter_green_channel(rgb)
mask_not_gray = filter.filter_grays(rgb)
mask_no_red_pen = filter.filter_red_pen(rgb)
mask_no_green_pen = filter.filter_green_pen(rgb)
mask_no_blue_pen = filter.filter_blue_pen(rgb)
mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
mask_remove_small = filter.filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
result = util.mask_rgb(rgb, mask_remove_small)
# util.display_img(result)
'''

# # 将滤镜应用到多个图像上
# filter.singleprocess_apply_filters_to_images(html=False)

"""
生成块区域
出于可视化的目的，每个tile的组织含量被用不同颜色标注。若组织含量超过80%，tile为绿色；若组织含量在10%~80%之间，tile为黄色；若组织含量在0~10%之
间，tile为橘色；若没有组织，则tile为红色。
热图的阈值可以通过TISSUE_HIGH_THRESH和TISSUE_LOW_THRESH两个常量进行调整，默认值为80%和10%。热图的颜色可以通过HIGH_COLOR、MEDIUM_COLOR、
LOW_COLOR和NONE_COLOR常量来调整。
Tile的边界宽度默认值被设为2，tile的长宽由ROW_TILE_SIZE和COL_TILE_SIZE设置，值分别为1024
为了产生一个单独切片的tiles，我们使用summary_and_tiles()功能，能够产生tile的总结并返回一个切片的高分tiles
"""


# tiles.summary_and_tiles(15, slide_path, save_summary=True, save_data=True, save_top_tiles=True)

# slide.slide_info()
# slide.singleprocess_training_slides_to_images()
# filter.singleprocess_apply_filters_to_images()
# tiles.singleprocess_filtered_images_to_tiles()


def main():
    slide.slide_info()
    slide.singleprocess_training_slides_to_images()
    filter.singleprocess_apply_filters_to_images()
    tiles.singleprocess_filtered_images_to_tiles()
    # slide.multiprocess_training_slides_to_images()
    # filter.multiprocess_apply_filters_to_images()
    # tiles.multiprocess_filtered_images_to_tiles()


if __name__ == "__main__":
    main()
