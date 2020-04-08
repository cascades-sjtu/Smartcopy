import os.path
import cv2
import numpy as np
import PySimpleGUI as sg

'''
这是谌翔的应用软件课程设计大作业-智能图片裁剪工具
本项目基于python-3.7.6+opencv-4.1.0+PySimpleGUI-4.18.0实现
核心算法为opencv中的canny算法，详见官方文档http://www.woshicver.com/
本项目已上传github：https://github.com/cascades-sjtu/smartcrop_gui
'''

# ---------- 检测图像边缘，返回裁剪顶点信息 ---------- 
def vertex_detect(image_path, edge_threshold, variance_threshold):
    image = cv2.imread(image_path)  
    # 调用canny算子进行边缘检测
    ratio = 4
    kernel_size = 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gauss, edge_threshold, ratio * edge_threshold, apertureSize=kernel_size)
    # 纵向分割为 32 个块，每个块都是正方形
    blocksize = 32
    # 求出每一块的边长cellpx（以像素为单位），shape[0] 为图片高度，shape[1] 为图片宽度，rows和cols分别为行数和列数
    cellPx = image.shape[1] // blocksize
    rows = image.shape[0] // cellPx
    cols = blocksize
    # 每一块的像素数量
    blockPx = cellPx * cellPx
    # 对图片的RGB总和初始化
    imgR = 0
    imgG = 0
    imgB = 0
    # 每一块是否包含边界
    has_edge = [[0] * cols] * rows
        # 用二维数组存储并遍历每一块的xx值，将对应块处理为白色或者黑色
    cellValues = [0] * (rows)
    for i in range(rows):
        # 对于每一行，以列数64为大小创建数组
        cellValues[i] = [0] * blocksize
        for j in range(cols):
            # 求出当前块的行和列的起始和终止的像素位置
            rowbeg = cellPx * i
            rowend = rowbeg + cellPx
            colbeg = cellPx * j
            colend = colbeg + cellPx
            # 对当前块的RGB总和以及描述是否包含边界的bool变量初始化
            r = 0
            g = 0
            b = 0
            # 用二维数组遍历当前块内的每个像素，求出当前块是否为边界值和RGB值
            for ii in range(rowbeg, rowend):
                for jj in range(colbeg, colend):
                    # 利用edge图检测某一像素是否包含边界
                    if edge[ii][jj] > 0:
                        has_edge[i][j] = 1
                    # 将每一个像素的RGB值累计到当前块
                    r = r + image[ii][jj][0]
                    g = g + image[ii][jj][1]
                    b = b + image[ii][jj][2]

            # 将每一块的RGB总和累计到整张图
            imgR = imgR + r
            imgG = imgG + g
            imgB = imgB + b
            # 求出当前块每一个像素的RGB平均值作为该块的RGB值
            rv = int(r / blockPx)
            gv = int(g / blockPx)
            bv = int(b / blockPx)
            # 取平均rgb为该块的rgb
            value = [rv, gv, bv]
            cellValues[i][j] = value
    # 图片包含总像素个数
    imgCells = (blockPx) * (rows * cols)
    # 求出整张图片中单像素的RGB均值
    avgR = imgR // imgCells
    avgG = imgG // imgCells
    avgB = imgB // imgCells
    # 初始化裁剪边缘，在遍历每一块的过程中不断更新
    up = image.shape[0]
    down = 0
    left = image.shape[1]
    right = 0
    # 遍历每一块
    for i in range(len(cellValues)):
        for j in range(len(cellValues[i])):
            # 求每一块的RGB值与块平均RGB值的差异绝对值
            rdiff = abs(cellValues[i][j][0] - avgR)
            gdiff = abs(cellValues[i][j][1] - avgG)
            bdiff = abs(cellValues[i][j][2] - avgB)
            # 求出当前块的行和列的起始和终止的像素位置
            rowbeg = cellPx * i
            rowend = rowbeg + cellPx
            colbeg = cellPx * j
            colend = colbeg + cellPx
            # 判断是否超过阈值和是否为黑色和rgb是否超过阈值
            pxDiff = True if (rdiff > variance_threshold or
                              gdiff > variance_threshold or
                              bdiff > variance_threshold) else False
            isBlack = True if (cellValues[i][j][0] < 30 and
                               cellValues[i][j][1] < 30 and
                               cellValues[i][j][2] < 30) else False
            # 如果在阈值内且和不为黑色，则将该块标记为白色
            if pxDiff and (isBlack == False) and has_edge[i][j]:
                # 更新裁剪边缘
                up = min(up, rowend)
                down = max(down, rowbeg)
                left = min(left, colend)
                right = max(right, colbeg)
    # 更新裁剪信息，显示原始图片
    crop_message = image_path + '已经完成裁剪检测'
    window['-CROP MESSAGE-'].update(crop_message)
    current_image = '当前裁剪图片为' + image_path
    window['-CURRENT IMAGE-'].update(current_image)
    image = cv2.resize(image, (600, 450))
    window['-IMAGE-'].update(data=cv2.imencode('.png', image)[1].tobytes())  
    return up, down, left, right

# ---------- 根据顶点图生成预览，返回裁剪区域 ---------- 
def preview(image_path, crop_vertex, mode, ratio):
    image = cv2.imread(image_path)
    # 矩形裁剪
    if mode == 'rectangle':
        # 按照比例裁剪
        if not (ratio == (0, 0)):
            center_x = (crop_vertex[2] + crop_vertex[3]) // 2
            center_y = (crop_vertex[0] + crop_vertex[1]) // 2
            row = crop_vertex[1] - crop_vertex[0]
            col = crop_vertex[3] - crop_vertex[2]
            # 判断目前的比例情况，以相对长的一边为基准，按比例放大另一边
            if ratio[0] * row >= ratio[1] * col:
                col = row * ratio[0] // ratio[1]
            else:
                row = col * ratio[1] // ratio[0]
            # 如果越界则按比例缩小
            if (center_y - row //2 < 0):
                row = 2 * center_y
                col = row * ratio[0] // ratio[1]
            if (center_y + row // 2 > image.shape[0]):
                row = 2 * (image.shape[0] - center_y)
                col = row * ratio[0] // ratio[1]
            if (center_x - col //2 < 0):
                col = 2 * center_x
                row = col * ratio[1] // ratio[0]
            if (center_x + col //2 > image.shape[1]):
                col = 2 * (image.shape[1] - center_x)
                row = col * ratio[1] // ratio[0]
            crop_area = (center_y - row // 2, center_y + row // 2, center_x - col // 2, center_x + col // 2)
        # 生成预览裁剪图
        preview_image = cv2.rectangle(image, (crop_area[2], crop_area[0]), (crop_area[3], crop_area[1]), (0, 0, 0), 2)
        
    # 圆形裁剪
    if mode == 'circle':
        # 初始化圆形参数
        round_x = 0
        round_y = 0
        round_radius = 0
        # 以矩形长宽的平均值为裁剪圆的直径
        round_x = (crop_vertex[2] + crop_vertex[3]) // 2
        round_y = (crop_vertex[0] + crop_vertex[1]) // 2
        round_radius = (crop_vertex[3] - crop_vertex[2] + crop_vertex[1] - crop_vertex[0]) // 4
        # 如果裁剪区域超过原图则采用内切圆
        if (round_x - round_radius < 0 or round_y - round_radius < 0 or round_y + round_radius > image.shape[0] or round_x + round_radius > image.shape[1]):
            round_radius = min(crop_vertex[3] - crop_vertex[2], crop_vertex[1] - crop_vertex[0])//2
        # 生成裁剪预览图
        preview_image = cv2.circle(image, (round_x, round_y), round_radius, (0, 0, 0), 2)
        crop_area = (round_x, round_y, round_radius)

    # 返回并更新裁剪信息，生成预览
    crop_message = '当前裁剪参数为\t' + str(crop_area) + '\t' + mode + '\t' + str(ratio)
    window['-CROP MESSAGE-'].update(crop_message)
    preview_image = cv2.resize(preview_image, (600, 450))
    window['-IMAGE-'].update(data=cv2.imencode('.png', preview_image)[1].tobytes())
    return crop_area

# ---------- 进行裁剪，并切换到下一图片 ---------- 
def crop(image_path, crop_area, mode):
    image = cv2.imread(image_path)
    # 裁剪出矩形图片
    if mode == 'rectangle':
        crop_image = image[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]]

    # 裁剪出圆形图片
    if mode == 'circle':
        # 创建四通道的新图片，前三通道复制原图，第四通道为透明通道
        rows, cols, channels = image.shape
        crop_image = np.zeros((rows, cols, 4), np.uint8)
        crop_image[:,:, 0:3] = image[:,:, 0:3]
        # 创建单通道图片,裁剪圆部分设置为不透明
        circle = np.zeros((rows, cols, 1), np.uint8)
        circle[:,:,:] = 0
        circle = cv2.circle(circle, (crop_area[0], crop_area[1]), crop_area[2], (255), -1)   
        # 融合图片
        crop_image[:,:, 3] = circle[:,:, 0]

    # 更新裁剪情况，输出新图片
    crop_message = image_path + '已经完成裁剪'
    window['-CROP MESSAGE-'].update(crop_message)
    new_image_name = 'crop_' + image_path
    cv2.imwrite(new_image_name, crop_image)

# ---------- 读取键盘输入，对预览图片进行调整 ---------- 
def modify(image_path, crop_area, mode, ratio, event):
    image = cv2.imread(image_path)
    crop_area = list(crop_area)
    # 针对矩形的调整
    if mode=='rectangle':
        if event == 'w':
            crop_area[0] = crop_area[0] - 5
            crop_area[1] = crop_area[1] - 5
        elif event == 's':
            crop_area[0] = crop_area[0] + 5
            crop_area[1] = crop_area[1] + 5
        elif event == 'a':
            crop_area[2] = crop_area[2] - 5
            crop_area[3] = crop_area[3] - 5
        elif event == 'd':
            crop_area[2] = crop_area[2] + 5
            crop_area[3] = crop_area[3] + 5
        if event == '1':
            crop_area[0] = crop_area[0] - ratio[1]
            crop_area[1] = crop_area[1] + ratio[1]
            crop_area[2] = crop_area[2] - ratio[0]
            crop_area[3] = crop_area[3] + ratio[0]
        if event == '2':
            crop_area[0] = crop_area[0] + ratio[1]
            crop_area[1] = crop_area[1] - ratio[1]
            crop_area[2] = crop_area[2] + ratio[0]
            crop_area[3] = crop_area[3] - ratio[0]
        preview_image = cv2.rectangle(image, (crop_area[2], crop_area[0]), (crop_area[3], crop_area[1]), (0, 0, 0), 2)
    # 针对圆形的调整
    elif mode == 'circle':
        if event == 'w':
            crop_area[1] = crop_area[1] - 5
        elif event == 's':
            crop_area[1] = crop_area[1] + 5
        elif event == 'a':
            crop_area[0] = crop_area[0] - 5
        elif event == 'd':
            crop_area[0] = crop_area[0] + 5
        if event == '1':
            crop_area[2] = crop_area[2] + 5
        if event == '2':
            crop_area[2] = crop_area[2] - 5
        preview_image = cv2.circle(image, (crop_area[0], crop_area[1]), crop_area[2], (0, 0, 0), 2)

    # 返回并更新裁剪信息，生成预览
    crop_message = '当前裁剪参数为\t' + str(crop_area) + '\t' + mode + '\t' + str(ratio)
    window['-CROP MESSAGE-'].update(crop_message)
    preview_image = cv2.resize(preview_image, (600, 450))
    window['-IMAGE-'].update(data=cv2.imencode('.png', preview_image)[1].tobytes())
    return crop_area

# -------------------- GUI -------------------- 

# ---------- 设置gui主题 ----------  
sg.theme('Tan')

# ---------- 设置gui各部分结构 ----------

# ---------- 左侧为控制部分 ---------- 
control_col = [
    [sg.Text('请输入图片所在文件夹')],
    [sg.Input(key='-FOLDER PATH-', size=(12, 1)), sg.FolderBrowse(size=(6, 1))],
    [sg.Open(size=(8, 1)), sg.Cancel(size=(8, 1))],
    [sg.Listbox(values=[], key='-IMAGE LIST-', size=(19, 25), enable_events=True)],
    [sg.Text('当前裁剪图片为',key='-CURRENT IMAGE-',size=(18,1))]
]

# ---------- 右侧为图片显示部分 ---------- 
ratio_list = [(1, 1), (3, 2), (4, 3), (16, 9), (46, 20)]
image_col = [
    [sg.InputCombo(values=('rectangle', 'circle'), default_value='rectangle', key='-MODE-', readonly=True),
    sg.InputCombo(values=ratio_list, default_value=(1, 1), key='-RATIO-', readonly=True),
    sg.Text('上下左右 w s a d / 放大缩小 1 2')],
    [sg.Button('裁剪预览', size=(10, 1), key='-PREVIEW-'),
    sg.Button('实施裁剪', size=(10,1), key='-CROP-')],
    [sg.Text('当前裁剪情况为', key='-CROP MESSAGE-', size=(50, 1))],
    [sg.Image(filename='cover.png', key='-IMAGE-', size=(600, 450))]
]

# ---------- 设置layout全局 ---------- 
layout = [
    [sg.Column(control_col), sg.VSeperator(), sg.Column(image_col)]
]

# ---------- 生成窗口 ---------- 
window = sg.Window('智能图片裁剪工具', layout, grab_anywhere=True, return_keyboard_events=True, use_default_focus=True)

# ---------- 初始化循环变量 ---------- 
crop_vertex = (0, 0, 0, 0)
crop_area = (0, 0, 0, 0)
image_index = 0
keyboard_input = ('w', 's', 'a', 'd', '1', '2')

# ---------- 循环处理事件 ---------- 
while True:
    event, values = window.read()
    # 关闭窗口
    if event in (None, 'Cancel'):
        break

    # 从文件夹读入图片
    elif event == 'Open':
        folder_path = values['-FOLDER PATH-']
        os.chdir(folder_path)
        image_list = os.listdir(folder_path)
        # 检测是否为图片文件
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        image_list = [f for f in image_list if os.path.isfile(f) and f.lower().endswith(img_types)]
        window['-IMAGE LIST-'].update(image_list)
        
    # 从图片列表读入图片，进行边缘检测
    elif event == '-IMAGE LIST-':
        image_path = values['-IMAGE LIST-'][0]
        crop_vertex = vertex_detect(image_path, 100, 25)
    
    # 根据边缘检测进行预览
    elif event == '-PREVIEW-':
        mode = values['-MODE-']
        ratio = values['-RATIO-']
        crop_area = preview(image_path, crop_vertex, mode, ratio)
        
    # 实施裁剪，跳转到下一图片
    elif event == '-CROP-':
        crop(image_path, crop_area, mode)
        # 跳转到下一张图片
        image_index = image_list.index(image_path)
        image_index = image_index + 1
        image_path = image_list[image_index]
        vertex_detect(image_path, 100, 25)

    # 读入键盘输入进行调整
    elif event in keyboard_input:
        crop_area = modify(image_path, crop_area, mode, ratio, event)

# ---------- 关闭窗口 ---------- 
window.close()
