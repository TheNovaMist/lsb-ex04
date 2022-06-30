import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from skimage.metrics import structural_similarity as compare_ssim


def messageToBinary(message):
    """
    将要加密的字符串转换为二进制
    """
    if type(message) == str:
        # return ''.join([format(ord(i), "08b") for i in message])
        return ''.join(['{:0>8b}'.format(ord(x)) for x in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        # return [format(i, "08b") for i in message]
        return ['{:0>8b}'.format(i) for i in message]
    elif type(message) == int or type(message) == np.uint8:
        # return format(message, "08b")
        return '{:0>8b}'.format(message)
    else:
        raise TypeError("Input type not supported")


def hideData(image, secret_message, embed_bit):
    """ 将信息嵌入图片
    :param image:
    :param secret_message:
    :param embed_bit: 要嵌入的位 0-7
    :return: 成成的图片
    """
    # use for slice
    l_embed_bit = 8 - embed_bit - 1

    # calculate the maximum bytes to encode
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    print("Maximum bytes to encode:", n_bytes)

    # Check if the number of bytes to encode is less than the maximum bytes in the image
    if len(secret_message) > n_bytes:
        raise ValueError("Error encountered insufficient bytes, need bigger image or less data !!")

    secret_message += "#####"  # you can use any string as the delimeter

    data_index = 0
    # convert input data to binary format using messageToBinary() fucntion

    binary_secret_msg = messageToBinary(secret_message)

    data_len = len(binary_secret_msg)  # Find the length of data that needs to be hidden
    for values in image:
        for pixel in values:
            # convert RGB values to binary format
            r, g, b = messageToBinary(pixel)
            # modify the least significant bit only if there is still data to store
            if data_index < data_len:
                # hide the data into least significant bit of red pixel
                # pixel[0] = int(r[:-1] + binary_secret_msg[data_index], 2)
                pixel[0] = int(r[:l_embed_bit] + binary_secret_msg[data_index] + r[l_embed_bit + 1:], 2)
                data_index += 1
            if data_index < data_len:
                # hide the data into least significant bit of green pixel
                # pixel[1] = int(g[:-1] + binary_secret_msg[data_index], 2)
                pixel[1] = int(g[:l_embed_bit] + binary_secret_msg[data_index] + r[l_embed_bit + 1:], 2)
                data_index += 1
            if data_index < data_len:
                # hide the data into least significant bit of  blue pixel
                # pixel[2] = int(b[:-1] + binary_secret_msg[data_index], 2)
                pixel[2] = int(b[:l_embed_bit] + binary_secret_msg[data_index] + r[l_embed_bit + 1:], 2)
                data_index += 1
            # if data is encoded, just break out of the loop
            if data_index >= data_len:
                break

    return image


def showData(image, embed_bit):
    # use for index
    embed_bit = embed_bit + 1

    binary_data = ""
    for values in image:
        for pixel in values:
            r, g, b = messageToBinary(pixel)  # convert the red,green and blue values into binary format
            binary_data += r[-embed_bit]  # extracting data from the least significant bit of red pixel
            binary_data += g[-embed_bit]  # extracting data from the least significant bit of red pixel
            binary_data += b[-embed_bit]  # extracting data from the least significant bit of red pixel

    # split by 16-bits
    all_bytes = [binary_data[i: i + 8] for i in range(0, len(binary_data), 8)]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "#####":  # check if we have reached the delimeter which is "#####"
            break
    return decoded_data[:-5]  # remove the delimeter to show the original hidden message


def get_bitPlane(img):
    """
    将图片分解为各个位平面
    :param img:
    :return:
    """
    h, w, c = img.shape
    bitPlane = np.zeros(shape=(h, w, 8, c))
    for c_i in range(c):
        flag = 0b00000001
        for bit_i in range(bitPlane.shape[-2]):
            bp = img[..., c_i] & flag
            bp[bp != 0] = 1  # 阈值处理 非0即1
            bitPlane[..., bit_i, c_i] = bp  # 处理后的数据载入到某个位平面
            flag <<= 1  # 获取下一位信息

    bitPlane = bitPlane.astype(np.uint8)

    images = []
    h, w, b, c = bitPlane.shape

    # 单位平面
    for i in range(b):  # 8
        plane = np.zeros((h, w, c), dtype=np.uint8)
        for c_i in range(c):  # 3
            plane[:, :, c_i] = bitPlane[:, :, i, c_i] * np.power(2, i)
        m = plane[:, :] > 0
        plane[m] = 255
        images.append(plane)

    return images


def show_bitPlane(image):
    # 图片各个位平面
    images = get_bitPlane(image)
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('./dog.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    w, h, c = image.shape
    print(image.shape)
    image2 = image.copy()  # 备份原图像

    # 显示各个位平面
    show_bitPlane(image)

    # 读取文件
    file = open('三体.txt', 'r', encoding='utf-8')
    with file:
        message = file.read()

    # 使用base64编码utf-8字符串 因为含有中文
    message = message.encode('utf-8')
    message = base64.b64encode(message).decode('ascii')  # 变为ascii str

    # print(message)

    # 画图
    plt.figure(0)
    for i in range(8):
        encoded_image = hideData(image, message, i)

        # 图像评估
        psnr = compare_psnr(encoded_image, image2)
        psnr = round(psnr, 2)  # 保留小数点二位数
        print(psnr)
        ssim = compare_ssim(encoded_image, image2, multichannel=True)
        ssim = round(ssim, 2)
        print(ssim)

        plt.subplot(2, 4, i + 1)
        plt.imshow(encoded_image)
        plt.title(f'位平面{i}')
        text = f'psnr:{psnr}\nssim:{ssim}'
        plt.text(w / 4, h * 1.3, text)  # 在画板上显示文字
        plt.axis("off")  # 不显示坐标轴
    plt.show()

    # 选择在任意一位平面嵌入文本的图像 用来测试提取文本
    encoded_image = hideData(image, message, 7)
    text = showData(encoded_image, 7)
    print(base64.b64decode(text).decode('utf-8'))

