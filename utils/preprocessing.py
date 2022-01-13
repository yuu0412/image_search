import cv2
import os


def crop_image(path):
    """
        画像の余白を削除
        <参考>
        https://www.nishika.com/competitions/22/topics/169
    """
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert BGR to GRAY 
    img2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1] # しきい値処理
    contours = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] #輪郭抽出
    x1 = [] #x座標の最小値
    y1 = [] #y座標の最小値 
    x2 = [] #x座標の最大値
    y2 = [] #y座標の最大値
    for i in range(1, len(contours)):
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    if x1:
        x1_min = min(x1)
        y1_min = min(y1)
        x2_max = max(x2)
        y2_max = max(y2)
        crop_img = img[y1_min:y2_max, x1_min:x2_max]
    else:
        crop_img = img
    
    return crop_img

def make_crop(input):
    input_path, output_path, output_dir = input
    # print(input_path)
    crop_img = crop_image(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, crop_img)
    return

if __name__ == "__main__":
    test_input_path = "data/apply_images/1000000075/1000000075.jpg"
    test_output_path = "outputs/test/100000075.jpg"
    test_output_dir = "outputs/test"
    make_crop([test_input_path, test_output_path, test_output_dir])

