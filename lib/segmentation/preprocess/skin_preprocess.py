import cv2
import numpy as np
import math
import copy


class Prep(object):

    def __init__(self, path):
        self.__img = cv2.imread(path, cv2.IMREAD_COLOR)
        self.__imgray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
        self.__invimgray = self.__negate()
        self.__ottlvl = self.__OtsuAutoThresh()
        self.__binimg = self.__imBinarize()
        (self.__seg_col, self.__seg_gray) = self.__cvtBinToColAndGray()

    def __negate(self):
        inv_img = (self.__imgray).copy()
        (r, c) = inv_img.shape
        for x in range(0, r, 1):
            for y in range(0, c, 1):
                inv_img[x, y] = np.invert(inv_img[x, y])
        return inv_img

    def getColorPlates(self, src_clrimg, plate):
        temp_img = src_clrimg.copy()
        for x in temp_img:
            for y in x:
                if plate == 'B':
                    y[1] = 0
                    y[2] = 0
                elif plate == 'G':
                    y[0] = 0
                    y[2] = 0
                elif plate == 'R':
                    y[0] = 0
                    y[1] = 0
        return temp_img

    def __rmHoles(self, src_binimg):
        ffill_img = src_binimg.copy()
        mask = np.zeros((((ffill_img.shape)[0]) + 2, ((ffill_img.shape)[1]) + 2), np.uint8, 'C')
        cv2.floodFill(ffill_img, mask, (0, 0), 255)
        final_img = src_binimg | cv2.bitwise_not(ffill_img)
        return final_img

    def __OtsuAutoThresh(self):
        app_grlvls_wth_freq = getArrayOfGrayLevelsWithFreq(self.__invimgray)
        dt = np.dtype([('wcv', float), ('bcv', float), ('glvl', np.uint8)])
        var_ary = np.empty(0, dt, 'C')
        for x in range(0, app_grlvls_wth_freq.size, 1):
            thrslvl = (app_grlvls_wth_freq[x])[0]
            wb = 0.0
            mb = 0.0
            varb2 = 0.0
            wf = 0.0
            mf = 0.0
            varf2 = 0.0
            (wf, mf, varf2) = self.__threshSubPt(x, app_grlvls_wth_freq.size, app_grlvls_wth_freq, wf, mf, varf2)
            if (x == 0):
                pass
            else:
                (wb, mb, varb2) = self.__threshSubPt(0, x, app_grlvls_wth_freq, wb, mb, varb2)
            wcv = (wb * varb2) + (wf * varf2)
            bcv = (wb * wf) * math.pow((mb - mf), 2)
            var_ary = np.append(var_ary, np.array([(wcv, bcv, thrslvl)], dtype=dt), 0)
        quickSort(var_ary, 0, var_ary.size - 1)
        ottlvl = (var_ary[0])[2]
        return ottlvl

    def __threshSubPt(self, lower, upper, app_grlvls_wth_freq, w, m, var2):
        for h in range(lower, upper, 1):
            w = w + (app_grlvls_wth_freq[h])[1]
            m = m + float(np.uint32((app_grlvls_wth_freq[h])[0]) * np.uint32((app_grlvls_wth_freq[h])[1]))
        m = m / w
        for h in range(lower, upper, 1):
            var2 = var2 + float((math.pow((((app_grlvls_wth_freq[h])[0]) - m), 2)) * ((app_grlvls_wth_freq[h])[1]))
        var2 = var2 / w
        w = w / float((math.pow(app_grlvls_wth_freq.size, 2)))
        return (w, m, var2)

    def __imBinarize(self):
        binimg = np.zeros((self.__invimgray).shape, np.uint8, 'C')
        for x in range(0, ((self.__invimgray).shape)[0], 1):
            for y in range(0, ((self.__invimgray).shape)[1], 1):
                if (self.__invimgray[x, y] < self.__ottlvl):
                    binimg[x, y] = np.uint8(0)
                else:
                    binimg[x, y] = np.uint8(255)
        binimg = self.__rmHoles(binimg)
        return binimg

    def __cvtBinToColAndGray(self):
        seg_col = np.zeros((self.__img).shape, np.uint8, 'C')
        seg_gray = np.zeros((self.__imgray).shape, np.uint8, 'C')
        i = 0
        for x in seg_col:
            j = 0
            for y in x:
                if ((self.__binimg)[i, j] == 255):
                    y[0] = (self.__img)[i, j, 0]
                    y[1] = (self.__img)[i, j, 1]
                    y[2] = (self.__img)[i, j, 2]
                    seg_gray[i, j] = self.__imgray[i, j]
                j = j + 1
            i = i + 1
        return (seg_col, seg_gray)

    def getActImg(self):
        return self.__img

    def getGrayImg(self):
        return self.__imgray

    def getInvrtGrayImg(self):
        return self.__invimgray

    def getBinaryImg(self):
        return self.__binimg

    def getOtsuThresholdLevel(self):
        return self.__ottlvl

    def getSegColImg(self):
        return self.__seg_col

    def getSegGrayImg(self):
        return self.__seg_gra


def search(arr, ins_val, low, high):
    fnd_idx = -1
    if (arr.size == 0):
        pass
    else:
        while (low <= high):
            mid = int(low + ((high - low) / 2))
            if (ins_val > (arr[mid])[0]):
                low = mid + 1
                continue
            if (ins_val < (arr[mid])[0]):
                high = mid - 1
                continue
            if (ins_val == (arr[mid])[0]):
                fnd_idx = mid
                break
    return fnd_idx


def quickSort(arr, low, high):
    if low < high:
        pi = __partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


def __partition(arr, low, high):
    i = (low - 1)
    pivot = (arr[high])[0]
    for j in range(low, high, 1):
        if (arr[j])[0] <= pivot:
            i = i + 1
            temp = copy.deepcopy(arr[i])
            arr[i] = copy.deepcopy(arr[j])
            arr[j] = copy.deepcopy(temp)
    temp2 = copy.deepcopy(arr[i + 1])
    arr[i + 1] = copy.deepcopy(arr[high])
    arr[high] = copy.deepcopy(temp2)
    return (i + 1)


def __ins(arr, ins_val, index):
    if (arr.size == 0):
        arr = np.insert(arr, index, (ins_val, np.array([1], np.uint32)), 0)
        return arr
    else:
        fnd_idx = search(arr, ins_val, 0, arr.size - 1)
        if (fnd_idx >= 0):
            ((arr[fnd_idx])[1])[0] = np.uint32(((arr[fnd_idx])[1])[0]) + np.uint32(1)
            return arr
        else:
            while (index >= 0):
                if (ins_val > (arr[index - 1])[0]):
                    arr = np.insert(arr, index, (ins_val, np.array([1], np.uint32)), 0)
                    break
                if (ins_val < (arr[index - 1])[0]):
                    if (index == 0):
                        arr = np.insert(arr, index, (ins_val, np.array([1], np.uint32)), 0)
                    index = index - 1
                    continue
                else:
                    ((arr[index - 1])[1])[0] = np.uint32(((arr[index - 1])[1])[0]) + np.uint32(1)
                    break
            return arr


def getArrayOfGrayLevelsWithFreq(gray_img, lvldtype=np.uint8):
    aryoflst = np.empty(0, np.dtype([('glvl', lvldtype), ('freq', np.uint32, (1,))]), 'C')
    for x in range(0, (gray_img.shape)[0], 1):
        for y in range(0, (gray_img.shape)[1], 1):
            aryoflst = __ins(aryoflst, gray_img[x, y], index=aryoflst.size)
    return aryoflst
