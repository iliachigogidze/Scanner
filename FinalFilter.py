import cv2
import numpy as np

# funqcias unda gaeces surati da 'mode' (Optional)
# mode = 'light' - abrunebs, sufta/msubuq versias
# mode = 'bold' - abrunebs muq/bold versias
# Default mode = 'light'
def scan(img, mode='light'):
    #convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dilate_img - teqsts shlis da ertmanetshi urevs
    # medianblur - adebs blars
    # am ori funqciis mizania mivigot foni, romelzec teqsti washlilia
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)

    # tu zemota funqciebma rame teqstis washla ver moaxerxa (didi, muqi teqsti, logo)
    # Canny ipovis mag teqsts da mere findContours gaigebs koordinatebs
    edge = cv2.Canny(bg_img, 50, 200)

    _, contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # findContour-ma sheidzleba ipovos ramdenime obieqti (teqsti/konturi).
    # for loopit chamovuvlit am obieqtebs da satitaod gavuketebt dilate-s, magram ufro didi Value-ti (60,60)
    for cnt in contours:
        if cv2.contourArea(cnt) > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            dilated_img[y:y + h, x:x + w] = cv2.dilate(dilated_img[y:y + h, x:x + w], np.ones((60, 60), np.uint8))

    bg_img = cv2.medianBlur(dilated_img, 21)

    # abdiff - sawyis/original surats vaklebt migebul fons da mere am yvelafers vaklebt 255-s (tetri)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    # migebul shedegs vuketebt normalizacias
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # tu parametrshi mode == 'light' vabrunebt light versias
    if mode == 'light':
        final_image = thr_img
        return final_image

    # tu parametrshi mode == 'bold' vabrunebt bold versias
    if mode == 'bold':
        thresh = 240
        maxValue = 255

        th, dst = cv2.threshold(thr_img, thresh, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C)
        dst = cv2.medianBlur(dst, 3)
        dst = cv2.dilate(dst, np.ones((1, 1), np.uint8))
        final_image = dst
        return final_image

