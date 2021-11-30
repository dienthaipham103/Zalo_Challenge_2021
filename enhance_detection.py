import numpy as np

def get_coors(box, classify=False):
    if classify == True:
        _, _, xmin, ymin, xmax, ymax = box
    else:
        xmin, ymin, xmax, ymax, _ = box
    return xmin, ymin, xmax, ymax


def intersact_area(box1, box2, classify=False):
    xmin1, ymin1, xmax1, ymax1 = get_coors(box1, classify=classify)
    xmin2, ymin2, xmax2, ymax2 = get_coors(box2, classify=classify)

    if (xmin1 >= xmax2) or (xmax1 <= xmin2) or (ymin1 >= ymax2) or (ymax1 <= ymin2):
        return 0
    else:
        xmin = max(xmin1, xmin2)
        xmax = min(xmax1, xmax2)
        ymin = max(ymin1, ymin2)
        ymax = min(ymax1, ymax2)
        return (xmax-xmin)*(ymax-ymin)

def overlap_ratio(box1, box2, classify=False):
    xmin1, ymin1, xmax1, ymax1 = get_coors(box1, classify=classify)
    xmin2, ymin2, xmax2, ymax2 = get_coors(box2, classify=classify)

    overlap_area = intersact_area(box1, box2, classify=classify)
    union_area = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2) - overlap_area
    return overlap_area/union_area

def merge_box(box1, box2, classify=False):
    xmin1, ymin1, xmax1, ymax1 = get_coors(box1, classify=classify)
    xmin2, ymin2, xmax2, ymax2 = get_coors(box2, classify=classify)
    
    xmin = (xmin1+xmin2)/2
    ymin = (ymin1+ymin2)/2
    xmax = (xmax1+xmax2)/2
    ymax = (ymax1+ymax2)/2

    if classify == True:
        class1, conf1 = box1[0], box1[1]
        class2, conf2 = box2[0], box2[1]

        class_id = max(class1, class2)
        conf = max(conf1, conf2)

        return [class_id, conf, xmin, ymin, xmax, ymax]
    
    conf1, conf2 = box1[-1], box2[-1]
    conf = max(conf1, conf2)
    return [xmin, ymin, xmax, ymax, conf]

def merge_boxes(boxes_li1, boxes_li2, classify=False):
    if classify == False:
        boxes_li1 = boxes_li1.tolist()
        boxes_li2 = boxes_li2.tolist()

    if len(boxes_li1) < len(boxes_li2):
        boxes1, boxes2 = boxes_li1, boxes_li2
    else:
        boxes1, boxes2 = boxes_li2, boxes_li1
    
    boxes = []
    
    for i in range(len(boxes1)):
        overlap_idx = None
        overlap_max = 0
        for j in range(len(boxes2)):
            if overlap_ratio(boxes1[i], boxes2[j], classify=classify) > 0.1 and \
                overlap_ratio(boxes1[i], boxes2[j], classify=classify) > overlap_max:
                overlap_idx = j
                overlap_max = overlap_ratio(boxes1[i], boxes2[j], classify=classify)
        if overlap_idx is not None:
            merged_box = merge_box(boxes1[i], boxes2[overlap_idx], classify=classify)
            boxes.append(merged_box)
            del boxes2[overlap_idx]
        else:
            boxes.append(boxes1[i])
    boxes.extend(boxes2)

    if classify == True:
        return boxes
    else:
        return np.array(boxes)

            



if __name__ == '__main__':
    # box1 = [-1, -1, 0, 0, 4, 5]
    # box2 = [0, -1, 2, 2, 6, 7]
    # print(intersact_area(box1, box2))
    
    print('Hello')
    boxes1 = [[0, 0.5, 70, 60, 90, 100], [1, 1, 89, 69, 90, 100], [1, 0.9, 200, 250, 250, 310], [1, 0.0, 400, 390, 420, 410]]
    boxes2 = [[-1, -1, 71, 63, 95, 101], [-1, -1, 35, 40, 45, 50]]
    print(merge_boxes(boxes1, boxes2, classify=True))