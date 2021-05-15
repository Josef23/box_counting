import numpy as np
from skimage import io


def box_draw(image):
    image = (image / image.max()).astype(np.uint8)
    boxes = np.zeros((*image.shape, 3), dtype=np.uint8)
    for jumb in range(max(image.shape) - 1, 5, -1):
        y_coords, x_coords = np.meshgrid(
            range(0, image.shape[0], jumb), range(0, image.shape[1], jumb), indexing='ij'
        )
        coords = np.array([y_coords, x_coords]).reshape((2, -1)).T
        for point0, point1 in coords:
            if np.sum(image[point0:point0 + jumb, point1:point1 + jumb]):
                boxes[point0:point0 + 1, point1:point1 + jumb, 0] = 1
                boxes[point0:point0 + jumb, point1:point1 + 1, 0] = 1
                boxes[point0:point0 + jumb, point1 + jumb:point1 + jumb + 1, 0] = 1
                boxes[point0 + jumb:point0 + jumb + 1, point1:point1 + jumb, 0] = 1
            else:
                boxes[point0:point0 + 1, point1:point1 + jumb, 1] = 1
                boxes[point0:point0 + jumb, point1:point1 + 1, 1] = 1
                boxes[point0:point0 + jumb, point1 + jumb:point1 + jumb + 1, 1] = 1
                boxes[point0 + jumb:point0 + jumb + 1, point1:point1 + jumb, 1] = 1

        boxes[..., 1][boxes[..., 0] > 0] = 0
        image_copy = image.copy()
        image_copy[np.logical_or(boxes[..., 0], boxes[..., 1]) > 0] = 0
        boxes[..., 0] += image_copy
        boxes[..., 1] += image_copy
        boxes[..., 2] += image_copy

        io.imsave(f"./result/{jumb}.png", boxes * 255, check_contrast=False)
        boxes = np.zeros((*image.shape, 3), dtype=np.uint8)


def box_edge_draw(image):
    image = (image / image.max()).astype(np.uint8)
    boxes = np.zeros((*image.shape, 3), dtype=np.uint8)
    for jumb in range(max(image.shape) - 1, 5, -1):
        y_coords, x_coords = np.meshgrid(
            range(0, image.shape[0], jumb), range(0, image.shape[1], jumb), indexing='ij'
        )
        coords = np.array([y_coords, x_coords]).reshape((2, -1)).T
        pixel_max_count = jumb * jumb
        for point0, point1 in coords:
            pixel_count = np.sum(image[point0:point0 + jumb, point1:point1 + jumb])
            if pixel_count and pixel_count != pixel_max_count:
                boxes[point0:point0 + 1, point1:point1 + jumb, 0] = 1
                boxes[point0:point0 + jumb, point1:point1 + 1, 0] = 1
                boxes[point0:point0 + jumb, point1 + jumb:point1 + jumb + 1, 0] = 1
                boxes[point0 + jumb:point0 + jumb + 1, point1:point1 + jumb, 0] = 1
            else:
                boxes[point0:point0 + 1, point1:point1 + jumb, 1] = 1
                boxes[point0:point0 + jumb, point1:point1 + 1, 1] = 1
                boxes[point0:point0 + jumb, point1 + jumb:point1 + jumb + 1, 1] = 1
                boxes[point0 + jumb:point0 + jumb + 1, point1:point1 + jumb, 1] = 1

        boxes[..., 1][boxes[..., 0] > 0] = 0
        image_copy = image.copy()
        image_copy[np.logical_or(boxes[..., 0], boxes[..., 1]) > 0] = 0
        boxes[..., 0] += image_copy
        boxes[..., 1] += image_copy
        boxes[..., 2] += image_copy

        io.imsave(f"./result/{jumb}.png", boxes * 255, check_contrast=False)
        boxes = np.zeros((*image.shape, 3), dtype=np.uint8)


def box_ratio(image):
    ratios = {}
    for jumb in range(max(image.shape) - 1, 5, -1):
        y_coords, x_coords = np.meshgrid(
            range(0, image.shape[0], jumb), range(0, image.shape[1], jumb), indexing='ij'
        )
        coords = np.array([y_coords, x_coords]).reshape((2, -1)).T
        non_empty_boxes = 0
        for point0, point1 in coords:
            if np.sum(image[point0:point0 + jumb, point1:point1 + jumb]):
                non_empty_boxes += 1
        ratios[jumb] = (len(coords), non_empty_boxes)
        print(f"Total boxes: {len(coords)}, non empty boxes: {non_empty_boxes}, ratio {non_empty_boxes/len(coords)}")


def box_edge_ratio(image):
    ratios = {}
    for jumb in range(max(image.shape) - 1, 5, -1):
        y_coords, x_coords = np.meshgrid(
            range(0, image.shape[0], jumb), range(0, image.shape[1], jumb), indexing='ij'
        )
        coords = np.array([y_coords, x_coords]).reshape((2, -1)).T
        non_empty_boxes = 0
        pixel_max_count = jumb * jumb
        for point0, point1 in coords:
            pixel_count = np.sum(image[point0:point0 + jumb, point1:point1 + jumb])
            if pixel_count and pixel_count != pixel_max_count:
                non_empty_boxes += 1
        ratios[jumb] = (len(coords), non_empty_boxes)
        print(f"Total boxes: {len(coords)}, non empty boxes: {non_empty_boxes}, ratio {non_empty_boxes/len(coords)}")


if __name__ == "__main__":
    image = io.imread("./images/GB_binary.png")
    box_draw(image)
    box_ratio(image)
