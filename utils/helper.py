import torch
from shapely.geometry import Polygon

step_size = 1


def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0, :, :] == ego_map[1, :, :]) * (ego_map[1, :, :] == ego_map[2, :, :]) + (
            ego_map[0, :, :] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))


def convert_map_to_road_map(ego_map):
    mask = (ego_map[0, :, :] == 1) * (ego_map[1, :, :] == 1) * (ego_map[2, :, :] == 1)

    return (~mask)


def collate_fn(batch):
    concated_input = []
    for i in range(len(batch)):
        concated_input.append(torch.reshape(batch[i][0], (1, 18, 128, 160)))
    concated_input = torch.cat(concated_input, 0)
    bbox_list = []
    category_list = []
    for i in range(len(batch)):
        min_x = torch.min(batch[i][1]['bounding_box'][:, 0, :], dim=1)[0]
        min_y = torch.min(batch[i][1]['bounding_box'][:, 1, :], dim=1)[0]
        max_x = torch.max(batch[i][1]['bounding_box'][:, 0, :], dim=1)[0]
        max_y = torch.max(batch[i][1]['bounding_box'][:, 1, :], dim=1)[0]
        width = torch.sub(max_x, min_x)
        height = torch.sub(max_y, min_y)
        bbox_list.append(torch.floor(torch.stack(
            [min_x * 10 + 400 + width * 10 / 2, min_y * 10 + 400 + height * 10 / 2, width * 10, height * 10], dim=1)))
        category_list.append(batch[i][1]['category'])
    concated_roadmap = []
    for i in range(len(batch)):
        concated_roadmap.append(batch[i][2].unsqueeze(0))
    concated_roadmap = torch.cat(concated_roadmap, 0).long()
    return [concated_input, bbox_list, category_list, concated_roadmap]


def collate_fn_lstm(batch):
    concated_input = []
    for i in range(len(batch)):
        concated_input.append(batch[i][0].view(step_size, 18, 128, 160))
    concated_input = torch.stack(concated_input)
    batch_bbox_list = []
    batch_category_list = []
    for i in range(len(batch)):
        for j in range(len(batch[i][1])):
            min_x = torch.min(batch[i][1][j]['bounding_box'][:, 0, :], dim=1)[0]
            min_y = torch.min(batch[i][1][j]['bounding_box'][:, 1, :], dim=1)[0]
            max_x = torch.max(batch[i][1][j]['bounding_box'][:, 0, :], dim=1)[0]
            max_y = torch.max(batch[i][1][j]['bounding_box'][:, 1, :], dim=1)[0]
            width = torch.sub(max_x, min_x)
            height = torch.sub(max_y, min_y)
            batch_bbox_list.append(torch.floor(torch.stack(
                [min_x * 10 + 400 + width * 10 / 2, min_y * 10 + 400 + height * 10 / 2, width * 10, height * 10],
                dim=1)))
            batch_category_list.append(batch[i][1][j]['category'])
    concated_roadmap = []
    for i in range(len(batch)):
        concated_roadmap.append(batch[i][2])
    concated_roadmap = torch.cat(concated_roadmap, dim=0).long()
    return [concated_input, batch_bbox_list, batch_category_list, concated_roadmap]


def collate_fn_lstm_self(batch):
    concated_input = []
    concated_loss_mask = []
    for i in range(len(batch)):
        concated_input.append(batch[i][0].view(step_size, 18, 128, 160))
        concated_loss_mask.append(batch[i][4])
    concated_input = torch.stack(concated_input)
    concated_loss_mask = torch.stack(concated_loss_mask)
    batch_bbox_list = []
    batch_category_list = []
    for i in range(len(batch)):
        for j in range(len(batch[i][1])):
            min_x = torch.min(batch[i][1][j]['bounding_box'][:, 0, :], dim=1)[0]
            min_y = torch.min(batch[i][1][j]['bounding_box'][:, 1, :], dim=1)[0]
            max_x = torch.max(batch[i][1][j]['bounding_box'][:, 0, :], dim=1)[0]
            max_y = torch.max(batch[i][1][j]['bounding_box'][:, 1, :], dim=1)[0]
            width = torch.sub(max_x, min_x)
            height = torch.sub(max_y, min_y)
            batch_bbox_list.append(torch.floor(torch.stack(
                [min_x * 10 + 400 + width * 10 / 2, min_y * 10 + 400 + height * 10 / 2, width * 10, height * 10],
                dim=1)))
            batch_category_list.append(batch[i][1][j]['category'])
    concated_roadmap = []
    for i in range(len(batch)):
        concated_roadmap.append(batch[i][2])
    concated_roadmap = torch.cat(concated_roadmap, dim=0).long()
    return [concated_input, batch_bbox_list, batch_category_list, concated_roadmap, concated_loss_mask]


def collate_fn_display(batch):
    return tuple(zip(*batch))


def collate_fn_unlabeled(batch):
    concated_input = []
    for i in range(len(batch)):
        concated_input.append(torch.reshape(batch[i], (1, 18, 128, 160)))
    concated_input = torch.cat(concated_input, 0)
    return [concated_input, concated_input]


def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])

    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)


def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)


def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull

    return a.intersection(b).area / a.union(b).area
