import os
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision
from utils.helper import convert_map_to_lane_map, convert_map_to_road_map
import matplotlib.pyplot as plt
NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]


# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform
        print("create unlabeled dataset")
        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                #image = torchvision.transforms.ToPILImage()(self.transform(image))
                #plt.imshow(image)
                #plt.show()
                images.append(self.transform(image))
            image_tensor = torch.stack(images)

            return image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE


# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, roadmap_transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
        self.roadmap_transform = roadmap_transform

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = self.roadmap_transform(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra

        else:
            return image_tensor, target, road_image


class LabeledDatasetScene(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, roadmap_transform, extra_info=True,
                 scene_batch_size=4):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
        self.roadmap_transform = roadmap_transform
        self.scene_batch_size = scene_batch_size

    def __len__(self):
        id_cand = NUM_SAMPLE_PER_SCENE
        return self.scene_index.size * math.floor(id_cand/self.scene_batch_size)

    def __getitem__(self, index):
        index_cand = math.floor(NUM_SAMPLE_PER_SCENE/self.scene_batch_size)
        scene_id = self.scene_index[index // index_cand]
        # sample_id = index % (NUM_SAMPLE_PER_SCENE // self.scene_batch_size)
        sample_id_list = np.arange(self.scene_batch_size*(index % index_cand),self.scene_batch_size*((index % index_cand) +1))
        
        sample_path_list = []        
        for sample_id in sample_id_list:
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')
            sample_path_list.append(sample_path)
        samples = []
        for sample_path in sample_path_list:
            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)
            samples.append(image_tensor)
        samples = torch.stack(samples)

        corners_list = []
        categories_list = []
        for sample_id in sample_id_list:
            data_entries = self.annotation_dataframe[
                (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
            corners_list.append(
                data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy())
            categories_list.append(data_entries.category_id.to_numpy())
        road_image_list = []
        target_list = []
        for i in range(len(sample_path_list)):
            sample_path = sample_path_list[i]
            ego_path = os.path.join(sample_path, 'ego.png')
            ego_image = Image.open(ego_path)
            ego_image = self.roadmap_transform(ego_image)
            road_image = convert_map_to_road_map(ego_image)
            road_image_list.append(road_image)
            target = {}
            target['bounding_box'] = torch.as_tensor(corners_list[i]).view(-1, 2, 4)
            target['category'] = torch.as_tensor(categories_list[i])
            target_list.append(target)
        road_images = torch.stack(road_image_list)

        return samples, target_list, road_images
