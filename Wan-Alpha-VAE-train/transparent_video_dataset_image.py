import torch
import torchvision

import random
import imageio
import pandas as pd
from einops import rearrange
from PIL import Image
import numpy as np


import torch
import pandas as pd
import random
import imageio
from PIL import Image
import torchvision.transforms
from einops import rearrange

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, num_frames=81, height=480, width=832, static_frames=17):
        metadata = pd.read_csv(metadata_path)
        self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.static_frames = static_frames  

        self.frame_process = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def crop_and_resize_video(self, image_lists):
        width, height = image_lists[0][0].size
        tmp_width = random.randint(self.width, self.width*2)
        tmp_height = random.randint(self.height, self.height*2)
        scale = max(tmp_width / width, tmp_height / height)

        image_lists = [
                [torchvision.transforms.functional.resize(
                        img,
                        (round(height*scale), round(width*scale)),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                    ) for img in image_list] for image_list in image_lists
            ]
        
        width, height = image_lists[0][0].size
        top_idx = random.randint(0, height-self.height)
        left_idx = random.randint(0, width-self.width)

        image_lists = [
                        [torchvision.transforms.functional.crop(img,
                            top=top_idx, left=left_idx,
                            height=self.height, width=self.width)
                            for img in image_list] for image_list in image_lists
                    ]
        
        return image_lists
        
    def crop_and_resize_static_video(self, image_lists):
        """
        Process a single image to generate a frame sequence with the same effect as the original static video (17 frames copied).

        The `image_lists` format is: `[[img1], [img2]]`, where each sublist contains one image
        """
        
        # 1. Calculate scaling parameters
        base_width, base_height = image_lists[0][0].size  
        tmp_width = random.randint(self.width, self.width * 2)
        tmp_height = random.randint(self.height, self.height * 2)
        scale = max(tmp_width / base_width, tmp_height / base_height)  
        
        # 2. Calculate the scaled dimensions
        scaled_width = round(base_width * scale)
        scaled_height = round(base_height * scale)

        # 3. Divide into 2x2 regions (all images use the same region division)
        mid_x = scaled_width // 2
        mid_y = scaled_height // 2

        # 4. Target crop size
        target_width, target_height = self.width, self.height

        # 5. Randomly select starting coordinates (all images use the same starting point)
        start_x = random.randint(0, max(0, scaled_width - target_width))
        start_y = random.randint(0, max(0, scaled_height - target_height))

        # 6. Determine the region of the starting coordinates
        if start_x < mid_x and start_y < mid_y:
            region = 0  # Top-left region
        elif start_x >= mid_x and start_y < mid_y:
            region = 1  # Top-right region
        elif start_x < mid_x and start_y >= mid_y:
            region = 2  # Bottom-left region
        else:
            region = 3  # Bottom-right region

        # 7. Handle boundary cases
        if scaled_width < target_width or scaled_height < target_height:
            if region == 0:
                start_x, start_y = 0, 0
            elif region == 1:
                start_x, start_y = max(0, scaled_width - target_width), 0
            elif region == 2:
                start_x, start_y = 0, max(0, scaled_height - target_height)
            else:
                start_x, start_y = max(0, scaled_width - target_width), max(0, scaled_height - target_height)

            # Generate static frame sequences for all images (no sliding)
            all_results = []
            for img_group in image_lists:
                # Scale a single image
                scaled_img = torchvision.transforms.functional.resize(
                    img_group[0],  # Take a single image
                    (scaled_height, scaled_width),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )

                # Generate the specified number of frames (originally 17 frames)
                processed_group = []
                for _ in range(self.static_frames):
                    cropped_img = torchvision.transforms.functional.crop(
                        scaled_img, 
                        top=start_y, 
                        left=start_x,
                        height=target_height, 
                        width=target_width
                    )
                    processed_group.append(cropped_img)
                all_results.append(processed_group)
            return all_results

        # 8. Determine the sliding direction (all images use the same direction)
        if region == 0:
            possible_directions = [(1, 0), (0, 1), (1, 1)]
        elif region == 1:
            possible_directions = [(-1, 0), (0, 1), (-1, 1)]
        elif region == 2:
            possible_directions = [(1, 0), (0, -1), (1, -1)]
        else:
            possible_directions = [(-1, 0), (0, -1), (-1, -1)]
        dir_x, dir_y = random.choice(possible_directions)

        # 9. Calculate the maximum sliding distance and step size
        if dir_x > 0:
            max_total_dx = scaled_width - target_width - start_x
        elif dir_x < 0:
            max_total_dx = start_x
        else:
            max_total_dx = 0
        
        if dir_y > 0:
            max_total_dy = scaled_height - target_height - start_y
        elif dir_y < 0:
            max_total_dy = start_y
        else:
            max_total_dy = 0

        # Calculate the sliding step size for each frame (based on static frame count)
        step_x = (dir_x * max_total_dx) // self.static_frames if self.static_frames > 0 else 0
        step_y = (dir_y * max_total_dy) // self.static_frames if self.static_frames > 0 else 0

        # Process all images to generate sliding frame sequences
        all_results = []
        for img_group in image_lists:
            # Scale a single image
            scaled_img = torchvision.transforms.functional.resize(
                img_group[0],  # Take a single image
                (scaled_height, scaled_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )

            # Generate sliding cropped frame sequences
            processed_group = []
            current_x, current_y = start_x, start_y
            
            for _ in range(self.static_frames):
                # Ensure the cropping area is within valid bounds
                current_x = max(0, min(current_x, scaled_width - target_width))
                current_y = max(0, min(current_y, scaled_height - target_height))

                # Crop the current frame
                cropped_img = torchvision.transforms.functional.crop(
                    scaled_img,
                    top=current_y, 
                    left=current_x,
                    height=target_height, 
                    width=target_width
                )
                processed_group.append(cropped_img)

                # Update the position for the next frame
                current_x += step_x
                current_y += step_y
            
            all_results.append(processed_group)
        
        return all_results
    
    def load_video_pairs(self, file_path_1, file_path_2, video_flag):
        if video_flag == "static":
            
            img1 = Image.open(file_path_1).convert('RGB')  
            img2 = Image.open(file_path_2).convert('RGB')
            
           
            frames_1 = [img1]
            frames_2 = [img2]
        elif video_flag == "dynamic":
            reader_1, reader_2 = imageio.get_reader(file_path_1), imageio.get_reader(file_path_2)

            num_of_frames = reader_1.count_frames()
            assert num_of_frames == reader_2.count_frames()
            if num_of_frames < self.num_frames:
                # If the video is too short, play it in reverse.
                id_list = [i for i in range(0, num_of_frames)]
                id_list += id_list[::-1]
                id_list = id_list * self.num_frames

                # Cut a clip
                start_frame_id = random.randint(0, num_of_frames)
                id_list = id_list[start_frame_id:start_frame_id+self.num_frames]
            else:
                # Cut a clip
                start_frame_id = random.randint(0, num_of_frames - self.num_frames)
                id_list = range(start_frame_id, start_frame_id+self.num_frames)
        
            # Read videos based on frame number
            frames_1, frames_2 = [], []
            for frame_id in id_list:
                frames_1.append(Image.fromarray(reader_1.get_data(frame_id)))
                frames_2.append(Image.fromarray(reader_2.get_data(frame_id)))

            reader_1.close()
            reader_2.close()


        # Image
        if video_flag == "static":
            frames_1, frames_2 = self.crop_and_resize_static_video([frames_1, frames_2])
        else:
            frames_1, frames_2 = self.crop_and_resize_video([frames_1, frames_2])

        # To tensor
        def frame_to_tensor(frames):
            frames = [self.frame_process(f) for f in frames]
            frames = torch.stack(frames, dim=0)
            frames = rearrange(frames, "T C H W -> C T H W")
            return frames

        return {
            "video_fgr": frame_to_tensor(frames_1),
            "video_pha": frame_to_tensor(frames_2)
        }

    def __getitem__(self, data_id):
        data = self.data[data_id]

        while True:
            try:
                video_fgr = data["video_fgr"]  
                video_pha = data["video_pha"] 
                # video_flag = "static"  
                video_flag = data["status"]
                return self.load_video_pairs(video_fgr, video_pha, video_flag)
            
            except Exception as e:
                print(video_fgr, "Error", e)
                data = random.choice(self.data)

    def __len__(self):
        return len(self.data)

