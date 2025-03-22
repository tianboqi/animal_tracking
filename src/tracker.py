import torch
import numpy as np

class AnimalTracker:

    def __init__(self, device='cuda', batch_size=1000):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.kernel = torch.ones((1, 1, 5, 5), device=self.device)
        self.binary_masks_tail = None

    def process_frame(self, batch_frames):

        # Convert the frames to a tensor
        batch_tensor = torch.from_numpy(np.stack(batch_frames, axis=0)).to(self.device, dtype=torch.float32).mean(axis=3, keepdim=True).permute(0,3,1,2) / 255.0
        
        with torch.amp.autocast('cuda'):      # Mixed precision

            # Compute median frame (adaptive background)
            median_frame = torch.median(batch_tensor, dim=0).values
            std_frame = torch.std(batch_tensor, dim=0)

            # Background subtraction and thresholding
            binary_masks = torch.abs(batch_tensor - median_frame)
            binary_masks.masked_fill_(binary_masks <= (3 * std_frame), 0.0)
        
            # Apply morphological operations (erosion + dilation)
            binary_masks[:,:,:] = (torch.nn.functional.conv2d(binary_masks, self.kernel, padding=2) > 1).float()  # dilation
            binary_masks[:,:,:] = (torch.nn.functional.conv_transpose2d(binary_masks, self.kernel, padding=2) > 0.8).float()  # erosion
        
            if self.binary_masks_tail is None:
                self.binary_masks_tail = torch.cat([torch.zeros_like(binary_masks[0]), torch.zeros_like(binary_masks[0])]).unsqueeze(1)
            binary_masks = torch.cat((self.binary_masks_tail, binary_masks), dim=0)
    
            self.binary_masks_tail = binary_masks[-2:]
    
            # Median Filter across frames (Temporal Smoothing)
            binary_masks = torch.median(torch.cat([binary_masks[:-2].unsqueeze(0), 
                                                   binary_masks[1:-1].unsqueeze(0), 
                                                   binary_masks[2:].unsqueeze(0)]), dim=0).values  # Shape: (BATCH_SIZE, 1, H, W)

        # Calculate center of mass
        return self.compute_center_of_mass(binary_masks)

    def compute_center_of_mass(self, mask):

        batch_size, _, h, w = mask.shape

    	# Get coordinate indices
        y_coords = torch.arange(h, device=mask.device).view(1, 1, h, 1)  # Shape (1,1,h,1)
        x_coords = torch.arange(w, device=mask.device).view(1, 1, 1, w)  # Shape (1,1,1,w)

        # Compute the sum of mask values
        mask_sum = mask.sum(dim=(2, 3), keepdim=False)  # Shape: (batch, 1)

        # Compute the mean coordinates
        y_com = (mask * y_coords).sum(dim=(2, 3), keepdim=False) / mask_sum
        x_com = (mask * x_coords).sum(dim=(2, 3), keepdim=False) / mask_sum

        # Set NaN where mask_sum == 0 (no 1s in the mask)
        y_com[mask_sum.squeeze(1) == 0] = torch.nan
        x_com[mask_sum.squeeze(1) == 0] = torch.nan

        return torch.stack([y_com, x_com], dim=1).squeeze()  # Shape: (batch, 2)
