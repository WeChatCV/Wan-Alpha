import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x



class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)



class VideoSobelEdgeLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.image_sobel = GradLoss()
        
    def forward(self, input, target):
       
        if input.dim() == 5 and target.dim() == 5:
            batch_size, channels, time_steps, height, width = input.size()
            input = input.permute(0, 2, 1, 3, 4)  # C T H W -> T C H W
            target = target.permute(0, 2, 1, 3, 4)  # C T H W -> T C H W
            
            input_reshaped = input.reshape(batch_size * time_steps, channels, height, width)
            target_reshaped = target.reshape(batch_size * time_steps, channels, height, width)
            
           
            loss = self.image_sobel(input_reshaped, target_reshaped)
            
            return loss
        else:
           
            return self.image_sobel(input, target)



def Fourier_filter(x, threshold, scale):
    ori_dtype = x.dtype
    B, C, T, H, W = x.shape
    
    # FFT
    x = x.to(dtype=torch.float32)
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    
    mask = torch.ones((B, C, T, H, W)).cuda() 

    ctime, crow, ccol = T // 2, H // 2, W //2
    mask[..., ctime - threshold:ctime + threshold, crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-3, -2, -1)).real
    return x_filtered.to(dtype=ori_dtype)
