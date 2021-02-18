import imageio
import torch

def torch2uint8(images, permute_order=[0,2,3,1]):
    r"""Convert batch of torch samples into a NumPy image array.
    Arguments:
        images (tensor): raw image samples.
        permute_order: adjusts the permute order of tensor.permute()
    Returns:
        numpy array of images. 
    """
    return (images.permute(*permute_order)*127.5 + 128).clamp(0,255).to(torch.uint8).cpu().numpy()

def batched_synthesis_generator(W, G, batch_size):
    r"""Generator that yields batches of images from W vectors.
    Arguments:
        W (tensor): W vectors to be generated (BS, 16, 512)
        G (model): The model used to generate images.
        batch_size (int): The size of each yielded image array
    """
    w_batches = torch.split(W, batch_size)
    for batch in w_batches:
        images = G.synthesis(batch, noise_mode='const', force_fp32=True)
        images = torch2uint8(images)
        yield images

def batched_generator(Z, G, batch_size, truncation_psi):
    r"""Generator that yields batches of images from Z vectors.
    Arguments:
        Z (tensor): Z vectors to be generated (BS, 512).
        G (model): The model used to generate images.
        batch_size (int): The size of each yielded image array.
        truncation_psi (float): The truncation to be applied in G.
    """
    z_batches = torch.split(Z, batch_size)
    for batch in z_batches:
        images = G(batch, None, truncation_psi=truncation_psi)
        images = torch2uint8(images)
        yield images

def generate_video(image_generator, filename, fps):
    r"""Generate mp4 from a collection of images.
    Arguments:
        image_generator (generator): The generator object to get images from. 
                                     Generators will be more memory efficient, 
                                     but tuples and lists can be used aswell. 
        filename (str): The location the mp4 will be saved to.
        fps (int): The FPS of the video. 
    """
    writer = imageio.get_writer(filename, format='FFMPEG', mode='I', fps=fps)
    try:
        for images in image_generator:
            map(writer.append_data, images)
    except Exception as e:
        print(e)
    finally:
        writer.close()

def slerp(v0, v1, t, DOT_THRESHHOLD=0.9995):
    r"""Spherical interpolation between two tensors
    Arguments:
        v0 (tensor): The first point to be interpolated from. 
        v1 (tensor): The second point to be interpolated from.
        t (float): The ratio between the two points.
        DOT_THRESHHOLD (float): How close should the dot product be to a
                                straight line before deciding to use a linear
                                 interpolation instead.
    Returns:
        Tensor of a single step from the interpolated path between v0 to v1
        at ratio t.  
    """
    v0_copy = torch.clone(v0)
    v1_copy = torch.clone(v1)

    v0 = v0 / torch.norm(v0)
    v1 = v1 / torch.norm(v1)

    dot = torch.sum(v0 * v1)

    if torch.abs(dot) > DOT_THRESHHOLD:
        return torch.lerp(t, v0_copy, c1_copy)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2
