# directory to store the results
results_dir = './results_new'

# root to the testsets
# dataroot = "./test/test3/lsun_bedroom"
dataroot = 'D:\zjl\pythonProject2\\fusing\\test\\gan_dnf'

# list of synthesis algorithms
#vals = ['adm','ddpm','iddpm','ldm','pndm','sdv2','vqdiffusion','dalle2','if','midjourney']
# vals = ['vqdiffusion','sdv2','if','midjourney','dalle2']
# vals = [ 'ADM',  'DALLE2','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong']
#vals = ['biggan','adm']
vals = ['biggan','cyclegan/apple','cyclegan/horse','cyclegan/orange','cyclegan/summer','cyclegan/winter','cyclegan/zebra','gaugan','stylegan/bedroom','stylegan/car','stylegan/cat','stylegan2/car','stylegan2/cat','stylegan2/church','stylegan2/horse','stargan']
#vals = ['FreeDoM','HPS','Midjourney','SDXL']
#vals=['glide','wukong']
#vals = ['adm','sdv1']
#vals = ['adm']
#vals = ['DreamBooth','FreeDoM','LoRA','SDXL_Refine']
#vals = ['dalle2','if','midjourney','sdv2']
# vals = ['adm','dalle2','ddpm','iddpm','if','ldm','midjourney','pndm','sdv2','vqdiffusion']
model_path = './weights/model_epoch_best.pth'
# model_path = './weights/model_epoch_0.pth'

