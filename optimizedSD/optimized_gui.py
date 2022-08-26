#coding:utf-8

import argparse, os, sys, glob, random
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config

import copy

import tkinter as tk
import tkinter.ttk as ttk
import configparser

from itertools import product


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def load_img(path, h0, w0):
   
    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")   
    if(h0 is not None and w0 is not None):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



class SDtxt2img:
    
    def __init__(self):#, dir_img_init):
        self.config = "optimizedSD/v1-inference.yaml"
        self.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
        self.device = "cuda"

        # self.dir_img_init = dir_img_init
        
        self.sd = load_model_from_config(f"{self.ckpt}")
        li = []
        lo = []
        for key, value in self.sd.items():
            sp = key.split('.')
            if(sp[0]) == 'model':
                if('input_blocks' in sp):
                    li.append(key)
                elif('middle_block' in sp):
                    li.append(key)
                elif('time_embed' in sp):
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            self.sd['model1.' + key[6:]] = self.sd.pop(key)
        for key in lo:
            self.sd['model2.' + key[6:]] = self.sd.pop(key)

        self.config = OmegaConf.load(f"{self.config}")

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        
        self.GUI()
        
    def GUI(self):
        self.root = tk.Tk()
        self.root.title("Stable Diffusion GUI")
        self.root.geometry("1300x900")
        
        
        # prompt
        self.label_prompt = tk.Label(self.root, text='prompt(the prompt to rende)')
        self.label_prompt.grid(column = 0, row = 0)

        self.box_prompt = tk.Entry(self.root, width=100)
        self.box_prompt.insert(tk.END,"a painting of a virus monster playing guitar")
        self.box_prompt.grid(column = 0, row = 1)
        # jumon
        self.label_jumon = tk.Label(self.root, text='jumon')
        self.label_jumon.grid(column = 0, row = 2)

        self.box_jumon = tk.Entry(self.root, width=100)
        self.box_jumon.insert(tk.END,"canon 5D, ultra detailed")
        self.box_jumon.grid(column = 0, row = 3)
        
        # mode プルダウン
        self.label_mode = tk.Label(self.root, text='mode')
        self.label_mode.grid(column = 0, row = 4)
        self.l_mode = ["txt2img","img2img"]
        self.var_mode = tk.StringVar ( )
        self.combo_mode = ttk.Combobox ( self.root , values = self.l_mode , textvariable = self.var_mode )
        self.combo_mode.current(0)
        self.combo_mode.grid(column = 0, row = 5)
        
        #dir_out
        self.label_dir_output = tk.Label(self.root, text='outdir(dir to write results to)')
        self.label_dir_output.grid(column = 0, row = 6)

        self.box_dir_output = tk.Entry(self.root, width=100)
        # self.box_dir_output.insert(tk.END,"outputs/img2img-samples")
        self.box_dir_output.grid(column = 0, row = 7)

        #dir_img
        self.label_dir_img = tk.Label(self.root, text='init-img(path to the input image)')
        self.label_dir_img.grid(column = 0, row = 8)

        self.box_dir_img = tk.Entry(self.root, width=100)
        self.box_dir_img.grid(column = 0, row = 9)

        # i_seed
        self.label_i_seed = tk.Label(self.root, text='the seed (for reproducible sampling)')
        self.label_i_seed.grid(column = 0, row = 10)
        self.scale_i_seed = tk.Scale(
            master = self.root,
            from_ = 1,
            to=501,
            length=600,
            tickinterval=50,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_i_seed.grid(column = 0, row = 11)

        # i_img_H i_img_W
        self.label_img_H = tk.Label(self.root, text='H')
        self.label_img_H.grid(column = 0, row = 12)
        self.box_img_H = tk.Entry(self.root, width=100)
        self.box_img_H.insert(tk.END,"512")
        self.box_img_H.grid(column = 0, row = 13)
        self.label_img_W = tk.Label(self.root, text='W')
        self.label_img_W.grid(column = 0, row = 14)
        self.box_img_W = tk.Entry(self.root, width=100)
        self.box_img_W.insert(tk.END,"512")
        self.box_img_W.grid(column = 0, row = 15)



        # b_small_batch.ラジオボタン作成
        self.label_small_batch = tk.Label(self.root, text='small_batch(Reduce inference time when generate a smaller batch of images)')
        self.label_small_batch.grid(column = 0, row = 16)
        self.var_small_batch = tk.IntVar()
        self.var_small_batch.set(0)
        self.rdo_small_batch_1 = tk.Radiobutton(self.root, value=1, variable=self.var_small_batch, text='True')
        self.rdo_small_batch_1.grid(column = 0, row = 17)
        self.rdo_small_batch_0 = tk.Radiobutton(self.root, value=0, variable=self.var_small_batch, text='False')
        self.rdo_small_batch_0.grid(column = 0, row = 18)

        # precision プルダウン
        self.label_precision = tk.Label(self.root, text='precision(evaluate at this precision)')
        self.label_precision.grid(column = 0, row = 19)
        self.l_precision = ["autocast","full"]
        self.var_precision = tk.StringVar ( )
        self.combo_precision = ttk.Combobox ( self.root , values = self.l_precision , textvariable = self.var_precision )
        self.combo_precision.current(0)
        self.combo_precision.grid(column = 0, row = 20)

        #n_samples
        self.label_n_samples = tk.Label(self.root, text='n_samples(how many samples to produce for each given prompt. A.k.a. batch size)')
        self.label_n_samples.grid(column = 0, row = 21)
        self.scale_n_samples = tk.Scale(
            master = self.root,
            from_ = 1,
            to=11,
            length=600,
            tickinterval=1,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_n_samples.set(1)
        self.scale_n_samples.grid(column = 0, row = 22)

        #n_rows
        self.label_n_rows = tk.Label(self.root, text='rows in the grid (default: n_samples)')
        self.label_n_rows.grid(column = 1, row = 0)
        self.scale_n_rows = tk.Scale(
            master = self.root,
            from_ = 0,
            to=10,
            length=600,
            tickinterval=1,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_n_rows.set(0)
        self.scale_n_rows.grid(column = 1, row = 1)

        # from_file
        self.label_from_file = tk.Label(self.root, text='from_file(if specified, load prompts from this file)')
        self.label_from_file.grid(column = 1, row = 2)
        self.box_from_file = tk.Entry(self.root, width=100)
        self.box_from_file.grid(column = 1, row = 3)

        #strength
        self.label_strength = tk.Label(self.root, text='strength(strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image)')
        self.label_strength.grid(column = 1, row = 4)
        self.scale_strength = tk.Scale(
            master = self.root,
            from_ = 0,
            to=1,
            length=600,
            tickinterval=0.1,
            resolution =   0.01,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_strength.set(0.75)
        self.scale_strength.grid(column = 1, row = 5)



        #ddim_steps
        self.label_ddim_steps = tk.Label(self.root, text='ddim_steps(number of ddim sampling steps, Please set in multiples of 5)')
        self.label_ddim_steps.grid(column = 1, row = 6)
        self.scale_ddim_steps = tk.Scale(
            master = self.root,
            from_ = 1,
            to=301,
            length=600,
            tickinterval=50,
            orient=tk.HORIZONTAL,
            resolution =   1,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_ddim_steps.set(50)
        self.scale_ddim_steps.grid(column = 1, row = 7)

        #n_iter
        self.label_n_iter = tk.Label(self.root, text='n_iter(sample this often)')
        self.label_n_iter.grid(column = 1, row = 8)
        self.scale_n_iter = tk.Scale(
            master = self.root,
            from_ = 1,
            to=11,
            length=600,
            tickinterval=2,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_n_iter.set(1)
        self.scale_n_iter.grid(column = 1, row = 9)

        # scale
        self.label_scale = tk.Label(self.root, text='scale(unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)))')
        self.label_scale.grid(column = 1, row = 10)
        self.scale_scale = tk.Scale(
            master = self.root,
            from_ = 0.01,
            to=20,
            length=600,
            tickinterval=2,
            resolution =   0.01,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_scale.set(7.5)
        self.scale_scale.grid(column = 1, row = 11)
        
        
        
        # fixed_code ラジオボタン作成
        self.label_fixed_code = tk.Label(self.root, text='fixed_code(if enabled, uses the same starting code across samples)')
        self.label_fixed_code.grid(column = 1, row = 12)
        self.var_fixed_code = tk.IntVar()
        self.var_fixed_code.set(0)
        self.rdo_fixed_code_1 = tk.Radiobutton(self.root, value=1, variable=self.var_small_batch, text='True')
        self.rdo_fixed_code_1.grid(column = 1, row = 13)
        self.rdo_fixed_code_0 = tk.Radiobutton(self.root, value=0, variable=self.var_small_batch, text='False')
        self.rdo_fixed_code_0.grid(column = 1, row = 14)
        
        
        
        #C
        self.label_C = tk.Label(self.root, text='C(latent channels)')
        self.label_C.grid(column = 1, row = 15)
        self.scale_C = tk.Scale(
            master = self.root,
            from_ = 0,
            to=10,
            length=600,
            tickinterval=2,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_C.set(4)
        self.scale_C.grid(column = 1, row = 16)
        
        #f
        self.label_f = tk.Label(self.root, text='f(downsampling factor)')
        self.label_f.grid(column = 1, row = 17)
        self.scale_f = tk.Scale(
            master = self.root,
            from_ = 0,
            to=10,
            length=600,
            tickinterval=2,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_f.set(8)
        self.scale_f.grid(column = 1, row = 18)
        
        
        # ddim_eta
        self.label_ddim_eta = tk.Label(self.root, text='ddim eta (eta=0.0 corresponds to deterministic sampling')
        self.label_ddim_eta.grid(column = 1, row = 19)
        self.scale_ddim_eta = tk.Scale(
            master = self.root,
            from_ = 0.0,
            to=1,
            length=600,
            tickinterval=0.1,
            resolution =   0.1,
            orient=tk.HORIZONTAL,
            # command=lambda val : var.set(scale.get()) 
        )
        self.scale_ddim_eta.set(0.0)
        self.scale_ddim_eta.grid(column = 1, row = 20)
        
        

        self.button = tk.Button(text="生成")
        self.button.grid(column = 1, row = 21)
        self.button.bind("<Button-1>",self.click) 

        self.root.mainloop()
        
        
        
    def click(self, event):
        print(
        self.box_dir_output.get(),"\n",
        self.box_prompt.get(),"\n",
        self.scale_i_seed.get(),"\n",
        self.var_small_batch.get(),"\n",
        self.box_dir_img.get(),"\n",
        self.box_img_H.get(),"\n",
        self.box_img_W.get(),"\n",
        self.var_precision.get(),"\n",
        self.scale_n_samples.get(),"\n",
        self.scale_n_rows.get(),"\n",
        self.box_from_file.get(),"\n",
        self.scale_strength.get(),"\n",
        self.scale_ddim_steps.get(),"\n",
        self.scale_n_iter.get(),"\n",
        self.scale_scale.get(),"\n",
        self.var_mode.get(),"\n",
        self.var_fixed_code.get(),"\n",
        self.scale_C.get(),"\n",
        self.scale_f.get(),"\n",
        self.scale_ddim_eta.get(),"\n",
        
        )
        # print(
        # type(box_dir_output.get()),"\n",
        # type(box_prompt.get()),"\n",
        # type(scale_i_seed.get()),"\n",
        # type(var_small_batch.get()),"\n",
        # type(box_dir_img.get()),"\n",
        # type(box_img_H.get()),"\n",
        # type(box_img_W.get()),"\n",
        # type(var_precision.get()),"\n",
        # type(scale_n_samples.get()),"\n",
        # type(scale_n_rows.get()),"\n",
        # type(box_from_file.get()),"\n",
        # type(scale_n_rows.get()),"\n",
        # type(scale_strength.get()),"\n",
        # type(scale_ddim_steps.get()),"\n",
        # type(scale_n_iter.get()),"\n",
        # type(scale_scale.get()),"\n",
        # )
        self.run(
            self.box_dir_output.get(),
            self.box_prompt.get(),
            self.box_jumon.get(),
            self.scale_i_seed.get(),
            self.var_small_batch.get(),
            self.box_dir_img.get(),
            int(self.box_img_H.get()),
            int(self.box_img_W.get()),
            self.var_precision.get(),
            self.scale_n_samples.get(),
            self.scale_n_rows.get(),
            self.box_from_file.get(),
            self.scale_strength.get(),
            self.scale_ddim_steps.get(),
            self.scale_n_iter.get(),
            self.scale_scale.get(),
            self.var_mode.get(),
            self.var_fixed_code.get(),
            self.scale_C.get(),
            self.scale_f.get(),
            self.scale_ddim_eta.get(),
            )
        # print(
        
        
    def run(self,dir_out,s_prompt,s_jumon,i_seed,b_small_batch,dir_img,i_img_H,i_img_W,s_precision,i_n_samples,i_n_rows,s_from_file,f_strength,i_ddim_steps,i_n_iter,f_scale, s_mode,b_fixed_code,i_C,i_f,f_ddim_eta):

        # 保存先の設定  
        tic = time.time()
        
        if dir_out=="":
            if s_mode == "txt2img":
                dir_out="outputs/txt2img-samples"
            else:
                dir_out="outputs/img2img-samples"
        
        os.makedirs(dir_out, exist_ok=True)
        outpath = dir_out
        
        if s_jumon=="":
            s_prompt_gen = s_prompt
        else:
            s_prompt_gen = s_prompt+","+s_jumon
        
        sample_path = os.path.join(outpath, "_".join(s_prompt_gen.split()))[:150]
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        #シード値の設定
        if i_seed == None:
            i_seed = randint(0, 1000000)
        print("init_seed = ", i_seed)
        seed_everything(i_seed)#再現性のあるシード値の設定

        #モデル/設定ロードは上部に移動

        # 設定切り替え
        if b_small_batch:
            self.config.modelUNet.params.small_batch = True
        else:
            self.config.modelUNet.params.small_batch = False

            
        if s_mode == "img2img":
            assert os.path.isfile(dir_img)
            init_image = load_img(dir_img, i_img_H, i_img_W).to(self.device)

        model = instantiate_from_config(self.config.modelUNet)
        _, _ = model.load_state_dict(self.sd, strict=False)
        model.eval()
            
        modelCS = instantiate_from_config(self.config.modelCondStage)
        _, _ = modelCS.load_state_dict(self.sd, strict=False)
        modelCS.eval()
            
        modelFS = instantiate_from_config(self.config.modelFirstStage)
        _, _ = modelFS.load_state_dict(self.sd, strict=False)
        modelFS.eval()
        # del self.sd
        if s_precision == "autocast":
            model.half()
            modelCS.half()
            if s_mode == "img2img":
                modelFS.half()
                init_image = init_image.half()
                
        if s_mode == "txt2img":
            start_code = None
            if b_fixed_code:
                start_code = torch.randn([i_n_samples, i_C, i_img_H // i_f, i_img_W // i_f], device=device)


        batch_size = i_n_samples
        n_rows = i_n_rows if i_n_rows > 0 else batch_size
        if not s_from_file:
            prompt = s_prompt_gen
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {s_from_file}")
            with open(s_from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        if s_mode == "img2img":
            modelFS.to(self.device)

            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

            mem = torch.cuda.memory_allocated()/1e6
            modelFS.to("cpu")
            while(torch.cuda.memory_allocated()/1e6 >= mem):
                time.sleep(1)


            assert 0. <= f_strength <= 1., 'can only work with strength in [0.0, 1.0]'
            t_enc = int(f_strength * i_ddim_steps)
            print(f"target t_enc is {t_enc} steps")

        print("===========================")




        precision_scope = autocast if s_precision=="autocast" else nullcontext
        with torch.no_grad():

            all_samples = list()
            
            
            
            iter = product(range(i_n_iter), data)
            # for n, prompts in tqdm(iter):
            
            
            
            # for n in trange(i_n_iter, desc="Sampling"):
                # for prompts in tqdm(data, desc="data"):
            for n, prompts in tqdm(iter):
                with precision_scope("cuda"):
                    modelCS.to(self.device)
                    uc = None
                    if f_scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    
                    c = modelCS.get_learned_conditioning(prompts)
                    if s_mode == "txt2img":
                        shape = [i_C, i_img_H // i_f, i_img_W // i_f]
                    mem = torch.cuda.memory_allocated()/1e6
                    modelCS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)
                        
                        
                    if s_mode == "img2img":
                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device), i_seed,i_ddim_steps)
                        # decode it
                        samples_ddim = model.decode(z_enc, c, t_enc, unconditional_guidance_scale=f_scale,
                                                    unconditional_conditioning=uc,)

                    else:
                        samples_ddim = model.sample(S=i_ddim_steps,
                                        conditioning=c,
                                        batch_size=i_n_samples,
                                        seed = i_seed,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=f_scale,
                                        unconditional_conditioning=uc,
                                        eta=f_ddim_eta,
                                        x_T=start_code)



                    modelFS.to(self.device)
                    print("saving images")
                    for i in range(batch_size):
                        
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(i_seed) + "_" + f"{base_count:05}.png"))
                        i_seed+=1
                        base_count += 1


                    mem = torch.cuda.memory_allocated()/1e6
                    modelFS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)

                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated()/1e6)

        toc = time.time()

        time_taken = (toc-tic)/60.0

        print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))
    
if __name__ == '__main__':

    

    """
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )



    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--small_batch",
        action='store_true',
        help="Reduce inference time when generate a smaller batch of images",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()


    # parser initializing
    dir_out = opt.outdir
    s_prompt = opt.prompt
    i_seed = opt.seed
    b_small_batch = opt.small_batch
    dir_img = opt.init_img
    i_img_H = opt.H
    i_img_W = opt.W
    s_precision = opt.precision
    i_n_samples = opt.n_samples
    i_n_rows = opt.n_rows
    s_from_file = opt.from_file
    f_strength = opt.strength
    i_ddim_steps = opt.ddim_steps
    i_n_iter = opt.n_iter
    f_scale = opt.scale
    
    print(dir_img, type(dir_img))
    """
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
        # "--init-img",
        # type=str,
        # nargs="?",
        # help="path to the input image"
    # )
    # opt = parser.parse_args()
    
    sdti = SDtxt2img()
    # sdti.run(dir_out,s_prompt,i_seed,b_small_batch,dir_img,i_img_H,i_img_W,s_precision,i_n_samples,i_n_rows,s_from_file,f_strength,i_ddim_steps,i_n_iter,f_scale )
