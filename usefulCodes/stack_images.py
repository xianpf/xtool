import matplotlib.pyplot as plt


def stack_imgs(img_mats):
    '''
    把任意的numpy / torch tensor 以括弧组织起来，能输出stack到一起的大图，大大方便了对比看图
    img_mat:[[im1, im2,],
                [None, im3,],
                [im4]]
        im: h*w*3
    '''
    h_num = len(img_mats)
    row_lens = [len(row) for row in img_mats]
    w_num = max(row_lens)
    shapes = np.array([[img_mats[i][j].shape[:2] if (
        j<len(img_mats[i]) and img_mats[i][j] is not None
        ) else [0,0] for j in range(w_num)] for i in range(h_num)])
    grille_size = 3 # dark-bright-dark
    Hs, Ws = shapes[:,:,0].max(1), shapes[:,:,1].max(0)
    canvas = np.zeros((Hs.sum()+grille_size*(h_num-1), Ws.sum()+grille_size*(w_num-1), 3), dtype=np.uint8)
    for k in range(1, len(Hs)):
        canvas[Hs[:k].sum()+grille_size*k-grille_size//2-1] = 255
    for k in range(1, len(Ws)):
        canvas[:,Ws[:k].sum()+grille_size*k-grille_size//2-1] = 255
    for i in range(h_num):
        for j in range(w_num):
            if j>=len(img_mats[i]) or img_mats[i][j] is None:
                continue
            img = img_mats[i][j]
            if isinstance(img, torch.Tensor): img = img.cpu().detach().numpy()
            if isinstance(img, np.ndarray):
                H_bias = Hs[:i].sum()+grille_size*i
                W_bias = Ws[:j].sum()+grille_size*j
                H_im, W_im = img.shape[:2]
                if len(img.shape) == 3:
                    canvas[H_bias:H_bias+H_im,W_bias:W_bias+W_im] = img[:,:,:3]
                elif len(img.shape) == 2:
                    import pdb; pdb.set_trace()
                    norm = plt.Normalize(vmin=img.min(), vmax=img.max())
                    # map the normalized data to colors
                    # image is now RGBA (512x512x4)
                    img3d = plt.cm.jet(norm(img))
                    canvas[H_bias:H_bias+H_im,W_bias:W_bias+W_im] = img3d[:,:,:3]
                else:
                    import pdb; pdb.set_trace()
            else:
                print('Not numpy array!')
                import pdb; pdb.set_trace()

    return canvas
  

stk_1 = stack_imgs([[None, img1, img2,],
                    [img3],
                    [None, img4, img5, img6]
])

