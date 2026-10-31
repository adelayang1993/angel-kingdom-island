import cv2, numpy as np, sys
sys.path.insert(0,'.'); from clean2 import masks
from clean import cfg
cfg['ch5']['solid'].append(('poly',[(1092,228),(1100,262),(1070,330),(1040,420),(1012,520),(1003,600),(998,650),(985,650),(970,560),(948,470),(945,380),(975,300),(1040,248)]))
def fill(im,m,sig):
    mf=(m>0).astype(np.float32); keep=1-mf
    num=cv2.GaussianBlur(im.astype(np.float32)*keep[...,None],(0,0),sig)
    den=cv2.GaussianBlur(keep,(0,0),sig)[...,None]+1e-4
    f=num/den
    # second pass for big holes
    num2=cv2.GaussianBlur(im.astype(np.float32)*keep[...,None],(0,0),sig*3); den2=cv2.GaussianBlur(keep,(0,0),sig*3)[...,None]+1e-4
    f=np.where(den<0.02,num2/den2,f)
    a=cv2.GaussianBlur(mf,(0,0),3)[...,None]; a=np.clip(a*1.6,0,1)
    out=im*(1-a)+f*a
    noise=np.random.normal(0,3,im.shape).astype(np.float32)*a
    return np.clip(out+noise,0,255).astype(np.uint8)
for n in sys.argv[1:]:
    im,text,solid=masks(n)
    out=fill(im,text|solid,14)
    cv2.imwrite(f'clean/{n}.png',out)
    cv2.imwrite(f'work/{n}_c.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,80])
