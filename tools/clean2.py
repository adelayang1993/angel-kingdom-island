import cv2, numpy as np, sys, time
sys.path.insert(0,'.'); from clean import cfg
def masks(n):
    im=cv2.imread(f'src/{n}.png'); h,w=im.shape[:2]; c=cfg[n]
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,k)          # bright thin strokes
    bh=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,k)        # dark thin strokes (text on plaques/pills)
    text=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in c['text']:
        t=((th[y0:y1,x0:x1]>38)|(bh[y0:y1,x0:x1]>60)).astype(np.uint8)*255
        text[y0:y1,x0:x1]|=t
    text=cv2.dilate(text,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
    solid=np.zeros((h,w),np.uint8)
    for s in c['solid']:
        if s[0]=='pill':
            _,x0,y0,x1,y1=s; r=(y1-y0)//2
            cv2.rectangle(solid,(x0+r,y0),(x1-r,y1),255,-1); cv2.circle(solid,(x0+r,y0+r),r,255,-1); cv2.circle(solid,(x1-r,y0+r),r,255,-1)
        elif s[0]=='circle':
            _,cx,cy,r=s; cv2.circle(solid,(cx,cy),r,255,-1)
        elif s[0]=='poly':
            cv2.fillPoly(solid,[np.array(s[1],np.int32)],255)
    solid=cv2.dilate(solid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    if n=='ch5':
        f=np.zeros_like(solid); cv2.fillPoly(f,[np.array(c['solid'][-1][1],np.int32)],255); solid|=cv2.dilate(f,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)))
    return im,text,solid
def run(n):
    t=time.time(); im,text,solid=masks(n)
    out=cv2.inpaint(im,text,5,cv2.INPAINT_TELEA)
    if solid.any():
        dst=out.copy(); cv2.xphoto.inpaint(out,255-solid,dst,cv2.xphoto.INPAINT_SHIFTMAP); out=dst
    cv2.imwrite(f'clean/{n}.png',out); cv2.imwrite(f'work/{n}_mask.png',text|solid)
    print(n,round(time.time()-t,1))
if __name__=='__main__':
    for n in sys.argv[1:]: run(n)
