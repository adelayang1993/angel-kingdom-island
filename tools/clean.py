import cv2, numpy as np, sys
cfg={
 'ch1':dict(text=[(60,10,345,95),(1195,28,1535,78),(75,228,600,582),(1565,340,1625,590)],
            solid=[('pill',100,612,425,692),('circle',1594,371,26)]),
 'ch2':dict(text=[(50,12,272,88),(900,28,1215,72),(70,160,470,340),(90,340,380,482),(1160,340,1215,535),(330,780,880,898)],
            solid=[('pill',88,505,340,568),('circle',1190,413,22)]),
 'ch3':dict(text=[(45,8,340,88),(1130,22,1520,64),(75,235,480,642),(1580,325,1632,565),
                  (335,490,478,640),(535,490,688,640),(770,470,955,600),(1035,488,1192,640),(1255,488,1408,640)],
            solid=[('pill',83,688,482,752),('circle',1605,451,27)]),
 'ch4':dict(text=[(65,15,345,92),(1140,25,1525,70),(75,185,440,415),(75,435,395,577),(1580,340,1636,668),(420,645,1480,812)],
            solid=[('pill',78,610,398,680),('circle',1607,574,26)]),
 'ch5':dict(text=[(65,12,360,92),(1240,28,1615,74),(80,165,455,397),(80,425,428,597),(1585,265,1630,565)],
            solid=[('pill',72,622,418,692),('circle',1605,528,27)]),
}
def build(n):
    im=cv2.imread(f'src/{n}.png'); h,w=im.shape[:2]; m=np.zeros((h,w),np.uint8); c=cfg[n]
    med=cv2.medianBlur(im,21)
    d=np.abs(im.astype(int)-med.astype(int)).max(2)
    for x0,y0,x1,y1 in c['text']:
        sub=(d[y0:y1,x0:x1]>26).astype(np.uint8)*255
        m[y0:y1,x0:x1]|=sub
    m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    for s in c['solid']:
        if s[0]=='pill':
            _,x0,y0,x1,y1=s; r=(y1-y0)//2
            cv2.rectangle(m,(x0+r,y0),(x1-r,y1),255,-1); cv2.circle(m,(x0+r,y0+r),r,255,-1); cv2.circle(m,(x1-r,y0+r),r,255,-1)
        else:
            _,cx,cy,r=s; cv2.circle(m,(cx,cy),r,255,-1)
    m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    return im,m
if __name__=='__main__':
    for n in sys.argv[1:]:
        im,m=build(n); cv2.imwrite(f'work/{n}_mask.png',m)
        out=cv2.inpaint(im,m,9,cv2.INPAINT_TELEA)
        cv2.imwrite(f'clean/{n}.png',out)
