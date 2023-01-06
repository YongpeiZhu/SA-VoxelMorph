"""
losses for VoxelMorph
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import pdb

def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.range(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)


def poisson(rhs):
    h=1
    tol=1e-8
    print(tol)
    itermax=2e3
    omega=0.618
    [n,m,l]=rhs.shape
    w=np.zeros([n,m,l])
    wo=np.zeros([n,m,l])
    error=1
    iter=1
    rate=h**2
    while (iter<itermax) and (error>tol):
        for k in range(1,l-1):
            for i in range(1,n-1):
                for j in range(1,m-1):
                    wt=0.16666667*(w[i-1,j,k]+wo[i+1,j,k]+w[i,j-1,k]+wo[i,j+1,k]+w[i,j,k-1]+wo[i,j,k+1]-rate*rhs[i,j,k])
                    w[i,j,k]=(1-omega)*wo[i,j,k]+omega*wt
        error_norm_2=np.sum((w-wo)**2)
        if error>tol:
            error=error_norm_2
        elif error<tol:
            break
        iter=iter+1
    return w



def dst(g,N):
    [m,n]=g.get_shape().as_list()
    # print(g.get_shape(),'****')
    C=tf.Variable(tf.zeros([m,n]), dtype=tf.float32)
    b=np.zeros([m,n],dtype=np.float32)
    for i in range(m):
        for j in range(n):
            b[i,j]=np.sin(np.pi*(i+1)*(j+1)/(N+1))

    tf.assign(C,tf.convert_to_tensor(b))
    y=tf.matmul(C,g)

    return y

def idst(g,N):
    [m,n]=g.get_shape().as_list()
    print(g.get_shape(), 'aaa')
    # C=np.zeros([m,n])
    C1 = tf.Variable(tf.zeros([m, n]), dtype=tf.float32)
    # for i in range(m):
    #     for j in range(n):
    #         tf.assign(C[i,j],(2/(N+1))*tf.sin(np.pi*(i+1)*(j+1)/(N+1)))
    # y=tf.matmul(C,g)
    b = np.zeros([m, n], dtype=np.float32)
    for i in range(m):
        for j in range(n):
            b[i, j] = (2/(N+1))*np.sin(np.pi*(i+1)*(j+1)/(N+1))

    tf.assign(C1, tf.convert_to_tensor(b))
    y = tf.matmul(C1, g)

    return y



def poisfft3D(F):
    [n1,n2,n3]=F.get_shape().as_list()
    assert n1==n2,n1==n3
    h=1
    G=h**2*F
    print(G,'dddd')

    U=tf.Variable(tf.zeros([n1,n1,n1]),dtype=tf.float32)
    for k in range(n1):
        Uwk=dst(tf.transpose(G[:,:,k],[1,0]),n1)
        Uwk=tf.transpose(Uwk,[1,0])
        tf.assign(U[:,:,k],dst(Uwk,n1))

    for i in range(n1):
        Uwk=tf.Variable(tf.zeros([n1,n1]),dtype=tf.float32)
        for j in range(n1):
            tf.assign(Uwk[:,j],U[i,j,:])
        Uwk=dst(Uwk,n1)
        tf.assign(U[i,:,:],tf.transpose(Uwk,[1,0]))
    # print(U,'!!!!!')
    theta = (np.pi/(2*(n1+1)))*np.arange(1,n1+1,dtype=np.float32).reshape(1,n1)
    print(theta.shape,'ffff')
    tmp = 4*(np.sin(theta.T))**2
    tmp1=np.zeros([n1,n1,n1],dtype=np.float32)
    Uwk1 = np.zeros([n1, n1, n1], dtype=np.float32)
    for i in range(n1):
        tmp1[:,:,i]=tmp[i]*np.ones([n1,n1],dtype=np.float32)
    print(tmp1.shape,'fff1')
    Uwk = np.dot(tmp,np.ones([1,n1],dtype=np.float32)) + np.dot(np.ones([n1, 1],dtype=np.float32),tmp.T)
    print(Uwk.shape, 'fff2')
    for k in range(n1):
        Uwk1[:,:,k]=Uwk+tmp1[:,:,k]
    G = tf.convert_to_tensor(Uwk1)
    print(U, 'ssssss111')
    print(G,'ssssss')

    # Uwk= tf.convert_to_tensor(Uwk)
    # T=tf.Variable(tf.zeros_like(U),dtype=tf.float32)
    U=tf.realdiv(U,G)
    # Uwk = tf.Variable(tf.zeros([n1, n1]), dtype=tf.float32)
    print(tf.transpose(U[:,:,1],[1,0]),'hhhh')
    # for i in range(n1):
    #     for j in range(n1):
    #         for k in range(n1):
    #             tf.assign(T[i,j,k],U[i,j,k]/(G[i,j,k]))


    print('fff1')
    # for k in range(n1):
    #     Uwk = idst(tf.transpose(U[:,:,k],[1,0]),n1)
    #     Uwk = tf.transpose(Uwk,[1,0])
    #     tf.assign(U[:,:,k],idst(Uwk,n1))
    # for i in range(n1):
    #     Uwk=tf.Variable(tf.zeros([n1,n1]),dtype=tf.float32)
    #     for j in range(n1):
    #         tf.assign(Uwk[:, j],U[i, j,:])
    #     Uwk =idst(Uwk, n1)
    #     tf.assign(U[i, :, :],tf.transpose(Uwk,[1,0]))


    return U

# a=np.ones([4,4,4])
# print(a.shape,'ggggg')
# w=poisson(a)
# print(w,'kkkkkk')
#
# w=poisfft3D(a)
# print(w,'fffff')
# a=np.ones([3,3])
# print(dst(a,3),'dddd')
# b=np.ones([3,3])
# print(idst(b,3),'fffff')
# x = np.linspace(0, 1, 10+1)
# y = np.linspace(0, 1, 10+1)
# xx,yy = np.meshgrid(x, y)
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # fig=plt.figure(1)
# # ax=fig.add_subplot(2,2,1,projection='3d')
# # ax.plot_surface(xx,yy)
# plt.plot(xx,yy)
# plt.plot(yy,xx)
# plt.show()
def matrixpad3D(A,x):
    [m, n, l] = A.shape
    B = np.ones([m + 2, n + 2, l + 2]) * x
    B[1: m+1 , 1: n+1, 1: l+1]=A
    return B

def gradient(y):
    print(y, '888')
    vol_shape = y.get_shape().as_list()[1:-1]
    ndims = len(vol_shape)

    df = [None] * ndims
    print(df)
    # dfi=tf.Variable(tf.zeros([1,224,224,224,3]),dtype=tf.float32)
    # dfi=dfi[np.newaxis,:]
    dfi=tf.zeros_like(y)
    print(dfi.shape,'1111')
    for i in range(ndims):
        d = i + 1
        # permute dimensions to put the ith dimension first
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        y1 = K.permute_dimensions(y, r)
        # dfi1 = np.transpose(dfi, r)
        # print(y1.shape,'dddd')
        dfi = y1[1:, ...] - y1[:-1, ...]
        # tf.assign(dfi[:-1, ...],y1[1:, ...] - y1[:-1, ...])
        # dfii = dfi[1:, ...] - dfi[:-1, ...]
        print(dfi.shape, '2222')

        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        # r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        r = [*range(1,d+1), 0, *range(d + 1, ndims + 2)]
        # df[i] = np.transpose(dfi1, r)
        df[i] = K.permute_dimensions(dfi, r)
        # df[i]=dfi

        print(df, '11111')
    # sess=tf.Session()
    # [m,n,l]=a.get_shape().as_list()
    # print(a.get_shape(),'ssss')
    # print(m,'kkk')
    #
    # # a=a.eval(session=sess)
    # print(a.shape,'hhhh')
    # b1x = np.zeros([m,n,l])
    # b1y = np.zeros([m,n,l])
    # b1z = np.zeros([m,n,l])
    #
    #
    #
    # b1z[:,:,0] = a[:,:,1] - a[:,:,0]
    # b1z[:,:,-1] = a[:,:,-1] - a[:,:,-2]
    # for i in range(a.shape[2] - 2):
    #     b1z[:,:,i + 1] = (a[:,:,i + 2] - a[:,:,i]) / 2
    #
    # print(b1z, 'qqqqqqq')
    # # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)
    #
    #
    # b1x[0, :,:] = a[1, :,:] - a[0, :,:]
    # b1x[-1, :,:] = a[-1, :,:] - a[-2, :,:]
    # for i in range(a.shape[0] - 2):
    #     b1x[i + 1, :,:] = (a[i + 2, :,:] - a[i, :,:]) / 2
    #
    # print(b1x, 'aaaaa')
    #
    #
    #
    # b1y[:, 0,:] = a[:, 1,:] - a[:, 0,:]
    # b1y[:, -1,:] = a[:, -1,:] - a[:, -2,:]
    # for i in range(a.shape[1] - 2):
    #     b1y[:, i + 1,:] = (a[:, i + 2,:] - a[:, i,:]) / 2
    #
    # print(b1y, 'ssssss')
    return df[0], df[1], df[2]
def interp3d(pos_x,pos_y,pos_z,R):
    [m,n,l]=R.shape
    Ri=np.zeros_like(R)
    ceil_x=0
    ceil_y=0
    ceil_z=0
    floor_x=0
    floor_y=0
    floor_z=0
    for i in range(m):
        for j in range(n):
            for k in range(l):
                x=pos_x[i,j,k]
                y=pos_y[i,j,k]
                z=pos_z[i,j,k]
                if (x>=m-1):
                    x=m-1
                    ceil_x=m-1
                    floor_x=ceil_x-1
                else:
                    if (x<=0):
                        x=0
                    floor_x=int(np.floor(x))
                    ceil_x=floor_x+1
                if (y>=n-1):
                    y=n-1
                    ceil_y=n-1
                    floor_y=ceil_y-1
                else:
                    if (y<=0):
                        y=0
                    floor_y=int(np.floor(y))
                    ceil_y=floor_y+1
                if (z>=l-1):
                    z=l-1
                    ceil_z=l-1
                    floor_z=ceil_z-1
                else:
                    if (z<=0):
                        z=0
                    floor_z=int(np.floor(z))
                    ceil_z=floor_z+1
                Ri[i, j, k] = R[ceil_x, ceil_y, ceil_z] * (x - floor_x) * (y - floor_y) * (z - floor_z) + R[floor_x, floor_y, floor_z] * (ceil_x - x) * (ceil_y - y) * (ceil_z - z) \
                +R[floor_x, ceil_y, ceil_z]* (ceil_x - x) * (y - floor_y) * (z - floor_z) + R[floor_x, ceil_y, floor_z] * (ceil_x - x) * (y - floor_y) * (ceil_z - z) \
                +R[floor_x, floor_y, ceil_z] * (ceil_x - x) * (ceil_y - y) * (z - floor_z) + R[ceil_x, ceil_y,floor_z] * (x - floor_x) * (y - floor_y) * (ceil_z - z) \
                +R[ceil_x, floor_y, ceil_z] * (x - floor_x) * (ceil_y - y) * (z - floor_z) + R[ceil_x, floor_y,floor_z] * (x - floor_x) * (ceil_y - y) * (ceil_z - z)
    return Ri




def img_Reg3D(T_temp,R,N):
    m=N
    n=N
    l=N
    tstep=1e-4
    # tstep_ratio=1e-16
    # tstep_up=1.2
    # tstep_down = 0.8
    # ite_max = 2.5e3
    # ratiotol = 1e-5
    # ratio = 1
    # F=tf.Variable(tf.zeros([162-2,162-2,162-2,3]),dtype=tf.float32)
    N=64
    grid_size = [N, N, N]
    f1 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    f2 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    f3 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    TR_diff = T_temp - R
    [T_x1, T_x2, T_x3] = gradient(T_temp)

    # lap(a) = [R(phi(X)) - T(X)] * grad(R(phi(X)))
    lap_a1 = (TR_diff[:,0:239,:,:,:]* T_x1)
    lap_a2 = (TR_diff[:,:,0:239,:,:]* T_x2)
    lap_a3 = (TR_diff[:,:,:,0:239,:]* T_x3)
    # lap_a1 = (TR_diff* T_x1)
    # lap_a2 = (TR_diff* T_x2)
    # lap_a3 = (TR_diff* T_x3)
    print(lap_a1.shape,'8888')

    lap_a1_interior = lap_a1[0,1:N+1, 1: N+1, 1:N+1,0]
    lap_a2_interior = lap_a2[0,1:N+1, 1: N+1, 1: N+1,0]
    lap_a3_interior = lap_a3[0,1:N+1, 1: N+1, 1: N+1,0]
    print((lap_a1_interior.shape), '99999')


    w1 = poisfft3D(lap_a1_interior)
    w2 = poisfft3D(lap_a2_interior)
    w3 = poisfft3D(lap_a3_interior)
    f1_new = f1 - (w1) * tstep
    print(f1_new.shape, '4444')
    f2_new = f2 - (w2) * tstep
    f3_new = f3 - (w3) * tstep
    F = [None] * 3
    lap = [None] * 3
    f1_new=tf.expand_dims(f1_new,axis=3)
    f2_new = tf.expand_dims(f2_new, axis=3)
    f3_new = tf.expand_dims(f3_new, axis=3)

    F[0]=tf.expand_dims(tf.concat([f1_new,f1_new,f1_new],axis=3),axis=0)
    F[1]=tf.expand_dims(tf.concat([f2_new,f2_new,f2_new],axis=3),axis=0)
    F[2]=tf.expand_dims(tf.concat([f3_new,f3_new,f3_new],axis=3),axis=0)
    print(F,'1111')
    print('nnn')

    lap[0] = lap_a1[0, 1:N + 1, 1: N + 1, 1:N + 1, :]
    lap[1] = lap_a2[0, 1:N + 1, 1: N + 1, 1:N + 1, :]
    lap[2] = lap_a3[0, 1:N + 1, 1: N + 1, 1:N + 1, :]

    # #
    #     intU1 = poisfft3D(f1_new)
    #     intU2 = poisfft3D(f2_new)
    #     intU3 = poisfft3D(f3_new)
    #
    #     U1 = matrixpad3D(intU1, 0)
    #     U2 = matrixpad3D(intU2, 0)
    #     U3 = matrixpad3D(intU3, 0)
    #
    #     pos_x = xI + U1
    #     pos_y = yI + U2
    #     pos_z = zI + U3
    #
    #     T_temp = interp3d(pos_x, pos_y, pos_z, T)
    #
    #
    #     ssd = np.sum((T_temp - R)**2)
    #     ratio = ssd / ssd_initial
    #     if ssd > ssd_old:
    #         tstep = tstep * tstep_down
    #         Better = 0
    #     else:
    #         tstep = tstep * tstep_up
    #         ssd_old = ssd
    #         Better = 1
    #         f1 = f1_new
    #         f2 = f2_new
    #         f3 = f3_new
    # print([' tstep:', tstep, ' ssd:', ssd, ' ratio:', ratio, ' ii:', ii])
    return F[0],F[1],F[2],lap[0],lap[1],lap[2]

def img_Reg3D1(T_temp,R,N):
    m=N
    n=N
    l=N
    tstep=1e-4
    # tstep_ratio=1e-16
    # tstep_up=1.2
    # tstep_down = 0.8
    # ite_max = 2.5e3
    # ratiotol = 1e-5
    # ratio = 1
    # F=tf.Variable(tf.zeros([162-2,162-2,162-2,3]),dtype=tf.float32)
    N=64
    grid_size = [N, N, N]
    f1 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    f2 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    f3 = tf.Variable(tf.zeros(grid_size),dtype=tf.float32)
    TR_diff = T_temp - R
    [T_x1, T_x2, T_x3] = gradient(T_temp)

    lap = [None] * 3
    # lap(a) = [R(phi(X)) - T(X)] * grad(R(phi(X)))
    lap_a1 = (TR_diff[:,0:239,:,:,:]* T_x1)
    lap_a2 = (TR_diff[:,:,0:239,:,:]* T_x2)
    lap_a3 = (TR_diff[:,:,:,0:239,:]* T_x3)
    # lap_a1 = (TR_diff* T_x1)
    # lap_a2 = (TR_diff* T_x2)
    # lap_a3 = (TR_diff* T_x3)
    # print(lap_a1.shape,'8888')

    lap[0] = lap_a1[:,1:N+1, 1: N+1, 1:N+1,:]
    lap[1] = lap_a2[:,1:N+1, 1: N+1, 1:N+1,:]
    lap[2] = lap_a3[:,1:N+1, 1: N+1, 1:N+1,:]
    # print((lap_a1_interior.shape), '99999')
    # F = [None] * 3
    # f1_new = tf.expand_dims(f1_new, axis=3)
    # f2_new = tf.expand_dims(f2_new, axis=3)
    # f3_new = tf.expand_dims(f3_new, axis=3)



    return lap[0],lap[1],lap[2]

def JD(data):
    data1 = data[0,:, :, :, 0]
    data2 = data[0,:, :, :, 1]
    data3 = data[0,:, :, :, 2]
    a = data1
    # a=np.random.randn(N,N)*10
    [m,n,l]=a.get_shape().as_list()
    b1x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b1y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b1z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    # b1x=np.zeros([N,N],dtype=np.float)
    # b1y=np.zeros([N,N],dtype=np.float)
    # b2x=np.zeros([N,N],dtype=np.float)
    # b2y=np.zeros([N,N],dtype=np.float)

    tf.assign(b1z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b1z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b1z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b1z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b1x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b1x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b1x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b1x, 'aaaaa')

    tf.assign(b1y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b1y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b1y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b1y, 'aaaaa')
    # b1x = np.squeeze(b1x)
    # b1y = np.squeeze(b1y)
    # b1z = np.squeeze(b1z)
    # b1x = np.pad(b1x, [(0, 224 - len_) for len_ in b1x.shape], "constant")
    # b1y= np.pad(b1y, [(0, 224 - len_) for len_ in b1y.shape], "constant")
    # b1z = np.pad(b1z, [(0, 224 - len_) for len_ in b1z.shape], "constant")
    a = data2
    # print(a)

    tf.assign(b2z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b2z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b2z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b2z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b2x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b2x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b2x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b2x, 'aaaaa')

    tf.assign(b2y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b2y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b2y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b2y, 'aaaaa')
    # b2x = np.pad(b2x, [(0, 224 - len_) for len_ in b2x.shape], "constant")
    # b2y= np.pad(b2y, [(0, 224 - len_) for len_ in b2y.shape], "constant")
    # b2z = np.pad(b2z, [(0, 224 - len_) for len_ in b2z.shape], "constant")
    a = data3
    # print(a)

    tf.assign(b3z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b3z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b3z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b3z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b3x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b3x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b3x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b3x, 'aaaaa')

    tf.assign(b3y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b3y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b3y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b3y, 'aaaaa')
    # b3x = np.pad(b3x, [(0, 224 - len_) for len_ in b3x.shape], "constant")
    # b3y= np.pad(b3y, [(0, 224 - len_) for len_ in b3y.shape], "constant")
    # b3z = np.pad(b3z, [(0, 224 - len_) for len_ in b3z.shape], "constant")
    # print(np.dot(b1x,b2y),'jjjj')
    # print(np.dot(b1y,b2x),'dddddd')
    JDx = b2y*b3z- b3y*b2z
    JDy = b1y*b3z- b3y*b1z
    JDz = b1y*b2z- b2y*b1z

    JD1 = b1x*JDx-b1x*b1x-0.01*tf.ones([m,n,l],dtype=tf.float32)
    JD2 =-b2x*JDy-b2y*b2y-0.01*tf.ones([m,n,l],dtype=tf.float32)
    JD3 = b3x*JDz-b3z*b3z-0.01*tf.ones([m,n,l],dtype=tf.float32)
    JD1 = tf.expand_dims(JD1, axis=3)
    JD2 = tf.expand_dims(JD2, axis=3)
    JD3 = tf.expand_dims(JD3, axis=3)
    jd=[None] * 3
    jd[0] = tf.expand_dims(tf.concat([JD1, JD1, JD1], axis=3), axis=0)
    jd[1] = tf.expand_dims(tf.concat([JD2, JD2, JD2], axis=3), axis=0)
    jd[2] = tf.expand_dims(tf.concat([JD3, JD3, JD3], axis=3), axis=0)
    # print(F, '1111')
    print('nnn')
    return jd[0],jd[1],jd[2]

def JD1(data):
    data1 = data[0,:, :, :, 0]
    data2 = data[0,:, :, :, 1]
    data3 = data[0,:, :, :, 2]
    a = data1
    # a=np.random.randn(N,N)*10
    [m,n,l]=a.get_shape().as_list()
    b1x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b1y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b1z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b2z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3x = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3y = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    b3z = tf.Variable(tf.zeros_like(a,dtype=tf.float32))
    # b1x=np.zeros([N,N],dtype=np.float)
    # b1y=np.zeros([N,N],dtype=np.float)
    # b2x=np.zeros([N,N],dtype=np.float)
    # b2y=np.zeros([N,N],dtype=np.float)

    tf.assign(b1z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b1z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b1z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b1z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b1x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b1x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b1x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b1x, 'aaaaa')

    tf.assign(b1y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b1y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b1y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b1y, 'aaaaa')
    # b1x = np.squeeze(b1x)
    # b1y = np.squeeze(b1y)
    # b1z = np.squeeze(b1z)
    # b1x = np.pad(b1x, [(0, 224 - len_) for len_ in b1x.shape], "constant")
    # b1y= np.pad(b1y, [(0, 224 - len_) for len_ in b1y.shape], "constant")
    # b1z = np.pad(b1z, [(0, 224 - len_) for len_ in b1z.shape], "constant")
    a = data2
    # print(a)

    tf.assign(b2z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b2z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b2z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b2z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b2x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b2x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b2x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b2x, 'aaaaa')

    tf.assign(b2y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b2y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b2y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b2y, 'aaaaa')
    # b2x = np.pad(b2x, [(0, 224 - len_) for len_ in b2x.shape], "constant")
    # b2y= np.pad(b2y, [(0, 224 - len_) for len_ in b2y.shape], "constant")
    # b2z = np.pad(b2z, [(0, 224 - len_) for len_ in b2z.shape], "constant")
    a = data3
    # print(a)

    tf.assign(b3z[:, :, 0],a[:, :, 1] - a[:, :, 0])
    tf.assign(b3z[:, :, -1],a[:, :, -1] - a[:, :, -2])
    for i in range(a.shape[2] - 2):
        tf.assign(b3z[:, :, i + 1],(a[:, :, i + 2] - a[:, :, i]) / 2)

    # print(b3z, 'qqqqqqq')
    # a=np.array([[1,2,4,5],[3,4,6,7],[1,3,4,5],[5,6,8,9]],dtype=np.float)

    tf.assign(b3x[0, :, :],a[1, :, :] - a[0, :, :])
    tf.assign(b3x[-1, :, :],a[-1, :, :] - a[-2, :, :])
    for i in range(a.shape[0] - 2):
        tf.assign(b3x[i + 1, :, :],(a[i + 2, :, :] - a[i, :, :]) / 2)

    # print(b3x, 'aaaaa')

    tf.assign(b3y[:, 0, :],a[:, 1, :] - a[:, 0, :])
    tf.assign(b3y[:, -1, :],a[:, -1, :] - a[:, -2, :])
    for i in range(a.shape[1] - 2):
        tf.assign(b3y[:, i + 1, :],(a[:, i + 2, :] - a[:, i, :]) / 2)

    # print(b3y, 'aaaaa')
    # b3x = np.pad(b3x, [(0, 224 - len_) for len_ in b3x.shape], "constant")
    # b3y= np.pad(b3y, [(0, 224 - len_) for len_ in b3y.shape], "constant")
    # b3z = np.pad(b3z, [(0, 224 - len_) for len_ in b3z.shape], "constant")
    # print(np.dot(b1x,b2y),'jjjj')
    # print(np.dot(b1y,b2x),'dddddd')
    JDx = b2y*b3z- b3y*b2z
    JDy = b1y*b3z- b3y*b1z
    JDz = b1y*b2z- b2y*b1z

    JD1 = b1x * JDx
    JD2 = -b2x * JDy
    JD3 = b3x * JDz
    JD1 = tf.expand_dims(JD1, axis=3)
    JD2 = tf.expand_dims(JD2, axis=3)
    JD3 = tf.expand_dims(JD3, axis=3)
    # jd = [None] * 3
    jd = tf.expand_dims(tf.concat([JD1, JD2, JD3], axis=3), axis=0)

    # JD = b1x*JDx-b2x*JDy+b3x*JDz
    # JD = tf.expand_dims(JD, axis=3)
    #
    # # jd=[None] * 3
    # jd= tf.expand_dims(tf.concat([JD, JD, JD], axis=3), axis=0)
    # jd[1] = tf.expand_dims(tf.concat([JD2, JD2, JD2], axis=3), axis=0)
    # jd[2] = tf.expand_dims(tf.concat([JD3, JD3, JD3], axis=3), axis=0)
    # print(F, '1111')
    print('nnn')
    return jd


class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        # y=JD1(y)

        df = [None] * ndims
        print(df)
        dd = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y1 = K.permute_dimensions(y, r)
            dfi = y1[1:, ...] - y1[:-1, ...]
            dfii=dfi[1:, ...] - dfi[:-1, ...]
            # dfiii = dfii[1:, ...] - dfii[:-1, ...]
            # print(type(dfii.shape),'2222')
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # r = [*range(1,d+1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfii, r)
            # dd[i] = K.permute_dimensions(dfiii, r)

            print(df,'11111')
        # dff=df[1]+df[2]+df[3],dd[0],dd[1],dd[2]
        # print(dff,'33333')
            # pdb.set_trace()
        
        return df[0],df[1],df[2]

    def _diffs1(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        print(df)
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y1 = K.permute_dimensions(y, r)
            dfi = y1[1:, ...] - y1[:-1, ...]
            dfii = dfi[1:, ...] - dfi[:-1, ...]
            dfiii = dfii[1:, ...] - dfii[:-1, ...]
            # print(type(dfii.shape),'2222')

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # r = [*range(1,d+1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfiii, r)

            print(df, '11111')
        # dff=df[1]+df[2]+df[3]
        # print(dff,'33333')
        # pdb.set_trace()

        return df[0], df[1], df[2]
    # def ncc(self, I, J):
    #     # get dimension of volume
    #     # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    #     ndims = len(I.get_shape().as_list()) - 2
    #     assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    #
    #     # set window size
    #     if self.win is None:
    #         self.win = [9] * ndims
    #
    #     # get convolution function
    #     conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
    #
    #     # compute CC squares
    #     I2 = I*I
    #     J2 = J*J
    #     IJ = I*J
    #
    #     # compute filters
    #     sum_filt = tf.ones([*self.win, 1, 1])
    #     strides = 1
    #     if ndims > 1:
    #         strides = [1] * (ndims + 2)
    #     padding = 'SAME'
    #
    #     # compute local sums via convolution
    #     I_sum = conv_fn(I, sum_filt, strides, padding)
    #     J_sum = conv_fn(J, sum_filt, strides, padding)
    #     I2_sum = conv_fn(I2, sum_filt, strides, padding)
    #     J2_sum = conv_fn(J2, sum_filt, strides, padding)
    #     IJ_sum = conv_fn(IJ, sum_filt, strides, padding)
    #
    #     # compute cross correlation
    #     win_size = np.prod(self.win)
    #     u_I = I_sum/win_size
    #     u_J = J_sum/win_size
    #
    #     cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    #     I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    #     J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    #
    #     cc = cross*cross / (I_var*J_var + self.eps)
    #
    #     # return negative cc.
    #     return tf.reduce_mean(cc)
    def F_loss(self, y_true, y_pred):
        """ reconstruction loss """

        vol_shape = y_pred.get_shape().as_list()[1:-1]
        print(y_pred.get_shape(),'llll')
        mv=np.max(np.array([vol_shape[0],vol_shape[1],vol_shape[2]]))
        print(mv,'kkkk')
        N=64
        # y_pred=np.array(y_pred)
        print(y_pred.get_shape(),'ddd')
        # y_true=np.array(y_true)
        # y_pred=np.squeeze(y_pred)
        # y_true=np.squeeze(y_true)-self.F_loss(y_true, y_pred)self._diffs(y_pred)
        y_pred = tf.pad(y_pred, [[0,0],[0,0],[0,0],[96,96],[0,0]])
        print(y_pred.get_shape(),'eeee')
        y_true = tf.pad(y_true, [[0,0],[0,0],[0,0],[96,96],[0,0]])
        df = [None] * 3
        f0,f1,f2,lap0, lap1, lap2=img_Reg3D(y_pred,y_true,mv)
        d0,d1,d2=self._diffs(y_pred)
        # print(d0.get_shape(),'fffff11111')
        # jd0,jd1,jd2=JD(y_pred)
        # lap0, lap1, lap2 = img_Reg3D1(y_pred, y_true, mv)
        # d0, d1, d2 = self._diffs1(y_pred)
        # d0=-tf.concat([tf.expand_dims(tf.matrix_inverse(d0[0,0:N,0:N,0:N,0]),axis=3),tf.expand_dims(tf.matrix_inverse(d0[0,0:N,0:N,0:N,1]),axis=3),tf.expand_dims(tf.matrix_inverse(d0[0,0:N,0:N,0:N,2]),axis=3)],axis=3)
        # d1 = -tf.concat([tf.expand_dims(tf.matrix_inverse(d1[0, 0:N, 0:N, 0:N, 0]),axis=3), tf.expand_dims(tf.matrix_inverse(d1[0, 0:N, 0:N, 0:N, 1]),axis=3),
        #                 tf.expand_dims(tf.matrix_inverse(d1[0, 0:N, 0:N, 0:N, 2]),axis=3)], axis=3)
        # d2 = -tf.concat([tf.expand_dims(tf.matrix_inverse(d2[0, 0:N, 0:N, 0:N, 0]),axis=3), tf.expand_dims(tf.matrix_inverse(d2[0, 0:N, 0:N, 0:N, 1]),axis=3),
        #                 tf.expand_dims(tf.matrix_inverse(d2[0, 0:N, 0:N, 0:N, 2]),axis=3)], axis=3)
        # df0=lap0*(d0[0,0:N,0:N,0:N,:])
        # df1=lap1*(d1[0,0:N,0:N,0:N,:])
        # df2=lap2*(d2[0,0:N,0:N,0:N,:])
        # df[0]=df0*(g0[0,0:N,0:N,0:N,:]-f0)
        # df[1]=df1*(g1[0,0:N,0:N,0:N,:]-f1)
        # df[2]=df2*(g2[0,0:N,0:N,0:N,:]-f2)
        # df[0]=df0*(jd0[0,0:N,0:N,0:N,:])
        # df[1]=df1*(jd1[0,0:N,0:N,0:N,:])
        # df[2]=df2*(jd2[0,0:N,0:N,0:N,:])
        df[0] = tf.sin(d0[0, 0:N, 0:N, 0:N, :]-f0)*(d0[0, 0:N, 0:N, 0:N, :]-f0)
        df[1] = tf.sin(d1[0, 0:N, 0:N, 0:N, :]-f1)*(d1[0, 0:N, 0:N, 0:N, :]-f1)
        df[2] = tf.sin(d2[0, 0:N, 0:N, 0:N, :]-f2)*(d2[0, 0:N, 0:N, 0:N, :]-f2)
        print('222333')
        # df[0]=tf.sin(g0[0,0:N,0:N,0:N,:]-f0)*(jd0-f0*f0)
        # df[1]=tf.sin(g1[0,0:N,0:N,0:N,:]-f1)*(jd1-f1*f1)
        # df[2]=tf.sin(g2[0,0:N,0:N,0:N,:]-f2)*(jd2-f2*f2)
        # df[0]=tf.sin(jd0)*(jd0)
        # df[1]=tf.sin(jd1)*(jd1)
        # df[2]=tf.sin(jd2)*(jd2)

        return df

    def loss(self, y_true, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f*f) for f in self._diffs(y_pred)]

        return tf.add_n(df) / len(df)


class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        print(vol_shape,'fffff')
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        print(y_pred.get_shape(),'sssss')
        ndims = len(y_pred.get_shape())-2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well


    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))
