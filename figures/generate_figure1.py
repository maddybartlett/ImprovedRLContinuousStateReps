
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os

sys.path.insert(0,'../network/rlnet')
from sspspace import HexagonalSSPSpace, _get_sub_SSP, _proj_sub_SSP

sspspace = HexagonalSSPSpace(domain_dim=2,n_rotates=6,n_scales=6,scale_min=0.8, scale_max=3)
d = sspspace.ssp_dim
K = sspspace.phase_matrix
basis = sspspace.axis_matrix
sub_dim=3
N = (d-1)//(sub_dim*2)

# This is my old SSP code. TODO: switch to use the code in sspspace.py
def ssp_vectorized(basis, positions):
    positions = positions.reshape(-1,basis.shape[1])
    S_list = np.zeros((basis.shape[0],positions.shape[0]),dtype=complex)
    for i in np.arange(positions.shape[0]):
        S_list[:,i] = np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**positions[i,:], axis=1), axis=0)  
    return S_list.real

def similarity_plot(basis, xs, ys, x=0, y=0, S_list = None, S0 = None, check_mark= False, axis=None,cmap=None, **kwargs):
    # Heat plot of SSP similarity of x and y values of xs and ys
    # Input:
    #  X, Y - SSP basis vectors
    #  x, y - A single point to compare SSPs over the space with
    #  xs, ys - The x, y points to make the space tiling
    #  titleStr - (optional) Title of plot
    #  S_list - (optional) A list of the SSPs at all xs, ys tiled points (useful for high dim X,Y so that these do not 
    #           have to recomputed every time this function is called)
    #  S0 - (optional) The SSP representing the x, y point (useful if for some reason you want a similarity plot
    #       of tiled SSPs with a non-SSP vector or a SSP with a different basis)
    #  check_mark - (default True) Whether or not to put a black check mark at the x, y location
    xx,yy = np.meshgrid(xs,ys)
    positions = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    position0 = np.array([x,y])
    sim_dots, S_list = _similarity_values(basis,  positions, position0 = position0, S0 = S0, S_list = S_list)
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111)
    
    im=axis.pcolormesh(xx, yy, sim_dots.reshape(xx.shape).real, cmap=cmap,**kwargs)
    if check_mark:
        axis.plot(x,y, 'k+')
    return im,sim_dots, S_list 
    
def _similarity_values(basis, positions, position0 = None, S0 = None, S_list = None):
    if position0 is None:
        position0 = np.zeros(basis.shape[1])
    if S0 is None:
        S0 = ssp_vectorized(basis, position0)
    if S_list is None: 
        S_list = ssp_vectorized(basis, positions)
    sim_dots = S_list.T @ S0
    return(sim_dots, S_list)

cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

xs = np.linspace(-5,5,100)
ys = xs

i = 3*10
sub_mat = _get_sub_SSP(i,(d-1)//2,sublen=1)
proj_mat = _proj_sub_SSP(i,(d-1)//2,sublen=1)
basis_i = sub_mat @ basis

fig = plt.figure(figsize=(7.,2.))
gs = fig.add_gridspec(1,3,wspace=0.2,width_ratios=[3.5/4,3.5/4,1])
ax1 = plt.subplot(gs[:,0])
ax2 = plt.subplot(gs[:,1])
ax3 = plt.subplot(gs[:,2])

similarity_plot(basis_i,xs,ys,cmap=cmap,axis=ax1);
ax1.quiver(0, 0, K[i,0], K[i,1], angles='xy', scale_units='xy', scale=1,color='white')
ax1.axis(xmin=-5,ymin=-5,xmax=5,ymax=5)
ax1.text(K[i,0]+0.2, K[i,1]+0.2, '$\\theta_1$',color='white')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('$S(\mathbf{x}) = \mathcal{F}^{-1}\{ e^{i \\theta_1^T\mathbf{x}} \}$')

i = 10
sub_mat = _get_sub_SSP(i,N,sublen=3)
proj_mat = _proj_sub_SSP(i,N,sublen=3)
basis_i = sub_mat @ basis
similarity_plot(basis_i,xs,ys,cmap=cmap,axis=ax2);
ax2.quiver(np.zeros(3), np.zeros(3), K[3*i:3*(i+1),0], K[3*i:3*(i+1),1], angles='xy', scale_units='xy', scale=1,color='white')
ax2.axis(xmin=-5,ymin=-5,xmax=5,ymax=5)
for j in range(3):
    ax2.text(K[3*i + j,0]+0.2, K[3*i + j,1]+0.2, '$\\theta_' + str(j+1) + '$',color='white')
ax2.set_xlabel('$x$')
ax2.set_title('$S(\mathbf{x}) = \mathcal{F}^{-1} \{ e^{i   [ \\theta_1 \, \\theta_2 \, \\theta_3 ]^T \mathbf{x}}  \}$')

heatmap, _, _ = similarity_plot(basis,xs,ys,cmap=cmap,axis=ax3)
ax3.quiver(np.zeros(K.shape[0]), np.zeros(K.shape[0]), K[:,0], K[:,1], angles='xy', scale_units='xy', scale=1,color='white')
ax3.text(np.max(K[:,0]), np.max(K[:,1]), '$\Theta$',color='white')
ax3.axis(xmin=-5,ymin=-5,xmax=5,ymax=5)
ax3.set_xlabel('$x$')
ax3.set_title('$S(\mathbf{x}) = \mathcal{F}^{-1}\{ e^{i\Theta\mathbf{x}}\}$')

cbar = plt.colorbar(heatmap,ax=ax3)
cbar.ticklocation='right'

cbar.set_label('Similarity', rotation=270, labelpad=15)
cbar.draw_all()

plt.savefig('figure1_hexssp_encoding.pdf',format='pdf')

plt.show()