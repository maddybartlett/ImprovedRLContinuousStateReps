import scipy.special
from scipy.special import log_softmax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import nengo
import math


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

##Softmax Function used for selecting next action
def softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    filtered_x = np.nan_to_num(x-x.max()) 
    return np.exp(log_softmax(filtered_x,axis=axis))

def rend(env):
    return plt.imshow(env.render())

def save_gifs(figure, images, trial, directory):
    ani = animation.ArtistAnimation(figure, images, interval=50, blit=True,
                                    repeat_delay=1e5)
    writergif = animation.PillowWriter(fps=10)
    ani.save(directory+'trial%s.gif' % (trial), writer=writergif)

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
    
def get_ac_output(net, env, n_grid=30, n_ranges=(6,6,4)):
    ''''''
    if "OneHot" in str(net.representation):
        x = n_ranges[0]
        y = n_ranges[1]
        z = n_ranges[2]
        X, Y, Z = np.meshgrid(np.arange(x), np.arange(y), np.arange(z))
    else:
        x = np.linspace(0,env.width,n_grid)
        y = np.linspace(0,env.height,n_grid)
        z = np.linspace(0,360,n_grid)
        ## create coordinate matrix from the dimensions of the state space
        X, Y, Z = np.meshgrid(x,y,z)

    #X, Y = np.meshgrid(np.arange(50), np.arange(50))
    ## array of coordinates in state space
    pts = np.array([X, Y, Z])
    #pts = np.array([X, Y])

    ## flatten into 2D array
    pts = pts.reshape(3, -1)
    ## translate array into chosen state representation of entire state space
    SSPs = [net.representation.map(x).copy() for x in pts.T]

    ## if using ensemble, calculate the tuning curves of the ensemble
    if net.state_neurons is not None:
        _, A = nengo.utils.ensemble.tuning_curves(net.ensemble, net.sim, inputs=SSPs)
        ## reshape the activities of the ensemble neurons 
        if "OneHot" in str(net.representation):
            A = A.reshape((x,y,z,-1))
        else:
            A = A.reshape((len(x),len(y),len(z),-1))

    ## if not using ensemble, just reshape the state representationof the entire state space
    else:
        try:
            A = np.array(SSPs).reshape((len(x),len(y),len(z),-1))
        except TypeError: 
            A = np.array(SSPs).reshape((x,y,z,-1))

    ## get weight matrix
    w = net.sim.signals[net.sim.model.sig[net.rule.output]['_state_w']]

    ## Calculate policy by calculating dot product of state space and weight matrix
    V = A.dot(w.T)

    ## return policy
    return softmax(V[:,:,:,1:], axis=-1), V[:,:,:,0]  ,[X,Y,Z]

def plot_policy(policy, pts, values=None, plot_type='vector',ax=None,vmin='auto',vmax='auto',cmap='viridis'):
    # plot_type options: vector, stream
    if ax is None:
        fig,(ax,cax) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[50,1]})
        
    policy = np.mean(policy, axis=2) # avg over direction
    values = np.mean(values, axis=2)
    
    if vmin == 'auto':
        vmin = round(values.min(),2)
    if vmax == 'auto':
        vmax = round(values.max(),2)
    
    if values is not None:
        im = ax.contourf(pts[0][:,:,0], pts[1][:,:,0], 
                        values, levels = np.linspace(vmin,vmax,10),
                        vmin=vmin,vmax=vmax,cmap=cmap)
        
        ticks = [vmin,round((vmax+vmin)/2,2),vmax]
        cbar = fig.colorbar(im, cax=cax,ticks=ticks,orientation='vertical')  
        cbar.set_label('Value',va='top',ha='left',rotation=90,in_layout=True)
        cbar.ax.set_yticklabels(ticks)

    if plot_type=='vector':
        ax.quiver(pts[0][:,:,0], pts[1][:,:,0], 
              policy[:,:,2]-policy[:,:,3], # E - W
                 policy[:,:,0]-policy[:,:,1], color='k') # N-S # or is it?
    elif plot_type=='stream':
        ax.streamplot(pts[0][:,:,0], pts[1][:,:,0], 
              policy[:,:,2]-policy[:,:,3], # E - W
                      policy[:,:,1]-policy[:,:,0], color='k') 
    else:
        print('Plot type not supported')
    return ax

def plot_table(P):
    plt.figure(figsize=(14,14))
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1)
            plt.imshow(P[:,:,j,i], vmin=0, vmax=1)
            plt.colorbar()
            if j == 0:
                plt.ylabel(['value', 'turn left', 'turn right', 'forward'][i])
            if i == 0:
                plt.title(['east', 'south', 'west', 'north'][j])
    plt.show()

