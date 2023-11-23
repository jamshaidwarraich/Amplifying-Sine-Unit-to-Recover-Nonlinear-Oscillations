#!/usr/bin/env python
# coding: utf-8

# In[10]:


from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
c0,c1,c2,c3,c4,c5,c6 = 1,2,3,4,5,6,7
def f(u,t):
    return (u[1], (-(c3*u[0]+c4*u[0]**3+c5*u[0]**5+c6*u[0]**7)/(c0+c1*u[0]**2+c2*u[0]**4)))
u0 = [1.0472,0]
ts = np.linspace(0,10,200)
us = odeint(f,u0,ts)
ys = us[:,0]
plt.figure()
plt.plot(ts,ys,'-')
#plt.plot(ts,ys,'ro')
plt.xlabel('Time (t)')
plt.ylabel('Deflection (u)')
plt.title('Nonlinear MEMS')
plt.show()


# In[1]:


import numpy as np
import torch
from neurodiffeq.neurodiffeq import safe_diff as diff
from neurodiffeq.ode import solve, solve_system
from neurodiffeq.solvers import Solver1D
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.conditions import IVP

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


from neurodiffeq.networks import FCNN # fully-connect neural network
import torch.nn as nn                 # PyTorch neural network module
from neurodiffeq.networks import SinActv # sin activation


# In[3]:


#parameters
c0,c1,c2,c3,c4,c5,c6 = 1,2,3,4,5,6,7
nonlinear_MEMS = lambda u, t: [ diff(u, t, order=2) + (c3*u + c4*u**3 + c5*u**5 + c6*u**7)/(c0 + c1*u**2 + c2*u**4) ]
init_val_ho = IVP(t_0=0.0, u_0=1.0472, u_0_prime=0.0)


# In[45]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Specify the network architecture
net_ho = FCNN(
    hidden_units=(128, 128, 128), actv=SinActv
)

# Create a monitor callback
from neurodiffeq.monitors import Monitor1D
monitor_callback = Monitor1D(t_min=0.0, t_max=10, check_every=100).to_callback()

# Create a solver
solver = Solver1D(
    ode_system=nonlinear_MEMS,  
    conditions=[init_val_ho],        
    t_min=0.0,
    t_max=10,
    nets=[net_ho],                   
)

# Fit the solver
solver.fit(max_epochs=6000, callbacks=[monitor_callback])

# Obtain the solution
solution_1 = solver.get_solution()

internals = solver.get_internals()


# In[46]:


internals


# In[47]:


ts = np.linspace(0, 10, 200)
u_net = solution_1(ts, to_numpy=True)

fig = plt.figure(figsize=(10, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts,ys, '.', label='$u$ (LSODA)')
ax1.plot(ts, u_net, label='$u$ (DNN)')

ax1.set_ylabel('Deflection')
ax1.set_xlabel('time')
ax1.set_title('Comparison in case of Sine activation')
ax1.legend(loc='upper right')

ax2.set_title('Error in DNN-based solution from numerical solution')
ax2.plot(ts, u_net-ys, label='$e_s$')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time')
ax2.legend(loc='upper right')


# In[35]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Specify the network architecture
net_ho = FCNN(
    hidden_units=(128,128,128), actv=nn.Tanh
)

# Create a monitor callback
from neurodiffeq.monitors import Monitor1D
monitor_callback = Monitor1D(t_min=0.0, t_max=10, check_every=100).to_callback()

# Create a solver
solver = Solver1D(
    ode_system=nonlinear_MEMS,  
    conditions=[init_val_ho],        
    t_min=0.0,
    t_max=10,
    nets=[net_ho],                   
)

# Fit the solver
solver.fit(max_epochs=35000, callbacks=[monitor_callback])

# Obtain the solution
solution_3 = solver.get_solution()

internals = solver.get_internals()


# In[36]:


internals


# In[37]:


ts = np.linspace(0, 10, 200)
u3_net = solution_3(ts, to_numpy=True)

fig = plt.figure(figsize=(10, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts,ys, '.', label='$u$ (LSODA)')
ax1.plot(ts, u3_net, label='$u$ (DNN)')

ax1.set_ylabel('Deflection')
ax1.set_xlabel('time')
ax1.set_title('Comparison in case of Tanh activation')
ax1.legend(loc='upper right')

ax2.set_title('Error in DNN-based solution from numerical solution')
ax2.plot(ts, u3_net-ys, label='$e_t$')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time')
ax2.legend(loc='upper right')


# In[30]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Specify the network architecture
net_ho = FCNN(
    hidden_units=(128,128,128), actv=nn.Mish
)

# Create a monitor callback
from neurodiffeq.monitors import Monitor1D
monitor_callback = Monitor1D(t_min=0.0, t_max=10, check_every=100).to_callback()

# Create a solver
solver = Solver1D(
    ode_system=nonlinear_MEMS,  
    conditions=[init_val_ho],       
    t_min=0.0,
    t_max=10,
    nets=[net_ho],                   
)

# Fit the solver
solver.fit(max_epochs=20000, callbacks=[monitor_callback])

# Obtain the solution
solution_4 = solver.get_solution()

internals = solver.get_internals()


# In[31]:


internals


# In[32]:


ts = np.linspace(0, 10, 200)
u4_net = solution_4(ts, to_numpy=True)

fig = plt.figure(figsize=(10, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts,ys, '.', label='$u$ (LSODA)')
ax1.plot(ts, u4_net, label='$u$ (DNN)')

ax1.set_ylabel('Deflection')
ax1.set_xlabel('time')
ax1.set_title('Comparison in case of Mish activation')
ax1.legend(loc='upper right')

ax2.set_title('Error in DNN-based solution from numerical solution')
ax2.plot(ts, u4_net-ys, label='$e_m$')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time')
ax2.legend(loc='upper right')


# In[53]:


class gcu(torch.nn.Module):
    def __init__(self):
        super(gcu, self).__init__()
        return
    def forward(self, x):
        return x*torch.cos(x)


# In[48]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Specify the network architecture
net_ho = FCNN(
    hidden_units=(128,128,128), actv=gcu
)

# Create a monitor callback
from neurodiffeq.monitors import Monitor1D
monitor_callback = Monitor1D(t_min=0.0, t_max=10, check_every=100).to_callback()

# Create a solver
solver = Solver1D(
    ode_system=nonlinear_MEMS,  
    conditions=[init_val_ho],        
    t_min=0.0,
    t_max=10,
    nets=[net_ho],                   
)

# Fit the solver
solver.fit(max_epochs=4000, callbacks=[monitor_callback])

# Obtain the solution
solution_6 = solver.get_solution()

internals = solver.get_internals()


# In[50]:


internals


# In[52]:


ts = np.linspace(0, 10, 200)
u6_net = solution_6(ts, to_numpy=True)

fig = plt.figure(figsize=(10, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts,ys, '.', label='$u$ (LSODA)')
ax1.plot(ts, u6_net, label='$u$ (DNN)')

ax1.set_ylabel('Deflection')
ax1.set_xlabel('time')
ax1.set_title('Comparison in case of GCU activation')
ax1.legend(loc='upper right')

ax2.set_title('Error in DNN-based solution from numerical solution')
ax2.plot(ts, u6_net-ys, label='$e_g$')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time')
ax2.legend(loc='upper right')


# In[38]:


class asu(torch.nn.Module):
    def __init__(self):
        super(asu, self).__init__()
        return
    def forward(self, x):
        return x*torch.sin(x)


# In[42]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# Specify the network architecture
net_ho = FCNN(
    hidden_units=(128,128,128), actv=asu
)

# Create a monitor callback
from neurodiffeq.monitors import Monitor1D
monitor_callback = Monitor1D(t_min=0.0, t_max=10, check_every=100).to_callback()

# Create a solver
solver = Solver1D(
    ode_system=nonlinear_MEMS,  
    conditions=[init_val_ho],        
    t_min=0.0,
    t_max=10,
    nets=[net_ho],                   
)

# Fit the solver
solver.fit(max_epochs=2000, callbacks=[monitor_callback])

# Obtain the solution
solution_7 = solver.get_solution()

internals = solver.get_internals()


# In[43]:


internals


# In[44]:


ts = np.linspace(0, 10, 200)
u7_net = solution_7(ts, to_numpy=True)

fig = plt.figure(figsize=(10, 5))
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(ts,ys, '.', label='$u$ (LSODA)')
ax1.plot(ts, u7_net, label='$u$ (DNN)')

ax1.set_ylabel('Deflection')
ax1.set_xlabel('time')
ax1.set_title('Comparison in case of ASU activation')
ax1.legend(loc='upper right')

ax2.set_title('Error in DNN-based solution from numerical solution')
ax2.plot(ts, u7_net-ys, label='$e_a$')
ax2.set_ylabel('Error')
ax2.set_xlabel('Time')
ax2.legend(loc='upper right')

