
import matplotlib.pyplot as plt
plt.ion() #interactive plotting
import numpy as np

from scipy.integrate import odeint

import math 


#Parameters
############

Cm       = 0.025    # membrane capacitance  (nano-farad)
gNa_bar  = 3        # maximal conductance of Na  (micro-siemen)
gDR_bar  = 0.9      # maximal conductance of Kdr (delayed rectifier) (micro-siemen)
gA_bar   = 0.15     # maximal conductance of KA (transient)- micro-siemen
gAHP_bar = 0.15     # maximal conductance of KAHP(Ca dep K ) - (micro-siemen)
gCaL_bar = 0.0015   # maximal conductance of Ca (micro-siemen)
gCaN_bar = 0.002   # maximal conductance of Ca (micro-siemen)
gL       = 0.05     # maximal conductance of leak channel (micro-siemen)


ENa      = 55       # reversal potential of Na+ (milli-volt)
EK       = -94      # reversal potential of K+ (milli-volt)
ESynE    = -10       # reversal potential of (excitatory Synapse) (milli-volt)
ESynI    =-94       # reversal potential of (inhibitory Synapse) (milli-volt)
EL       = -43      # reversal potential of leak channel (milli-volt - (-41.31 for 1Hz  -52.57 for 2Hz  -52.56 for 3Hz))   */ 

v        = 2.5e-4   # Volume of the shell for Ca channel (nano-liter)
Btot     = 0.03     # Total concentration of Calcium (free + bound) milli-molar
K        = 0.001    # kb/kf (ratio of backward to forward rate constatnts (milli-molar)
Cai0     = 5e-5     # Equilibrium Ca2+ concentration (milli-molar)
tauE     = 30       # Time constatnt for excitatory synapse (milli-second)
tauI     = 30       # Time constatnt for inhibitory synapse (milli-second)

F        = 9.648e4  # Parameter for inwardly rectofying K+ current (coulomb/mol)
T        = 308      # Temp - parameter for inwardly rectofying K+ current (kelvin)
KR       = 1.5      # Boltzmann binding constatnt for  cAMP modulaiton
KRcAMP   = 1e-3     # Boltzmann binding constatnt for  cAMP modulaiton (milli-molar)
DRcAMP   = 0.4e-3   # Boltzmann binding constatnt for  cAMP modulaiton (milli-molar)
gR_bar   = (0.18/3) # maximal unmodulated conductance (micro-siemen)
Z        = 2        # Parameter for inwardly rectofying K+ current (coulomb/mol)
RT       = (8314*308) # Joule/Kmol


k_adc    = 0.6e-6   # Kinetic parameter for cAMP modulation (mM/m-sec)
K_mod    = 1.5      # Kinetic parameter for cAMP modulation
K_5HT    = 6e-3     # Kinetic parameter for cAMP modulation (milli-molar)
v_pde    = 2.4e-6   # Kinetic parameter for cAMP modulation (mM/m-sec)
K_pde    = 3e-3     # Kinetic parameter for cAMP modulation (milli-molar)



#Inputs
#######

InE  = 0.012      # Excitatory Synapse conductnace
InI  = 0.000      # Inhibitory Synapse conductance
Iinj = 0.012      # Injected input current 

HT5  = 0.00       # Seretonin concentration


        
# Steady State with IR and EL=-43 and gL=0.05. */
###############################################

y0=np.zeros(17)
y0[0]  = -6.165671106176441e+01 # Membrane volatge Vm
y0[1]  = 1.277236125734123e-02  # activation variable of Na+
y0[2]  = 3.429830012296679e-01  # inactivation variable of Na+
y0[3]  = 2.642252518522899e-02  # activation variable of Kdr
y0[4]  = 4.514268765351059e-01  # activation variable of transient potassium K_A1
y0[5]  = 6.157878435363050e-02  # inactivation variable of transient potassium K_A1
y0[6]  = 2.170679363371657e-01  # activation variable of transient potassium K_A2
y0[7]  = 6.157878220370643e-02  # inactivation variable of transient potassium K_A2
y0[8]  = 1.113929524548006e-01  # activation variable of Calcium dependent potassium K_AHP
y0[9]  = 1.027134762959492e-02  # activation variable of L-type calcium
y0[10] = 0.0                       # m1-activation variable of N-type calcium 
y0[11] = 0.0                     # h1-inactivation variable of N1-type calcium 
y0[12] = 0.0                       # h2-inactivation variable of N2-type calcium 
y0[13] = 5.007132151536936e-05  # Calcium concentration
y0[14] = 0.0                     # pre synaptic excitation
y0[15] = 0.0                      # pre synaptic inhibition
y0[16] = 0.0                      # cyclic AMP modulation
#%%


def tau_pump(x):
    return (17.7*math.exp(x/35))

def PB(x):
    return (Btot/(x+Btot+K))

def ECa(x):
    return (13.27*math.log(4/x))
#%%


    
def PN_model(y,t):
    
#variables
##########
    
    V     = y[0] # Membrane volatge from interg over Vm
    mNa   = y[1] # activation variable of Na+
    hNa   = y[2] # inactivation variable of Na+
    mDR   = y[3] # activation variable of Kdr
    mA1   = y[4] # activation variable of transient potassium K_A1
    hA1   = y[5] # inactivation variable of transient potassium K_A1
    mA2   = y[6] # activation variable of transient potassium K_A2
    hA2   = y[7] # inactivation variable of transient potassium K_A2
    mAHP  = y[8] # activation variable of Calcium dependent potassium K_AHP
    mCaL  = y[9] # activation variable of L-type calcium
    mCaN  = y[10] # m1-activation variable of N-type calcium 
    hCaN1 = y[11] # h1-inactivation variable of N1-type calcium 
    hCaN2 = y[12] # h2-inactivation variable of N2-type calcium  
    Cai   = y[13] # Calcium concentration
    gSynE = y[14] # pre synaptic excitation
    gSynI = y[15] # pre synaptic inhibition
    cAMP  = y[16] # cyclic AMP modulation
            
       
            
       
#Current from channels
######################

    INa   = -gNa_bar*mNa**3*hNa*(ENa-V) #Fast sodium
    IDR   = -gDR_bar*mDR**4*(EK-V)      #Potassium delayed rectifier
    IA    = -gA_bar*( 0.6*mA1**4*hA1 + 0.4*mA2**4*hA2 )*(EK-V) #Transient potassium
    IAHP  = -gAHP_bar*mAHP*mAHP*(EK-V) #Calcium dependent potassium
    FR    = 1   #1+KR/( 1+ math.exp((KRcAMP-cAMP)/DRcAMP) )  #cAMP regulation
    IR    = gR_bar*FR*(V-EK+5.66)/( 1+ math.exp((V-EK-15.3)*Z*F/RT) ) # Inwardly rectifying potassium, KAR
    ICa   = gCaL_bar*mCaL**2*(ECa(Cai)-V) #Calcium-L type - TO EDIT
     
  
    ISynE = gSynE*(ESynE-V) # Excitatory Synapse
    ISynI = gSynI*(ESynI-V) # Inhibitory Synapse
    IL    = gL*(EL-V)       # Leak channel current
        


#Steady state values
####################

    # Fast Sodium, Na_fast */

    miNa = ( (V+38)/(1- math.exp(-(V+38)/5)) ) 
    tmNa = 1/( 0.091*miNa + 0.062*miNa*math.exp(-(V+38)/5) ) 
    miNa = 0.091*miNa*tmNa 
    
    hiNa = 0.016* math.exp(-(V+55)/15) 
    thNa = 1/( hiNa+2.07/(1+ math.exp(-(V-17)/21)) ) 
    hiNa = hiNa*thNa 
    
    gNa  = gNa_bar*mNa**3*hNa 
    
    mNa_dot = (miNa-mNa)/tmNa 
    hNa_dot = (hiNa-hNa)/thNa 


    # Potassium-delayed rectifier, K_DR */

    miDR = ( 0.01*(V+45)/(1- math.exp(-(V+45)/5)) ) 
    tmDR = 1/( miDR+0.17* math.exp(-(V+50)/40) ) 
    miDR = miDR*tmDR 
    
    gDR  = gDR_bar*mDR**4 
    mDR_dot = (miDR-mDR)/tmDR 


    #Transient Potassium-A, K_A */

    miA1 = 1/( 1+ math.exp(-(V+60)/8.5) ) 
    tmA1 = 1/(  math.exp((V+35.82)/19.69) +  math.exp(-(V+79.69)/12.7) + 0.37 ) 
    hiA1 = 1/( 1+ math.exp((V+78)/6) ) 
    thA1 = 1/( 1+ math.exp((V+46.05)/5) +  math.exp(-(V+238.4)/37.45) ) 
    miA2 = 1/( 1+ math.exp(-(V+36)/20) ) 
    tmA2 = tmA1 
    hiA2 = hiA1 
    thA2 = thA1 

    if V > -63: 
        thA1 = 19
    elif V > -73: 
        thA2 = 60 

    
  
    gA = gA_bar*( 0.6*mA1**4*hA1 + 0.4*mA2**4*hA2 ) 
    
    mA1_dot = (miA1-mA1)/tmA1 
    hA1_dot = (hiA1-hA1)/thA1 
    mA2_dot = (miA2-mA2)/tmA2 
    hA2_dot = (hiA2-hA2)/thA2 


    # Calcium-dependent potassium, K_AHP */

    miAHP = 1.25e8*Cai*Cai 
    tmAHP = 1e3/(miAHP+2.5) 
    miAHP = miAHP*1e-3*tmAHP 
    
    gAHP  = gAHP_bar*mAHP*mAHP 
    mAHP_dot = (miAHP-mAHP)/tmAHP 



    # Inwardly rectifying (Anomalous Rectifier), K_AR */

    #FR = 1+KR/( 1+ math.exp((KRcAMP-cAMP)/DRcAMP) ) 
    FR=1 #Turning off the cAMP-modifier
    IR = gR_bar*FR*(V-EK+5.66)/( 1+ math.exp((V-EK-15.3)*Z*F/RT) ) 



    # High-threshold L-type calcium, CaL */

    miCaL = 1.6/( 1+ math.exp(-0.072*(V-5)) ) 
    tmCaL = 1/( miCaL + 0.02*(V-1.31)/( math.exp((V-1.31)/5.36)-1) ) 
    miCaL = miCaL*tmCaL 
    
    gCaL  = gCaL_bar*mCaL*mCaL 
    mCaL_dot = (miCaL-mCaL)/tmCaL 


   

    # Low-threshold N-type calcium, CaN */

    miCaN = 1.0/( 1+ math.exp(-(V+20)/4.5) ) 
    tmCaN = 0.364* math.exp(-(0.042**2)*(V+31)**2) +0.442
  
    hiCaN1 = 1.0/( 1+ math.exp(V+20)/25)  
    thCaN1 = 3.752* math.exp(-(0.0395**2)*(V+30)**2) +0.56
    
    hiCaN2 = 0.2/( 1+ math.exp(-(V+40)/10)) +  1.0/( 1+ math.exp((V+20)/40))
    thCaN2 = 25.2* math.exp(-(0.0275**2)*(V+40)**2) + 8.4

    gCaN  = gCaN_bar*mCaN*(0.55*hCaN1+0.45*hCaN2)  ## TO EDIT
    
    mCaN_dot = (miCaN-mCaN)/tmCaN 
    hCaN1_dot = (hiCaN1-hCaN1)/thCaN1 
    hCaN2_dot = (hiCaN2-hCaN2)/thCaN2 
    
 
    
    
 # cAMP balance in the cell */
    
    #cAMP_dot = k_adc*( 1+K_mod*(HT5/(HT5+K_5HT)) ) - v_pde*(cAMP/(cAMP+K_pde))
    cAMP_dot =0

   

  # Calcium concentration  - with buffer*/
  
    ICaL  = gCaL*(ECa(Cai)-V) #Calcium-L type
    ICaN  = gCaN*(ECa(Cai)-V) #Calcium-L type
    ICa = ICaL + ICaN
  
    Cai_dot = ( ICa*(1-PB(Cai))/(2*F*v) ) + ( (Cai0-Cai)/tau_pump(V) ) # Rate of change of Ca conc


   
    # Synaptic conductances */

    gSynE_dot = (InE-gSynE)/tauE 
    gSynI_dot = (InI-gSynI)/tauI


# Membrane potential */

    V_dot = gNa*(ENa-V) + (gDR+gA+gAHP)*(EK-V) - (IR) + (ICa) + gL*(EL-V) + gSynE*(ESynE-V) + 0*gSynI*(ESynI-V) + Iinj 

    #gDR+gA+gAHP
    V_dot = V_dot/Cm 




#Differential equations
#######################

    dy = np.zeros(17)
    
    
    dy[0] = (1e3*V_dot)
    dy[1] = (1e3*mNa_dot)
    dy[2] = (1e3*hNa_dot)
    dy[3] = (1e3*mDR_dot)
    dy[4] = (1e3*mA1_dot)
    dy[5] = (1e3*hA1_dot)
    dy[6] = (1e3*mA2_dot)
    dy[7] = (1e3*hA2_dot)
    dy[8] = (1e3*mAHP_dot)
    dy[9] = (1e3*mCaL_dot)
    dy[10] = (1e3*mCaN_dot)   
    dy[11] = (1e3*hCaN1_dot) 
    dy[12] = (1e3*hCaN2_dot) 
    dy[13] = (1e3*Cai_dot)
    dy[14] = (1e3*gSynE_dot)
    dy[15] = (1e3*gSynI_dot)
    dy[16] = (0*1e3*cAMP_dot)

    return dy



#ODE CALCULAITON AND PLOTTING
###############################

# initial time
ti = 0
# time step (0.0005 s)
dt = 0.5*10**(-3)
# final time 
tf = 1 # 800 for step response

tspan = np.arange(ti,tf,dt)


y1 = abs(y0)
Vy = odeint(PN_model, y0, tspan)


V     = Vy[:,0]
mNa   = Vy[:,1]
hNa   = Vy[:,2]
mDR   = Vy[:,3]
mA1   = Vy[:,4]
hA1   = Vy[:,5]
mA2   = Vy[:,6]
hA2   = Vy[:,7]
mAHP  = Vy[:,8]
mCaL  = Vy[:,9]
mCaN  = Vy[:,10]
hCaN1 = Vy[:,11]
hCaN2 = Vy[:,12]
Cai   = Vy[:,13]
gSynE = Vy[:,14]
gSynI = Vy[:,15]
cAMP  = Vy[:,16]

#%%


fig = plt.figure()
plt.plot(tspan,V)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Voltage (mV)')
fig.savefig('example_plot_python.png') 






  

