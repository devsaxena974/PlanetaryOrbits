import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from astroquery.jplhorizons import Horizons
from datetime import date, timedelta


def fgravity(t, m, q, g):
  qprime = []
  qprime.append(q[30:]) # Derivative of positions is velocities
  new_q1prime = []
  for j in range(0, 30, 3): # Loops through the bodies to calculate forces on
    accelx, accely, accelz = 0, 0, 0 
    for k in range(0, 30, 3): # Loops through the bodies to calculate forces from
      if j == k:
        continue
      elif m[j//3] == 0:
        continue
      else:
        diffx, diffy, diffz = q[k] - q[j], q[k+1] - q[j+1], q[k+2] - q[j+2] # Difference in position between the target body and the affecting body
        mag = math.sqrt(diffx**2 + diffy**2 + diffz**2) # Magnitude of the "difference" vector
        gravx = (g*m[j//3]*m[k//3])/(mag**3)*diffx # Calculation of force of gravity
        gravy = (g*m[j//3]*m[k//3])/(mag**3)*diffy
        gravz = (g*m[j//3]*m[k//3])/(mag**3)*diffz
        accelx += gravx/m[j//3]
        accely += gravy/m[j//3]
        accelz += gravz/m[j//3]
    new_q1prime.append(accelx)
    new_q1prime.append(accely)
    new_q1prime.append(accelz)
  qprime.append(new_q1prime)
  qprime = np.array(qprime)
  return qprime.flatten()

def rk4(f,t,q0):
  # create an array q0 from the q0 passed in
  q0=np.asarray(q0)
  t=np.asarray(t)
  # create an output array q
  # needs to have len(t) rows and len(q) columns
  q=np.zeros( (len(t),len(q0)) )
  q[0]=q0
  # iterate over times t_i and use forward Euler to compute y_i+1
  for i in range(0,len(t)-1):
    h=t[i+1]-t[i]
    k1=h*f(q[i],t[i])
    k2=h*f(q[i]+k1/2, t[i]+h/2)
    k3=h*f(q[i]+k2/2, t[i]+h/2)
    k4=h*f(q[i]+k3, t[i]+h)
    q[i+1]=q[i]+1/6 *(k1+2*k2+2*k3+k4)
  return q

def get_positions(q):

  positions = {
      "sun": [[], [], []],
      "mercury": [[], [], []],
      "venus": [[], [], []],
      "earth": [[], [], []],
      "moon": [[], [], []],
      "mars": [[], [], []],
      "jupiter": [[], [], []],
      "saturn": [[], [], []],
      "uranus": [[], [], []],
      "neptune": [[], [], []]
    }
  
  bodies = list(positions.keys())

  for i in range(len(q)):
    reshaped = np.reshape(q[i], (20, 3))
    for a in range(0, 10):
      for b in range(0, 3):
        positions[bodies[a]][b].append(reshaped[a][b])


  return positions

def get_ephemeris():
  cur_date = str(date.today())
  yesterday = str(date.today() - timedelta(days=1))
  # append objects for each body
  bodycodes = ['10', '199', '299', '399', '301', '499', '599', '699', '799', '899']
  body_objects = []
  for code in bodycodes:
      obj = Horizons(id=code, location='500@0', epochs={'start': yesterday,
                          'stop': cur_date,
                          'step': '1d'})
      
      body_objects.append(obj)

  # create q array
  q = []
  for i in range(len(body_objects)):
      # append x,y,z, coords
      table = body_objects[i].vectors()
      q.append(table['x'][1])
      q.append(table['y'][1])
      q.append(table['z'][1])
      
  for j in range(len(body_objects)):
      table = body_objects[j].vectors()
      # append vx. vy. vz initial velocities
      q.append(table['vx'][1])
      q.append(table['vy'][1])
      q.append(table['vz'][1])

  return np.array(q)


def main():
    sun_mass = 1988500e24
    mercury_mass = 3.302e23
    venus_mass = 48.685e23
    earth_mass = 5.97219e24
    moon_mass = 7.349e22
    mars_mass = 6.4171e23
    jupiter_mass = 189818722e19
    saturn_mass = 5.6834e26
    uranus_mass = 86.813e24
    neptune_mass = 102.409e24

    m = np.array([sun_mass, mercury_mass, venus_mass, earth_mass, moon_mass, mars_mass, jupiter_mass, saturn_mass, uranus_mass, neptune_mass])

    q = get_ephemeris()

    G_conv = 1.488e-34

    fgravity_lambda = lambda q,t: fgravity(t, m, q, G_conv)

    # Begin input code here:
    print("Enter the period for which you would like to see the orbit (LIMIT: 34,280):")
    period = input("Period (in Earth days): ")

    if int(period) > 34280:
      raise Exception("Please enter a period of days less than or equal to 34,280")

    t = np.linspace(0, int(period), num=10000)
    result = rk4(fgravity_lambda, t, q)

    final_positions = get_positions(result)


    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot3D(0.0, 0.0, 'oy', label="Sun", markersize=10)
    #plot final positions of each planet
    ax.plot3D(final_positions['mercury'][0][9999], final_positions['mercury'][1][9999], final_positions["mercury"][2][9999], 'ko')
    ax.plot3D(final_positions['venus'][0][9999], final_positions['venus'][1][9999], final_positions["venus"][2][9999], 'ko')
    ax.plot3D(final_positions['earth'][0][9999], final_positions['earth'][1][9999], final_positions["earth"][2][9999], 'ko')
    ax.plot3D(final_positions['mars'][0][9999], final_positions['mars'][1][9999], final_positions["mars"][2][9999], 'ko')
    ax.plot3D(final_positions['jupiter'][0][9999], final_positions['jupiter'][1][9999], final_positions["jupiter"][2][9999], 'ko')
    ax.plot3D(final_positions['saturn'][0][9999], final_positions['saturn'][1][9999], final_positions["saturn"][2][9999], 'ko')
    ax.plot3D(final_positions['uranus'][0][9999], final_positions['uranus'][1][9999], final_positions["uranus"][2][9999], 'ko')
    ax.plot3D(final_positions['neptune'][0][9999], final_positions['neptune'][1][9999], final_positions["neptune"][2][9999], 'ko')
    #plot orbital path over given perioD
    ax.plot3D(final_positions["mercury"][0], final_positions["mercury"][1], final_positions["mercury"][2], 'y-', label="Mercury")
    ax.plot3D(final_positions["venus"][0], final_positions["venus"][1], final_positions["venus"][2], 'b--', label="Venus")
    ax.plot3D(final_positions["earth"][0], final_positions["earth"][1], final_positions["earth"][2], 'g--', label="Earth")
    ax.plot3D(final_positions["mars"][0], final_positions["mars"][1], final_positions["mars"][2], '#80393C', label="Mars")
    ax.plot3D(final_positions["jupiter"][0], final_positions["jupiter"][1], final_positions["jupiter"][2], 'm--', label="Jupiter")
    ax.plot3D(final_positions["saturn"][0], final_positions["saturn"][1], final_positions["saturn"][2], 'c--', label="Saturn")
    ax.plot3D(final_positions["uranus"][0], final_positions["uranus"][1], final_positions["uranus"][2], 'y--', label="Uranus")
    ax.plot3D(final_positions["neptune"][0], final_positions["neptune"][1], final_positions["neptune"][2], 'b--', label="Neptune")
    #plt.xlim([-0.5, 0.5])
    #plt.ylim([-0.5, 0.5])
    plt.title("Planetary Orbits")
    plt.xlabel("X Axis Position (au)")
    plt.ylabel("Y Axis Position (au)")
    plt.legend()
    plt.show()

    return "Successful Graph!"

if __name__ == '__main__':
  print(main())