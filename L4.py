import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

#Get initialization values that are specific to the movie
print('Resolution in um/pixel:')
x = float(input())
print('FirstImage:')
y=int(input())
print('LastImage:')
z=int(input())

stack=z-y #is also important for the frame rate, the total time component of the data
dt=[0]*(stack-1) #initialization of the time array
Rdt=[0]*(stack-1) #initialization of the Radius at every time array
time=0 #time calculation
l=0 #Movie counter
while l<stack-1:
    time = float((l + 1)) * 0.00116 #in s
    image=str(y+l) #what image am I at?

	#Now I need to iterate through the image stack
	#convert to a string and add to file name

    image1 = cv2.imread('trial3/Basler_acA1920-155umMED__40015329__20200518_154445380_'+image+'.tiff')
    #cv2.imshow('image',image1)

	#threshold the most important information, converts everythin to a binary image, either 0 or 255 (white)
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 63, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Threshold', thresh1)
	# you can access a pixel value by its row and column coordinates. FOr thresh1, this is 200x200
	#we can also do ROI: ball = img[280:340, 330:390]
	#px=thresh1[199,0]
	#print(px)


	#print(img.shape)
	#print(type(img.shape))
    h=img.shape[1]
    w=img.shape[0]

    h=h-35

    #initializations that loop through the image to search for information
    k=0 # counter
    Dtotal=0
    D=[0]*(h-1) #will make an array of diameters as a placeholder

    #Here I have to go through the image, starting from the bottom, and get to the 0 pixel value
    #this has to happen from both sides
    while k<(h-1):#note!I have to start from the bottom!
        j = 0
        Dcount = 0
        w1=0 #reinitialize at every height
        w2=0
        #I need a new search criteria for the while loop that includes only the diameter
        #Include a % higher than the previous value

        if k<20 or D[k-1]<=D[k-18]*1.5: #this statement can definitley be more elegant!
            while j<w-1:

                D_search=thresh1[h-k-1,j]
                if D_search<70 and w1==0:#the polymer diameter start, should be stored as a width value. I also have to go from the other direction
                    w1=j #the first D parameter
                #I also have to go the other direction
                D_search2=thresh1[h-k-1,h-j-1]
                if D_search2<70 and w2==0:
                    w2=h-j-1
                Dcount=w2-w1 #this might be wrong if it never finds a pixel?

                j=j+1

            D[k]=Dcount #needs to be separated still
            k=k+1
        else:
            break
        #else truncate D to its final form

    Dfinal=np.trim_zeros(D)
    #print(Dfinal)
    ROI_h2=np.size(Dfinal)
    #print(ROI_h2)
    polymer = thresh1[(h-ROI_h2):h, 0:199]
    cv2.imshow('resized', polymer)



	#Now I have to convert the dimentions to real values, maybe even get R

    AverageR=np.mean(Dfinal)/2
    R_um=AverageR/x

    #print('um R',R_um)


    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()

    Rdt[l]=R_um
    dt[l]=time
    l=l+1

#In the end, I need to solve for L= R~A exp (-t/(3L)) Therefore I have to

def exponential(x, a, k,):
    return a*np.exp(x*k)

popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, dt, Rdt, p0=[1,-100])

L=-(1/(3*popt_exponential[1]))

print("L!", L)

#Need to create a new graph that has the fit and the new line
f=0
fit=[0]*(stack-1)
while f<stack-1:
	fit[f]=popt_exponential[0]*np.exp(dt[f]*popt_exponential[1])
	f=f+1

Title="L = "+str(L)+"!"

plt.plot(dt, Rdt)
plt.plot(dt,fit, '--')
plt.xlabel('Time (s)')
plt.ylabel('R (um)')
plt.title(Title)
plt.show()

#can also probaly include the error on the fit