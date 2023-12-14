## Synthetic datasets

We render our synthetic scenes with multiple spectral illuminants to generate spectral images. To acquire spectral illuminants, we divide the wavelength range of the light source spectrum in Mitsuba from 360nm to 830nm into 11 adjacent intervals.

As for the first 10 spectral intervals, the relative power at the midpoint is set to a certain value. The maximum power of the last interval (760nm to 830nm) is set on 780nm because 780nm is a commonly used termination value of visible light. Simultaneously, the power of two endpoints for each interval is set to 0. After a simple linear interpolation, we obtain 11 curves as the spectral power distributions of the spectral illuminants.

In addition, we use the CIE standard illuminant D65 as the default white light source for scenes in our dataset. The D65 light source is an artificial light source that simulates daylight, and its emission spectrum conforms to the average midday light of European and Pacific countries. 


## Real-world datasets

We utilize a camera and 8 color absorbers whose center wavelengths range from 400nm to 750nm with the interval of 50nm to capture the real-world scene. Different color absorbers are covered to the camera lens to obtain the spectral images.
