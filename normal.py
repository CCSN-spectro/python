from scipy.stats import norm
import scipy.integrate as integrate
import scipy.special as special

norm.pdf(0)
integrate.quad(lambda x: 2*norm.pdf(x), 0, 3)

p=0.997
