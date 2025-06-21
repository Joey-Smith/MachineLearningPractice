import numpy
from scipy import stats


speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

mean = numpy.mean(speed)
median = numpy.median(speed)
mode = stats.mode(speed)

print("Speeds: " + speed.__str__())

print()

print("Mean: " + str(mean))
print("Median: " + str(median))
print(f"Mode: {mode.mode}" + f" Count: {mode.count}")
print("Standard Deviation: " + str(numpy.std(speed)))

# Variance is the square of the standard deviation
# Variance is a measure of how far a set of numbers are spread out from their average value
print("Variance: " + str(numpy.var(speed)))

print()

print("Calculating Variance")

for x in speed:
    print(f"{x} - {mean} = {x - mean}")
    
print()

for x in speed:
    print(f"({x - mean}){chr(178)} = {(x - mean) ** 2}")
    
print()

